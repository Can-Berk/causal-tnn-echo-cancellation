from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ChannelNorm2d(nn.Module):
    """Non-trainable normalization across the |X|/|Y| channels.

    Input/output shape: (B, C=2, T, F).
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)


class CausalConv2dFreqStride(nn.Module):
    """Conv2D that is causal along time and strided only along frequency.

    Time axis is preserved. Frequency axis can be downsampled with stride_freq.
    This is the 2D analogue of the Conv1D frequency layers in autoTNN.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_time: int,
        kernel_freq: int,
        stride_freq: int = 1,
        dilation_time: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_time = int(kernel_time)
        self.kernel_freq = int(kernel_freq)
        self.stride_freq = int(stride_freq)
        self.dilation_time = int(dilation_time)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(self.kernel_time, self.kernel_freq),
            stride=(1, self.stride_freq),
            dilation=(self.dilation_time, 1),
            padding=(0, 0),
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.pad order for 4D tensors is (left_F, right_F, left_T, right_T).
        pad_time_left = (self.kernel_time - 1) * self.dilation_time    # 3 - 1 = 2
        pad_freq_total = self.kernel_freq - 1
        pad_freq_left = pad_freq_total // 2
        pad_freq_right = pad_freq_total - pad_freq_left
        x = F.pad(x, (pad_freq_left, pad_freq_right, pad_time_left, 0))    # The last 0 for no future padding.
        return self.conv(x)


class FreqDeconv2d(nn.Module):
    """Transposed Conv2D that upsamples frequency only.

    The time axis is not upsampled and no future frames are introduced. Temporal
    context has already been handled by the causal encoder convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_freq: int,
        stride_freq: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_freq),
            stride=(1, stride_freq),
            padding=(0, kernel_freq // 2),
            output_padding=(0, stride_freq - 1),
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


def _match_time_freq(x: torch.Tensor, target_t: int, target_f: int) -> torch.Tensor:
    """Crop or right-pad a (B, C, T, F) tensor to a target time/frequency size."""
    t, f = x.shape[-2], x.shape[-1]
    if t > target_t:
        x = x[..., :target_t, :]
    elif t < target_t:
        x = F.pad(x, (0, 0, 0, target_t - t))

    f = x.shape[-1]
    if f > target_f:
        x = x[..., :target_f]
    elif f < target_f:
        x = F.pad(x, (0, target_f - f, 0, 0))
    return x


class CausalConv2DAutoTNN(nn.Module):
    """Causal Conv2D version of the Table-2 autoTNN architecture.

    This model keeps the autoTNN encoder-decoder idea but expands the convolutions
    from frequency-only Conv1D to causal time-frequency Conv2D.

    Input:
        X_mag_seq, Y_mag_seq: (B, T, F), where T contains past frames plus the
        current frame. No future frames are used.

    Internal layout:
        stack channels -> (B, 2, T, F)
        norm
        causal Conv2D encoder, stride frequency 4:  (B, 2,  T, F)   -> (B, 4,  T, F/4)
        causal Conv2D encoder, stride frequency 4:  (B, 4,  T, F/4) -> (B, 16, T, F/16)
        frequency DeConv2D decoder:                 (B, 16, T, F/16)-> (B, 4, T, F/4)
        skip concat with first encoder output:      (B, 8,  T, F/4)
        frequency DeConv2D decoder:                 (B, 8,  T, F/4) -> (B, 2, T, F)
        skip concat with normalized input:          (B, 4,  T, F)
        final causal/linear Conv2D:                 (B, 1,  T, F)
        take current frame and sigmoid:             (B, F)
    """

    uses_sequence = True

    def __init__(
        self,
        feature_bins: int,
        context_frames: int = 8,
        kernel_time: int = 3,
        kernel_freq: int = 5,
        bottleneck_channels: int = 16,
    ):
        super().__init__()
        self.feature_bins = int(feature_bins)
        self.context_frames = int(context_frames)
        kt = int(kernel_time)
        kf = int(kernel_freq)
        bottleneck = int(bottleneck_channels)

        if bottleneck != 16:
            # Kept configurable, but Table-2-like default is 16 channels.
            pass

        self.norm = ChannelNorm2d()

        self.enc1 = nn.Sequential(
            CausalConv2dFreqStride(2, 4, kernel_time=kt, kernel_freq=kf, stride_freq=4),
            nn.PReLU(num_parameters=1),
        )
        self.enc2 = nn.Sequential(
            CausalConv2dFreqStride(4, bottleneck, kernel_time=kt, kernel_freq=kf, stride_freq=4),
            nn.PReLU(num_parameters=1),
        )
        self.dec1 = nn.Sequential(
            FreqDeconv2d(bottleneck, 4, kernel_freq=kf, stride_freq=4),
            nn.PReLU(num_parameters=1),
        )
        self.dec2 = nn.Sequential(
            FreqDeconv2d(8, 2, kernel_freq=kf, stride_freq=4),
            nn.PReLU(num_parameters=1),
        )
        # Linear final projection, analogous to Table 2's final Conv layer.
        self.final = CausalConv2dFreqStride(
            4,
            1,
            kernel_time=self.context_frames,
            kernel_freq=3,
            stride_freq=1,
            bias=True,
        )

    def forward(self, X_mag_seq: torch.Tensor, Y_mag_seq: torch.Tensor) -> torch.Tensor:
        if X_mag_seq.ndim != 3 or Y_mag_seq.ndim != 3:
            raise ValueError(
                "CausalConv2DAutoTNN expects X_mag_seq and Y_mag_seq with shape (B, T, F)"
            )
        if X_mag_seq.shape != Y_mag_seq.shape:
            raise ValueError("X_mag_seq and Y_mag_seq must have the same shape")

        z = torch.stack([X_mag_seq, Y_mag_seq], dim=1)  # (B, 2, T, F)
        z_norm = self.norm(z)
        target_t, target_f = z_norm.shape[-2], z_norm.shape[-1]

        skip_final = z_norm
        e1 = self.enc1(z_norm)                         # (B, 4, T, F/4)
        skip_dec = e1
        e2 = self.enc2(e1)                             # (B, 16, T, F/16)

        d1 = self.dec1(e2)                             # (B, 4, T, ~F/4)
        d1 = _match_time_freq(d1, skip_dec.shape[-2], skip_dec.shape[-1])
        d1 = torch.cat([d1, skip_dec], dim=1)           # (B, 8, T, F/4)

        d2 = self.dec2(d1)                             # (B, 2, T, ~F)
        d2 = _match_time_freq(d2, target_t, target_f)
        out = torch.cat([d2, skip_final], dim=1)        # (B, 4, T, F)

        logits = self.final(out)                       # (B, 1, T, F)
        logits = _match_time_freq(logits, target_t, target_f)
        current_logits = logits[:, 0, -1, :]           # current frame only
        return torch.sigmoid(current_logits)
