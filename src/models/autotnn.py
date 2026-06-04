from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ChannelNorm1d(nn.Module):
    """Non-trainable normalization across |X|/|Y| channels."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)


def _match_length(x: torch.Tensor, length: int) -> torch.Tensor:
    current = x.shape[-1]
    if current == length:
        return x
    if current > length:
        return x[..., :length]
    return F.pad(x, (0, length - current))


class AutoTNN(nn.Module):
    """Autoencoder-style TNN baseline.

    Input: |X(k)| and |Y(k)|, each (B, F).
    Output: frequency-dependent stepsize vector (B, F).
    """

    uses_sequence = False

    def __init__(self, feature_bins: int):
        super().__init__()
        self.feature_bins = int(feature_bins)
        self.norm = ChannelNorm1d()

        self.conv3 = nn.Sequential(
            nn.Conv1d(2, 4, kernel_size=5, stride=4, padding=2),
            nn.PReLU(num_parameters=1),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=5, stride=4, padding=2),
            nn.PReLU(num_parameters=1),
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose1d(16, 4, kernel_size=5, stride=4, padding=2, output_padding=3),
            nn.PReLU(num_parameters=1),
        )
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose1d(8, 2, kernel_size=5, stride=4, padding=2, output_padding=3),
            nn.PReLU(num_parameters=1),
        )
        self.conv7 = nn.Conv1d(4, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, X_mag: torch.Tensor, Y_mag: torch.Tensor) -> torch.Tensor:
        if X_mag.ndim != 2 or Y_mag.ndim != 2:
            raise ValueError("AutoTNN expects X_mag and Y_mag with shape (B, F)")
        z = torch.stack([X_mag, Y_mag], dim=1)
        z_norm = self.norm(z)

        skip_input = z_norm
        e1 = self.conv3(z_norm)
        skip_e1 = e1
        e2 = self.conv4(e1)

        d1 = self.deconv5(e2)
        d1 = _match_length(d1, skip_e1.shape[-1])
        d1 = torch.cat([d1, skip_e1], dim=1)

        d2 = self.deconv6(d1)
        d2 = _match_length(d2, skip_input.shape[-1])
        out = torch.cat([d2, skip_input], dim=1)

        logits = self.conv7(out).squeeze(1)
        return torch.sigmoid(logits)
