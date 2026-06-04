from __future__ import annotations

import torch
import torch.nn.functional as F


def pad_to_hop(x: torch.Tensor, hop_length: int) -> torch.Tensor:
    """Right-pad a batch of signals so length is divisible by hop_length."""
    if x.ndim != 2:
        raise ValueError(f"Expected (B, N), got {tuple(x.shape)}")
    remainder = x.shape[-1] % hop_length
    if remainder == 0:
        return x
    return F.pad(x, (0, hop_length - remainder))


def make_x_frames(x: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    """Create overlap-save far-end frames of length M.

    Frame k contains M samples and advances by R=hop_length. We left-pad with
    M-R zeros so the first output segment corresponds to the first R samples.
    """
    x = pad_to_hop(x, hop_length)
    x_pad = F.pad(x, (frame_length - hop_length, 0))
    frames = x_pad.unfold(dimension=-1, size=frame_length, step=hop_length)
    return frames.contiguous()


def make_r_frames(x: torch.Tensor, hop_length: int) -> torch.Tensor:
    """Split waveform into non-overlapping R-sample frames."""
    x = pad_to_hop(x, hop_length)
    frames = x.unfold(dimension=-1, size=hop_length, step=hop_length)
    return frames.contiguous()


def pad_r_to_m(frame_r: torch.Tensor, frame_length: int) -> torch.Tensor:
    """Place an R-sample frame in the last R positions of an M-sample vector."""
    if frame_r.ndim != 2:
        raise ValueError(f"Expected (B, R), got {tuple(frame_r.shape)}")
    batch, r = frame_r.shape
    if r > frame_length:
        raise ValueError("R cannot be greater than frame_length")
    out = frame_r.new_zeros(batch, frame_length)
    out[:, -r:] = frame_r
    return out


def positive_frequency_magnitude(X_full: torch.Tensor, feature_bins: int | None = None) -> torch.Tensor:
    """Return one-sided magnitude features for the TNN.

    For frame length M, this repo uses F=M/2 features to match the paper's
    reported fcTNN count 2*(M/2)=M.
    """
    if not torch.is_complex(X_full):
        raise TypeError("X_full must be complex")
    m = X_full.shape[-1]
    f = feature_bins or (m // 2)
    return X_full.abs()[..., :f]


def bins_to_full_spectrum(mu_bins: torch.Tensor, frame_length: int) -> torch.Tensor:
    """Mirror one-sided stepsizes to the full FFT length used by FRLS."""
    if mu_bins.ndim != 2:
        raise ValueError(f"Expected (B, F), got {tuple(mu_bins.shape)}")
    b, f = mu_bins.shape
    expected = frame_length // 2
    if f != expected:
        raise ValueError(f"Expected F=M/2={expected}, got {f}")
    full = mu_bins.new_empty(b, frame_length)
    full[:, :f] = mu_bins
    full[:, f:] = torch.flip(mu_bins, dims=(-1,))
    return full
