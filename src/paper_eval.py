from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch

from .features import (
    bins_to_full_spectrum,
    make_r_frames,
    make_x_frames,
    pad_r_to_m,
    positive_frequency_magnitude,
)
from .frls import FRLSFilter


@dataclass
class AECOutput:
    d_hat: torch.Tensor
    d_target: torch.Tensor


def _predict_mu_bins(
    model: Optional[torch.nn.Module],
    X_mag: torch.Tensor,
    Y_mag: torch.Tensor,
    x_hist: deque[torch.Tensor],
    y_hist: deque[torch.Tensor],
    context_frames: int,
    fixed_mu: Optional[float] = None,
) -> torch.Tensor:
    if fixed_mu is not None:
        return torch.full_like(X_mag, float(fixed_mu))
    if model is None:
        raise ValueError("model cannot be None unless fixed_mu is provided")
    if getattr(model, "uses_sequence", False):
        x_hist.append(X_mag)
        y_hist.append(Y_mag)
        while len(x_hist) < context_frames:
            x_hist.appendleft(torch.zeros_like(X_mag))
            y_hist.appendleft(torch.zeros_like(Y_mag))
        X_seq = torch.stack(list(x_hist), dim=1)
        Y_seq = torch.stack(list(y_hist), dim=1)
        return model(X_seq, Y_seq)
    return model(X_mag, Y_mag)


def run_aec_signals(
    model: Optional[torch.nn.Module],
    x: torch.Tensor,
    y: torch.Tensor,
    d: torch.Tensor,
    frls: FRLSFilter,
    device: torch.device,
    context_frames: int = 1,
    fixed_mu: Optional[float] = None,
) -> AECOutput:
    """Run one full utterance through TNN-FRLS or fixed-mu FRLS.

    Args:
        model: trained TNN model, or None when fixed_mu is used.
        x: far-end waveform, shape (N,)
        y: microphone waveform, shape (N,)
        d: clean echo target waveform, shape (N,)
        frls: FRLSFilter module
        fixed_mu: if set, bypass model and use constant stepsize.
    """
    n = min(x.numel(), y.numel(), d.numel())
    x = x[:n].unsqueeze(0).to(device)
    y = y[:n].unsqueeze(0).to(device)
    d = d[:n].unsqueeze(0).to(device)

    x_frames = make_x_frames(x, frls.frame_length, frls.hop_length)
    y_frames = make_r_frames(y, frls.hop_length)
    d_frames = make_r_frames(d, frls.hop_length)
    num_frames = min(x_frames.shape[1], y_frames.shape[1], d_frames.shape[1])
    x_frames = x_frames[:, :num_frames]
    y_frames = y_frames[:, :num_frames]
    d_frames = d_frames[:, :num_frames]

    state = frls.initial_state(batch_size=1, device=device)
    x_hist: deque[torch.Tensor] = deque(maxlen=context_frames)
    y_hist: deque[torch.Tensor] = deque(maxlen=context_frames)
    d_hat_frames = []

    if model is not None:
        model.eval()

    for k in range(num_frames):
        xk = x_frames[:, k, :]
        yk = y_frames[:, k, :]

        X = torch.fft.fft(xk, n=frls.frame_length, dim=-1)
        Y = torch.fft.fft(pad_r_to_m(yk, frls.frame_length), n=frls.frame_length, dim=-1)
        X_mag = positive_frequency_magnitude(X)
        Y_mag = positive_frequency_magnitude(Y)

        mu_bins = _predict_mu_bins(
            model=model,
            X_mag=X_mag,
            Y_mag=Y_mag,
            x_hist=x_hist,
            y_hist=y_hist,
            context_frames=context_frames,
            fixed_mu=fixed_mu,
        )
        mu_full = bins_to_full_spectrum(mu_bins, frls.frame_length)
        d_hat_k, state = frls(xk, yk, mu_full, state)
        d_hat_frames.append(d_hat_k)

    d_hat = torch.cat(d_hat_frames, dim=-1).squeeze(0)
    d_target = d_frames.reshape(1, -1).squeeze(0)[: d_hat.shape[-1]]
    return AECOutput(d_hat=d_hat.detach().cpu(), d_target=d_target.detach().cpu())


def frame_energy(signal: torch.Tensor, hop_length: int) -> torch.Tensor:
    frames = make_r_frames(signal.unsqueeze(0), hop_length).squeeze(0)
    return torch.mean(frames**2, dim=-1)


def moving_average(x: torch.Tensor, frames: int) -> torch.Tensor:
    if frames <= 1:
        return x
    kernel = torch.ones(1, 1, frames, dtype=x.dtype) / frames
    pad_left = frames - 1
    y = torch.nn.functional.conv1d(x.view(1, 1, -1), kernel, padding=0)
    # causal smoothing: left-pad then keep same length
    x_pad = torch.nn.functional.pad(x.view(1, 1, -1), (pad_left, 0))
    y = torch.nn.functional.conv1d(x_pad, kernel)
    return y.view(-1)


def erle_curve_from_outputs(
    outputs: list[AECOutput],
    hop_length: int,
    smooth_frames: int = 20,
    eps: float = 1e-8,
) -> torch.Tensor:
    num_list = []
    den_list = []
    for out in outputs:
        d = out.d_target
        err = out.d_target - out.d_hat
        n = min(d.numel(), err.numel())
        num = frame_energy(d[:n], hop_length)
        den = frame_energy(err[:n], hop_length)
        if smooth_frames > 1:
            num = moving_average(num, smooth_frames)
            den = moving_average(den, smooth_frames)
        m = min(num.numel(), den.numel())
        num_list.append(num[:m])
        den_list.append(den[:m])

    min_frames = min(v.numel() for v in num_list)
    num_stack = torch.stack([v[:min_frames] for v in num_list], dim=0)
    den_stack = torch.stack([v[:min_frames] for v in den_list], dim=0)
    # Average powers first, then convert to dB.
    return 10.0 * torch.log10((num_stack.mean(dim=0) + eps) / (den_stack.mean(dim=0) + eps))


def scalar_erle_db(outputs: list[AECOutput], eps: float = 1e-8) -> float:
    num = 0.0
    den = 0.0
    for out in outputs:
        d = out.d_target
        err = out.d_target - out.d_hat
        n = min(d.numel(), err.numel())
        num += float(torch.sum(d[:n] ** 2))
        den += float(torch.sum(err[:n] ** 2))
    return float(10.0 * torch.log10(torch.tensor((num + eps) / (den + eps))))
