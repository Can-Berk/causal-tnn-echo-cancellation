from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .features import pad_r_to_m


@dataclass
class FRLSState:
    W: torch.Tensor       # complex tensor, (B, M)
    psd_xx: torch.Tensor  # real tensor, (B, M)


class FRLSFilter(torch.nn.Module):
    """Compact frequency-domain RLS-style adaptive filter.

    W(k+1) = W(k) + mu(k) * (1-lambda) * psd_xx^{-1}(k) * X*(k) * E(k)
    psd_xx(k) = lambda * psd_xx(k-1) + (1-lambda) * |X(k)|^2
    """

    def __init__(self, frame_length: int, hop_length: int, lambda_: float, psd_init: float = 1e-3, eps: float = 1e-8):
        super().__init__()
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.lambda_ = float(lambda_)
        self.psd_init = float(psd_init)
        self.eps = float(eps)

    def initial_state(self, batch_size: int, device: torch.device) -> FRLSState:
        W = torch.zeros(batch_size, self.frame_length, dtype=torch.complex64, device=device)
        psd = torch.full((batch_size, self.frame_length), self.psd_init, dtype=torch.float32, device=device)
        return FRLSState(W=W, psd_xx=psd)

    def forward(
        self,
        x_frame: torch.Tensor,
        y_frame_r: torch.Tensor,
        mu_full: torch.Tensor,
        state: FRLSState,
    ) -> Tuple[torch.Tensor, FRLSState]:
        X = torch.fft.fft(x_frame, n=self.frame_length, dim=-1)

        d_hat_full = torch.fft.ifft(X * state.W, n=self.frame_length, dim=-1).real
        d_hat_r = d_hat_full[:, -self.hop_length:]

        e_r = y_frame_r - d_hat_r
        E = torch.fft.fft(pad_r_to_m(e_r, self.frame_length), n=self.frame_length, dim=-1)

        one_minus_lambda = 1.0 - self.lambda_
        psd_xx = self.lambda_ * state.psd_xx + one_minus_lambda * (X.abs() ** 2)
        update = mu_full * one_minus_lambda * X.conj() * E / (psd_xx + self.eps)
        W = state.W + update

        return d_hat_r, FRLSState(W=W, psd_xx=psd_xx)
