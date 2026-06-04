from __future__ import annotations

import torch
from torch import nn


class FcTNN(nn.Module):
    """Fully connected TNN baseline.

    Input: |X(k)| and |Y(k)|, each (B, F).
    Output: (B, F). Internally it predicts one scalar per frame and broadcasts
    the scalar to all frequency bins for a common interface.
    """

    uses_sequence = False

    def __init__(self, feature_bins: int):
        super().__init__()
        self.feature_bins = int(feature_bins)
        self.norm = nn.LayerNorm(2 * self.feature_bins, elementwise_affine=False)
        self.fc = nn.Linear(2 * self.feature_bins, 1, bias=False)

    def forward(self, X_mag: torch.Tensor, Y_mag: torch.Tensor) -> torch.Tensor:
        if X_mag.ndim != 2 or Y_mag.ndim != 2:
            raise ValueError("FcTNN expects X_mag and Y_mag with shape (B, F)")
        z = torch.cat([X_mag, Y_mag], dim=-1)
        z = self.norm(z)
        mu_scalar = torch.sigmoid(self.fc(z))
        return mu_scalar.expand(-1, self.feature_bins)
        # all values across frequency are equal because it came from one scalar
