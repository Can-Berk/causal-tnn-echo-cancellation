from __future__ import annotations

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def erle_db(d_true: torch.Tensor, d_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Echo Return Loss Enhancement in dB."""
    num = torch.mean(d_true ** 2)
    den = torch.mean((d_true - d_hat) ** 2)
    return 10.0 * torch.log10((num + eps) / (den + eps))


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
