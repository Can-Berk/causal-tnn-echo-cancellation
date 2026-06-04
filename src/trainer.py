from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .dataset import collate_pad, make_dataset
from .features import (
    bins_to_full_spectrum,
    make_r_frames,
    make_x_frames,
    pad_r_to_m,
    positive_frequency_magnitude,
)
from .frls import FRLSFilter
from .metrics import count_trainable_parameters, erle_db, mse
from .models import AutoTNN, CausalConv2DAutoTNN, CausalConv2DTNN, FcTNN


def build_model(cfg: dict) -> torch.nn.Module:
    n_fft = int(cfg["frls"]["n_fft"])
    feature_bins = n_fft // 2
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "fctnn")

    if name == "fctnn":
        return FcTNN(feature_bins=feature_bins)
    if name == "autotnn":
        return AutoTNN(feature_bins=feature_bins)
    if name == "causal_conv2d_tnn":
        return CausalConv2DTNN(
            feature_bins=feature_bins,
            context_frames=model_cfg.get("context_frames", 8),
            hidden_channels=model_cfg.get("hidden_channels", 8),
            kernel_time=model_cfg.get("kernel_time", 3),
            kernel_freq=model_cfg.get("kernel_freq", 5),
        )
    if name == "causal_conv2d_autotnn":
        return CausalConv2DAutoTNN(
            feature_bins=feature_bins,
            context_frames=model_cfg.get("context_frames", 8),
            kernel_time=model_cfg.get("kernel_time", 3),
            kernel_freq=model_cfg.get("kernel_freq", 5),
            bottleneck_channels=model_cfg.get("bottleneck_channels", 16),
        )
    raise ValueError(f"Unknown model.name: {name}")


def make_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = make_dataset(cfg, "train")
    val_ds = make_dataset(cfg, "val")
    test_ds = make_dataset(cfg, "test")
    batch_size = int(cfg["training"].get("batch_size", 6))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pad),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad),
    )


def make_epoch_train_loader(cfg: dict, epoch: int) -> DataLoader:
    """Paper-style training loader.

    The paper uses N=30 random training utterances per epoch and then forms
    batches of B=6. Set training.train_utterances_per_epoch to reproduce this
    behavior. If the setting is missing/null, all training utterances are used.
    """
    train_ds = make_dataset(cfg, "train")
    batch_size = int(cfg["training"].get("batch_size", 6))
    n_per_epoch = cfg["training"].get("train_utterances_per_epoch")
    if n_per_epoch is None:
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)

    n_per_epoch = min(int(n_per_epoch), len(train_ds))
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 0)) + epoch)
    indices = torch.randperm(len(train_ds), generator=generator)[:n_per_epoch].tolist()
    subset = Subset(train_ds, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)


def _model_mu(
    model: torch.nn.Module,
    X_mag: torch.Tensor,
    Y_mag: torch.Tensor,
    x_hist: deque[torch.Tensor],
    y_hist: deque[torch.Tensor],
    context_frames: int,
) -> torch.Tensor:
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


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    frls: FRLSFilter,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: dict,
    train: bool,
) -> Dict[str, float]:
    model.train(train)

    total_loss = 0.0
    total_erle = 0.0
    total_items = 0
    context_frames = int(cfg.get("model", {}).get("context_frames", 1))

    iterator = tqdm(loader, desc="train" if train else "eval", leave=False)
    for batch in iterator:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        d = batch["d"].to(device)

        batch_size = x.shape[0]
        x_frames = make_x_frames(x, frls.frame_length, frls.hop_length)
        y_frames = make_r_frames(y, frls.hop_length)
        d_frames = make_r_frames(d, frls.hop_length)
        num_frames = min(x_frames.shape[1], y_frames.shape[1], d_frames.shape[1])
        x_frames = x_frames[:, :num_frames]
        y_frames = y_frames[:, :num_frames]
        d_frames = d_frames[:, :num_frames]

        state = frls.initial_state(batch_size, device)
        d_hat_frames = []
        x_hist: deque[torch.Tensor] = deque(maxlen=context_frames)
        y_hist: deque[torch.Tensor] = deque(maxlen=context_frames)

        for k in range(num_frames):
            xk = x_frames[:, k, :]
            yk = y_frames[:, k, :]

            X = torch.fft.fft(xk, n=frls.frame_length, dim=-1)
            Y = torch.fft.fft(pad_r_to_m(yk, frls.frame_length), n=frls.frame_length, dim=-1)
            X_mag = positive_frequency_magnitude(X)
            Y_mag = positive_frequency_magnitude(Y)

            mu_bins = _model_mu(model, X_mag, Y_mag, x_hist, y_hist, context_frames)
            mu_full = bins_to_full_spectrum(mu_bins, frls.frame_length)

            d_hat_k, state = frls(xk, yk, mu_full, state)
            d_hat_frames.append(d_hat_k)

        d_hat = torch.cat(d_hat_frames, dim=-1)
        d_target = d_frames.reshape(batch_size, -1)[..., : d_hat.shape[-1]]
        loss = mse(d_hat, d_target)

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip = cfg["training"].get("grad_clip")
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

        with torch.no_grad():
            erle = erle_db(d_target, d_hat)
            n_items = batch_size
            total_loss += loss.item() * n_items
            total_erle += erle.item() * n_items
            total_items += n_items
            iterator.set_postfix(loss=loss.item(), erle_db=erle.item())

    return {"loss": total_loss / max(total_items, 1), "erle_db": total_erle / max(total_items, 1)}


def train_model(cfg: dict, device: torch.device) -> Dict[str, list[float]]:
    out_dir = Path(cfg["experiment_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    _train_loader, val_loader, _test_loader = make_dataloaders(cfg)
    model = build_model(cfg).to(device)
    print(f"Model: {cfg['model']['name']}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    frls_cfg = cfg["frls"]
    frls = FRLSFilter(
        frame_length=frls_cfg["n_fft"],
        hop_length=frls_cfg["hop_length"],
        lambda_=frls_cfg["lambda_"],
        psd_init=frls_cfg.get("psd_init", 1e-3),
        eps=frls_cfg.get("eps", 1e-8),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))
    epochs = int(cfg["training"].get("epochs", 20))
    patience = int(cfg["training"].get("patience", 5))
    best_val = float("inf")
    patience_count = 0
    history: Dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_erle_db": [], "val_erle_db": []}

    for epoch in range(1, epochs + 1):
        train_loader = make_epoch_train_loader(cfg, epoch)
        train_stats = run_epoch(model, train_loader, frls, optimizer, device, cfg, train=True)
        with torch.no_grad():
            val_stats = run_epoch(model, val_loader, frls, None, device, cfg, train=False)

        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["train_erle_db"].append(train_stats["erle_db"])
        history["val_erle_db"].append(val_stats["erle_db"])

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_stats['loss']:.6f}, ERLE {train_stats['erle_db']:.2f} dB | "
            f"val loss {val_stats['loss']:.6f}, ERLE {val_stats['erle_db']:.2f} dB"
        )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            patience_count = 0
            torch.save({"model_state": model.state_dict(), "cfg": cfg}, out_dir / "best.pt")
        else:
            patience_count += 1
            if patience_count >= patience:
                print("Early stopping.")
                break

        with (out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    return history


def evaluate_checkpoint(cfg: dict, checkpoint_path: str | Path, device: torch.device) -> Dict[str, float]:
    _train_loader, _val_loader, test_loader = make_dataloaders(cfg)
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    frls_cfg = cfg["frls"]
    frls = FRLSFilter(
        frame_length=frls_cfg["n_fft"],
        hop_length=frls_cfg["hop_length"],
        lambda_=frls_cfg["lambda_"],
        psd_init=frls_cfg.get("psd_init", 1e-3),
        eps=frls_cfg.get("eps", 1e-8),
    ).to(device)
    with torch.no_grad():
        return run_epoch(model, test_loader, frls, None, device, cfg, train=False)
