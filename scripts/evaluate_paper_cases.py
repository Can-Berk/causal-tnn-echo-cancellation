#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import get_device, load_config, set_seed
from src.dataset import AECPairDataset
from src.frls import FRLSFilter
from src.paper_eval import erle_curve_from_outputs, run_aec_signals, scalar_erle_db
from src.trainer import build_model


def make_frls(cfg: dict, device: torch.device) -> FRLSFilter:
    frls_cfg = cfg["frls"]
    return FRLSFilter(
        frame_length=frls_cfg["n_fft"],
        hop_length=frls_cfg["hop_length"],
        lambda_=frls_cfg["lambda_"],
        psd_init=frls_cfg.get("psd_init", 1e-3),
        eps=frls_cfg.get("eps", 1e-8),
    ).to(device)


def load_model_or_none(cfg: dict, checkpoint: Optional[str], fixed_mu: Optional[float], device: torch.device):
    if fixed_mu is not None:
        return None
    if checkpoint is None:
        raise SystemExit("Provide --checkpoint, or use --fixed-mu 1.0 for standalone FRLS.")
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def align_three(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor):
    n = min(x.numel(), y.numel(), d.numel())
    return x[:n], y[:n], d[:n]


def make_echo_path_change_case(item_a: dict, item_b: dict):
    """Approximate paper Fig. 4c case by splicing two far-end single-talk examples.

    The public AEC-Challenge synthetic files do not provide multiple echo paths
    for the exact same far-end waveform. This creates a practical paper-style
    echo-path-change test by switching from one echo-only example to the next at
    the midpoint. It is useful for reconvergence comparison, but it is not a
    bit-exact reproduction of the paper's hidden evaluation construction.
    """
    xa, ya, da = item_a["x"], item_a["d"], item_a["d"]
    xb, yb, db = item_b["x"], item_b["d"], item_b["d"]
    xa, ya, da = align_three(xa, ya, da)
    xb, yb, db = align_three(xb, yb, db)
    n = min(xa.numel(), xb.numel(), da.numel(), db.numel())
    mid = n // 2
    x = torch.cat([xa[:mid], xb[mid:n]])
    y = torch.cat([da[:mid], db[mid:n]])
    d = y.clone()
    return x, y, d


def save_curves_csv(curves: dict[str, torch.Tensor], time_s: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["time_s"] + list(curves.keys())
    min_len = min([time_s.numel()] + [v.numel() for v in curves.values()])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(min_len):
            row = {"time_s": float(time_s[i])}
            for key, curve in curves.items():
                row[key] = float(curve[i])
            writer.writerow(row)


def plot_curves(curves: dict[str, torch.Tensor], time_s: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for name, curve in curves.items():
        n = min(time_s.numel(), curve.numel())
        plt.plot(time_s[:n].numpy(), curve[:n].numpy(), label=name)
    plt.xlabel("time [s]")
    plt.ylabel("ERLE [dB]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-style evaluation: single-talk, double-talk, echo-path change.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Path to trained TNN checkpoint. Omit when using --fixed-mu.")
    parser.add_argument("--fixed-mu", type=float, default=None, help="Use constant stepsize, e.g. --fixed-mu 1.0 for standalone FRLS.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-files", type=int, default=None, help="Optional quick debug limit.")
    parser.add_argument("--smooth-frames", type=int, default=20, help="Causal moving-average smoothing for ERLE powers.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))
    device = get_device(cfg.get("training", {}).get("device", "auto"))
    print(f"Using device: {device}")

    model = load_model_or_none(cfg, args.checkpoint, args.fixed_mu, device)
    frls = make_frls(cfg, device)
    context_frames = int(cfg.get("model", {}).get("context_frames", 1))
    dataset = AECPairDataset(cfg["data"]["test_csv"])

    if args.out_dir is None:
        name = "fixed_frls" if args.fixed_mu is not None else cfg["model"]["name"]
        out_dir = Path(cfg["experiment_dir"]) / "paper_cases" / name
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = len(dataset) if args.max_files is None else min(args.max_files, len(dataset))
    outputs = {"single_talk": [], "double_talk": [], "echo_path_change": []}

    with torch.no_grad():
        for i in tqdm(range(count), desc="paper cases"):
            item = dataset[i]
            next_item = dataset[(i + 1) % len(dataset)]

            x, y, d = item["x"], item["y"], item["d"]
            assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and isinstance(d, torch.Tensor)

            # Fig. 4a style: far-end single-talk, no near-end disturbance.
            x_a, y_a, d_a = align_three(x, d, d)
            outputs["single_talk"].append(
                run_aec_signals(model, x_a, y_a, d_a, frls, device, context_frames, fixed_mu=args.fixed_mu)
            )

            # Fig. 4b style: double-talk microphone signal from AEC-Challenge.
            x_b, y_b, d_b = align_three(x, y, d)
            outputs["double_talk"].append(
                run_aec_signals(model, x_b, y_b, d_b, frls, device, context_frames, fixed_mu=args.fixed_mu)
            )

            # Fig. 4c style approximation: echo-only signal with midpoint change.
            x_c, y_c, d_c = make_echo_path_change_case(item, next_item)
            outputs["echo_path_change"].append(
                run_aec_signals(model, x_c, y_c, d_c, frls, device, context_frames, fixed_mu=args.fixed_mu)
            )

    hop = int(cfg["frls"]["hop_length"])
    sr = int(cfg["data"].get("sample_rate", 16000))
    curves = {
        case: erle_curve_from_outputs(case_outputs, hop, smooth_frames=args.smooth_frames)
        for case, case_outputs in outputs.items()
    }
    min_len = min(v.numel() for v in curves.values())
    time_s = torch.arange(min_len, dtype=torch.float32) * hop / sr
    curves = {k: v[:min_len] for k, v in curves.items()}

    summary = {
        "num_test_files": count,
        "sample_rate": sr,
        "hop_length": hop,
        "smooth_frames": args.smooth_frames,
        "fixed_mu": args.fixed_mu,
        "checkpoint": args.checkpoint,
        "cases": {},
        "note": (
            "single_talk uses y=d. double_talk uses nearend_mic_signal. "
            "echo_path_change is an approximation made by splicing two echo-only test examples at midpoint."
        ),
    }
    for case, case_outputs in outputs.items():
        curve = curves[case]
        last_second_frames = max(1, int(sr / hop))
        summary["cases"][case] = {
            "global_erle_db": scalar_erle_db(case_outputs),
            "mean_curve_erle_db": float(curve.mean()),
            "last_second_erle_db": float(curve[-last_second_frames:].mean()),
            "max_curve_erle_db": float(curve.max()),
        }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    save_curves_csv(curves, time_s, out_dir / "erle_curves.csv")
    plot_curves(curves, time_s, out_dir / "erle_curves.png")

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_dir / 'summary.json'}")
    print(f"Saved: {out_dir / 'erle_curves.csv'}")
    print(f"Saved: {out_dir / 'erle_curves.png'}")


if __name__ == "__main__":
    main()
