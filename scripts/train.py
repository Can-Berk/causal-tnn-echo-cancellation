#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import get_device, load_config, save_config, set_seed
from src.trainer import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TNN-FRLS echo cancellation model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))
    device = get_device(cfg.get("training", {}).get("device", "auto"))
    print(f"Using device: {device}")
    save_config(cfg, cfg["experiment_dir"])
    train_model(cfg, device)


if __name__ == "__main__":
    main()
