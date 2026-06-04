#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.metrics import count_trainable_parameters
from src.trainer import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Print model parameter count.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    print(model)
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")


if __name__ == "__main__":
    main()
