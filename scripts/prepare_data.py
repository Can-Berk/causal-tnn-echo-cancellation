#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def extract_id(path: Path) -> str:
    """Extract the shared file id from AEC-Challenge synthetic filenames."""
    stem = path.stem.lower()
    match = re.search(r"fileid[_-]?(\d+)", stem)
    if match:
        return match.group(1)
    match = re.search(r"(\d+)$", stem)
    if match:
        return match.group(1)
    return stem


def collect_wavs(folder: Path) -> dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return {extract_id(path): path for path in folder.glob("*.wav")}


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file_id", "x_path", "y_path", "d_path", "s_path"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare train/val/test CSV files for the AEC-Challenge synthetic dataset."
    )
    parser.add_argument(
        "--root",
        default="data/AEC-Challenge/datasets/synthetic",
        help="Path to AEC-Challenge datasets/synthetic folder.",
    )
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--train", type=int, default=448)
    parser.add_argument("--val", type=int, default=24)
    parser.add_argument("--test", type=int, default=26)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = resolve_path(args.root)
    out_dir = resolve_path(args.out_dir)

    x_files = collect_wavs(root / "farend_speech")
    y_files = collect_wavs(root / "nearend_mic_signal")
    d_files = collect_wavs(root / "echo_signal")
    s_files = collect_wavs(root / "nearend_speech") if (root / "nearend_speech").exists() else {}

    common_ids = sorted(set(x_files) & set(y_files) & set(d_files), key=lambda s: int(s) if s.isdigit() else s)
    if not common_ids:
        raise RuntimeError(
            "No matching x/y/d triplets found. Check that farend_speech, "
            "nearend_mic_signal, and echo_signal exist and contain .wav files."
        )

    rows = []
    for file_id in common_ids:
        rows.append(
            {
                "file_id": file_id,
                "x_path": rel_or_abs(x_files[file_id]),
                "y_path": rel_or_abs(y_files[file_id]),
                "d_path": rel_or_abs(d_files[file_id]),
                "s_path": rel_or_abs(s_files[file_id]) if file_id in s_files else "",
            }
        )

    random.Random(args.seed).shuffle(rows)

    total = args.train + args.val + args.test
    if len(rows) < total:
        raise RuntimeError(f"Need {total} triplets, but found only {len(rows)}.")

    rows = rows[:total]
    train_rows = rows[: args.train]
    val_rows = rows[args.train : args.train + args.val]
    test_rows = rows[args.train + args.val :]

    write_csv(train_rows, out_dir / "train_pairs.csv")
    write_csv(val_rows, out_dir / "val_pairs.csv")
    write_csv(test_rows, out_dir / "test_pairs.csv")

    print("\nNext:")
    print("  python scripts/train.py --config configs/fctnn.yaml")
    print("  python scripts/evaluate_paper_cases.py --config configs/fctnn.yaml --checkpoint experiments/fctnn/best.pt")


if __name__ == "__main__":
    main()
