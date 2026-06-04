#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

CASES = ["single_talk", "double_talk", "echo_path_change"]


def read_curve(csv_path: Path, case: str) -> tuple[list[float], list[float]]:
    time_s: list[float] = []
    values: list[float] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        if "time_s" not in reader.fieldnames:
            raise ValueError(f"CSV must contain a 'time_s' column: {csv_path}")
        if case not in reader.fieldnames:
            raise ValueError(
                f"Case '{case}' not found in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            time_s.append(float(row["time_s"]))
            values.append(float(row[case]))

    return time_s, values


def parse_runs(run_args: list[str]) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for run in run_args:
        if "=" not in run:
            raise ValueError(
                f"Invalid --run '{run}'. Use name=path/to/erle_curves.csv"
            )
        name, path = run.split("=", 1)
        name = name.strip()
        csv_path = Path(path.strip())
        if not name:
            raise ValueError(f"Run name is empty in: {run}")
        if not csv_path.exists():
            raise FileNotFoundError(f"Curve file not found: {csv_path}")
        runs.append((name, csv_path))
    return runs


def plot_case(case: str, runs: list[tuple[str, Path]], out_path: Path) -> None:
    plt.figure()

    for name, csv_path in runs:
        time_s, curve = read_curve(csv_path, case)
        plt.plot(time_s, curve, label=name)

    pretty_title = case.replace("_", " ").title()
    plt.xlabel("Time [s]")
    plt.ylabel("ERLE [dB]")
    plt.title(pretty_title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create paper/Fig.-4-style ERLE plots: one evaluation case with "
            "multiple model curves on the same graph."
        )
    )
    parser.add_argument(
        "--case",
        default="all",
        choices=CASES + ["all"],
        help="Which case to plot. Use 'all' to create one figure per case.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Model curve in the form name=path/to/erle_curves.csv. "
            "Use this argument multiple times."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Only used when --case is not 'all'.",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/paper_cases/figures",
        help="Output directory used when --case all.",
    )
    args = parser.parse_args()

    runs = parse_runs(args.run)

    if args.case == "all":
        out_dir = Path(args.out_dir)
        for case in CASES:
            plot_case(case, runs, out_dir / f"fig4_{case}.png")
    else:
        if args.out is None:
            out_path = Path(args.out_dir) / f"fig4_{args.case}.png"
        else:
            out_path = Path(args.out)
        plot_case(args.case, runs, out_path)


if __name__ == "__main__":
    main()
