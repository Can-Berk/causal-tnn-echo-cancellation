from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Literal

import soundfile as sf
import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


class AECPairDataset(Dataset):
    """AEC-Challenge synthetic dataset from a CSV file.

    Required CSV columns:
        file_id,x_path,y_path,d_path

    Optional CSV column:
        s_path

    x_path: far-end signal x(n)
    y_path: microphone signal y(n) = echo + near-end + noise
    d_path: clean echo target d(n)
    s_path: clean near-end speech s(n), used only for optional analysis/evaluation
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = _resolve_path(str(csv_path))
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.csv_path}\n"
                "Run: python scripts/prepare_data.py --root data/AEC-Challenge/datasets/synthetic"
            )

        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"x_path", "y_path", "d_path"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"CSV must contain columns {sorted(required)}; got {reader.fieldnames}"
                )
            self.rows = list(reader)

        if not self.rows:
            raise ValueError(f"CSV file is empty: {self.csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _read_mono(path: Path) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        audio, _sr = sf.read(path, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        return torch.from_numpy(mono)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        x = self._read_mono(_resolve_path(row["x_path"]))
        y = self._read_mono(_resolve_path(row["y_path"]))
        d = self._read_mono(_resolve_path(row["d_path"]))

        n = min(x.numel(), y.numel(), d.numel())
        item: Dict[str, torch.Tensor | str] = {
            "x": x[:n],
            "y": y[:n],
            "d": d[:n],
            "file_id": row.get("file_id", str(idx)),
        }

        if row.get("s_path"):
            s = self._read_mono(_resolve_path(row["s_path"]))
            item["s"] = s[: min(n, s.numel())]
        return item


def make_dataset(cfg: dict, split: Literal["train", "val", "test"]) -> Dataset:
    data_cfg = cfg["data"]
    key = f"{split}_csv"
    if key not in data_cfg:
        raise KeyError(f"Missing data.{key} in config")
    return AECPairDataset(data_cfg[key])


def collate_pad(batch: list[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor | list[str]]:
    """Pad waveform tensors to the longest utterance length in a batch."""
    max_len = max(item["x"].numel() for item in batch)  # type: ignore[union-attr]
    out: Dict[str, torch.Tensor | list[str]] = {}
    for key in ("x", "y", "d"):
        tensors = []
        for item in batch:
            t = item[key]  # type: ignore[index]
            assert isinstance(t, torch.Tensor)
            if t.numel() < max_len:
                t = torch.nn.functional.pad(t, (0, max_len - t.numel()))
            tensors.append(t)
        out[key] = torch.stack(tensors, dim=0).float()

    if any("s" in item for item in batch):
        tensors = []
        for item in batch:
            t = item.get("s")
            if not isinstance(t, torch.Tensor):
                t = torch.zeros(max_len)
            if t.numel() < max_len:
                t = torch.nn.functional.pad(t, (0, max_len - t.numel()))
            tensors.append(t[:max_len])
        out["s"] = torch.stack(tensors, dim=0).float()

    out["file_id"] = [str(item.get("file_id", "")) for item in batch]
    return out
