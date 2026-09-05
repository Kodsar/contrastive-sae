from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class PairedActivationDataset(Dataset[tuple[Tensor, Tensor]]):
    """Memory-mapped paired activation dataset produced by the extractor."""

    def __init__(self, activation_dir: str | Path, split: str) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")

        self.activation_dir = Path(activation_dir)
        manifest_path = self.activation_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Run activation extraction first."
            )

        with manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest: dict[str, Any] = json.load(handle)

        split_info = self.manifest["splits"][split]
        self.first_path = self.activation_dir / split_info["x1"]
        self.second_path = self.activation_dir / split_info["x2"]
        self.expected_length = int(split_info["pairs"])
        self.expected_dim = int(self.manifest["activation_dim"])
        self._first: np.ndarray | None = None
        self._second: np.ndarray | None = None
        self._open_arrays()

    def _open_arrays(self) -> None:
        self._first = np.load(self.first_path, mmap_mode="r")
        self._second = np.load(self.second_path, mmap_mode="r")
        expected_shape = (self.expected_length, self.expected_dim)
        if self._first.shape != expected_shape or self._second.shape != expected_shape:
            raise ValueError(
                "Activation array shape does not match manifest: "
                f"expected {expected_shape}, received "
                f"{self._first.shape} and {self._second.shape}"
            )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_first"] = None
        state["_second"] = None
        return state

    def __len__(self) -> int:
        return self.expected_length

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if self._first is None or self._second is None:
            self._open_arrays()
        assert self._first is not None and self._second is not None

        # Copy a row because read-only NumPy memmaps cannot safely back a tensor
        # that a caller might mutate.
        first = torch.from_numpy(np.array(self._first[index], copy=True))
        second = torch.from_numpy(np.array(self._second[index], copy=True))
        return first, second


def load_normalization_stats(activation_dir: str | Path) -> tuple[Tensor, float]:
    activation_dir = Path(activation_dir)
    mean = torch.from_numpy(np.load(activation_dir / "mean.npy")).float()
    scale_array = np.load(activation_dir / "scale.npy")
    scale = float(np.asarray(scale_array).reshape(()))
    if scale <= 0:
        raise ValueError(f"Activation scale must be positive, got {scale}")
    return mean, scale


def normalize_batch(batch: Tensor, mean: Tensor, scale: float) -> Tensor:
    return (batch.float() - mean) / scale

