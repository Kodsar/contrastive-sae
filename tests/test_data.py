from __future__ import annotations

import json

import numpy as np
import torch

from contrastive_sae.data import (
    PairedActivationDataset,
    load_normalization_stats,
    normalize_batch,
)


def test_memmap_dataset_and_normalization(tmp_path) -> None:
    first = np.arange(24, dtype=np.float16).reshape(4, 6)
    second = first + 2
    np.save(tmp_path / "train_x1.npy", first)
    np.save(tmp_path / "train_x2.npy", second)
    np.save(tmp_path / "validation_x1.npy", first[:2])
    np.save(tmp_path / "validation_x2.npy", second[:2])
    np.save(tmp_path / "mean.npy", np.ones(6, dtype=np.float32))
    np.save(tmp_path / "scale.npy", np.asarray(2.0, dtype=np.float32))
    manifest = {
        "activation_dim": 6,
        "splits": {
            "train": {"pairs": 4, "x1": "train_x1.npy", "x2": "train_x2.npy"},
            "validation": {
                "pairs": 2,
                "x1": "validation_x1.npy",
                "x2": "validation_x2.npy",
            },
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    dataset = PairedActivationDataset(tmp_path, "train")
    assert len(dataset) == 4
    loaded_first, loaded_second = dataset[2]
    assert torch.equal(loaded_first, torch.from_numpy(first[2]))
    assert torch.equal(loaded_second, torch.from_numpy(second[2]))

    mean, scale = load_normalization_stats(tmp_path)
    normalized = normalize_batch(loaded_first, mean, scale)
    assert torch.allclose(normalized, (loaded_first.float() - 1.0) / 2.0)

