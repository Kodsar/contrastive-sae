from __future__ import annotations

import sys
from types import ModuleType

import numpy as np

from contrastive_sae.extract_activations import (
    RunningActivationStats,
    load_text_pairs,
)


def test_running_stats_match_numpy() -> None:
    activations = np.asarray(
        [[1.0, 2.0, 3.0], [2.0, 5.0, 1.0], [-1.0, 0.0, 4.0]],
        dtype=np.float32,
    )
    stats = RunningActivationStats(dimension=3)
    stats.update(activations[:2])
    stats.update(activations[2:])
    mean, scale = stats.finalize()

    expected_mean = activations.mean(axis=0)
    expected_scale = np.sqrt(np.square(activations - expected_mean).mean())
    assert np.allclose(mean, expected_mean)
    assert np.allclose(scale, expected_scale)


def test_huggingface_dataset_uses_configured_cache_dir(
    tmp_path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class FakeDataset(list):
        column_names = ["anchor", "positive"]

        def shuffle(self, seed: int):
            captured["shuffle_seed"] = seed
            return self

    def fake_load_dataset(
        name: str,
        dataset_config: str,
        *,
        split: str,
        cache_dir: str | None,
    ) -> FakeDataset:
        captured.update(
            name=name,
            dataset_config=dataset_config,
            split=split,
            cache_dir=cache_dir,
        )
        return FakeDataset(
            {"anchor": f"anchor {index}", "positive": f"positive {index}"}
            for index in range(4)
        )

    fake_datasets = ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    cache_dir = tmp_path / "huggingface"
    pairs = load_text_pairs(
        {
            "source": "huggingface",
            "cache_dir": str(cache_dir),
            "dataset_name": "sentence-transformers/quora-duplicates",
            "dataset_config": "pair",
            "split": "train",
            "text1_column": "anchor",
            "text2_column": "positive",
            "shuffle": True,
        },
        seed=42,
        max_pairs=4,
    )

    assert len(pairs) == 4
    assert cache_dir.is_dir()
    assert captured["cache_dir"] == str(cache_dir)
    assert captured["shuffle_seed"] == 42
