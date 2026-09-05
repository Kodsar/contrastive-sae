from __future__ import annotations

import numpy as np

from contrastive_sae.extract_activations import RunningActivationStats


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

