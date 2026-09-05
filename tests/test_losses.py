from __future__ import annotations

import pytest
import torch

from contrastive_sae.losses import normalized_mse, symmetric_info_nce


def test_normalized_mse_is_zero_for_exact_reconstruction() -> None:
    target = torch.randn(8, 16)
    assert normalized_mse(target, target).item() == pytest.approx(0.0)


def test_info_nce_prefers_correct_alignment() -> None:
    embeddings = torch.eye(8)
    aligned = symmetric_info_nce(embeddings, embeddings, temperature=0.07)
    misaligned = symmetric_info_nce(
        embeddings, embeddings.roll(shifts=1, dims=0), temperature=0.07
    )
    assert aligned < misaligned


def test_info_nce_rejects_single_pair() -> None:
    with pytest.raises(ValueError, match="at least two"):
        symmetric_info_nce(torch.ones(1, 4), torch.ones(1, 4))


def test_info_nce_rejects_different_shapes() -> None:
    with pytest.raises(ValueError, match="equal shape"):
        symmetric_info_nce(torch.ones(2, 4), torch.ones(2, 5))

