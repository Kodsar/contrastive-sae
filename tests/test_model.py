from __future__ import annotations

import torch
from torch.nn import functional as F

from contrastive_sae.losses import normalized_mse, symmetric_info_nce
from contrastive_sae.model import SplitTopKSAE, topk_relu


def make_model() -> SplitTopKSAE:
    torch.manual_seed(0)
    return SplitTopKSAE(
        input_dim=12,
        high_features=20,
        low_features=10,
        high_topk=4,
        low_topk=2,
    )


def test_topk_relu_obeys_budget_and_nonnegativity() -> None:
    values = torch.tensor(
        [[-2.0, 5.0, 1.0, 4.0], [3.0, -1.0, 2.0, 0.5]]
    )
    result = topk_relu(values, k=2)
    assert torch.all(result >= 0)
    assert torch.equal((result > 0).sum(dim=-1), torch.tensor([2, 2]))
    assert torch.allclose(result[0], torch.tensor([0.0, 5.0, 0.0, 4.0]))


def test_split_model_shapes_and_separate_sparsity_budgets() -> None:
    model = make_model()
    output = model(torch.randn(7, 12))

    assert output["reconstruction"].shape == (7, 12)
    assert output["high_code"].shape == (7, 20)
    assert output["low_code"].shape == (7, 10)
    assert torch.all((output["high_code"] > 0).sum(dim=-1) <= 4)
    assert torch.all((output["low_code"] > 0).sum(dim=-1) <= 2)


def test_reconstruction_gradient_reaches_both_branches() -> None:
    model = make_model()
    inputs = torch.randn(16, 12)
    output = model(inputs)
    loss = normalized_mse(output["reconstruction"], inputs)
    loss.backward()

    assert model.high.encoder.weight.grad is not None
    assert model.high.decoder.weight.grad is not None
    assert model.low.encoder.weight.grad is not None
    assert model.low.decoder.weight.grad is not None
    assert model.high.encoder.weight.grad.abs().sum() > 0
    assert model.low.encoder.weight.grad.abs().sum() > 0


def test_contrastive_gradient_does_not_reach_low_branch() -> None:
    model = make_model()
    first = model(torch.randn(16, 12))
    second = model(torch.randn(16, 12))
    loss = symmetric_info_nce(first["high_code"], second["high_code"])
    loss.backward()

    assert model.high.encoder.weight.grad is not None
    assert model.high.encoder.weight.grad.abs().sum() > 0
    assert model.low.encoder.weight.grad is None
    assert model.low.decoder.weight.grad is None


def test_decoder_column_normalization() -> None:
    model = make_model()
    with torch.no_grad():
        model.high.decoder.weight.mul_(3.0)
        model.low.decoder.weight.mul_(0.25)
    model.normalize_decoder_columns_()

    high_norms = model.high.decoder.weight.norm(dim=0)
    low_norms = model.low.decoder.weight.norm(dim=0)
    assert torch.allclose(high_norms, torch.ones_like(high_norms), atol=1e-6)
    assert torch.allclose(low_norms, torch.ones_like(low_norms), atol=1e-6)


def test_low_branch_receives_high_residual() -> None:
    model = make_model()
    inputs = torch.randn(5, 12)
    output = model(inputs)
    expected = inputs - model.decoder_bias - output["high_reconstruction"]
    assert torch.allclose(output["residual"], expected)
    expected_reconstruction = (
        model.decoder_bias
        + output["high_reconstruction"]
        + output["low_reconstruction"]
    )
    assert torch.allclose(output["reconstruction"], expected_reconstruction)

