from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import SAEForwardOutput


class LossOutput(TypedDict):
    loss: Tensor
    reconstruction: Tensor
    contrastive: Tensor
    contrastive_weight: Tensor


def normalized_mse(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Mean squared error divided by target energy."""
    prediction_f = prediction.float()
    target_f = target.float()
    mse = (prediction_f - target_f).square().mean()
    target_energy = target_f.square().mean().clamp_min(eps)
    return mse / target_energy


def symmetric_info_nce(
    first: Tensor,
    second: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """Symmetric in-batch InfoNCE for aligned rows of two embedding batches."""
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("InfoNCE inputs must both have shape [batch, features]")
    if first.shape != second.shape:
        raise ValueError(
            f"InfoNCE inputs must have equal shape, got {first.shape} and "
            f"{second.shape}"
        )
    if first.shape[0] < 2:
        raise ValueError("InfoNCE requires at least two paired examples")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    first_norm = F.normalize(first.float(), dim=-1, eps=1e-8)
    second_norm = F.normalize(second.float(), dim=-1, eps=1e-8)
    logits = first_norm @ second_norm.T / temperature
    labels = torch.arange(first.shape[0], device=first.device)
    forward = F.cross_entropy(logits, labels)
    backward = F.cross_entropy(logits.T, labels)
    return 0.5 * (forward + backward)


def split_sae_loss(
    first: SAEForwardOutput,
    second: SAEForwardOutput,
    target_first: Tensor,
    target_second: Tensor,
    contrastive_weight: float,
    temperature: float,
) -> LossOutput:
    """Joint reconstruction plus high-code-only contrastive objective."""
    reconstruction = 0.5 * (
        normalized_mse(first["reconstruction"], target_first)
        + normalized_mse(second["reconstruction"], target_second)
    )
    contrastive = symmetric_info_nce(
        first["high_code"], second["high_code"], temperature=temperature
    )
    weight = reconstruction.new_tensor(contrastive_weight)
    loss = reconstruction + weight * contrastive
    return {
        "loss": loss,
        "reconstruction": reconstruction,
        "contrastive": contrastive,
        "contrastive_weight": weight,
    }

