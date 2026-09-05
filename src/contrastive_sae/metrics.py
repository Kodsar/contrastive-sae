from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from .data import normalize_batch
from .losses import normalized_mse, symmetric_info_nce
from .model import SplitTopKSAE
from .utils import autocast_context, tensor_float


@torch.no_grad()
def evaluate_model(
    model: SplitTopKSAE,
    batches: Iterable[tuple[Tensor, Tensor]],
    mean: Tensor,
    scale: float,
    device: torch.device,
    temperature: float,
    contrastive_weight: float,
    precision: str,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    mean = mean.to(device)
    weighted_sums: dict[str, float] = {}
    total_examples = 0
    high_feature_counts = torch.zeros(
        model.high.feature_dim, dtype=torch.long, device=device
    )
    low_feature_counts = torch.zeros(
        model.low.feature_dim, dtype=torch.long, device=device
    )

    for batch_index, (raw_first, raw_second) in enumerate(batches):
        if max_batches is not None and batch_index >= max_batches:
            break
        if raw_first.shape[0] < 2:
            continue

        first = normalize_batch(raw_first.to(device, non_blocking=True), mean, scale)
        second = normalize_batch(
            raw_second.to(device, non_blocking=True), mean, scale
        )
        with autocast_context(device, precision):
            first_output = model(first)
            second_output = model(second)

        reconstruction = 0.5 * (
            normalized_mse(first_output["reconstruction"], first)
            + normalized_mse(second_output["reconstruction"], second)
        )
        contrastive = symmetric_info_nce(
            first_output["high_code"],
            second_output["high_code"],
            temperature,
        )

        high_first = first_output["high_code"].float()
        high_second = second_output["high_code"].float()
        low_first = first_output["low_code"].float()
        low_second = second_output["low_code"].float()
        high_positive = F.cosine_similarity(high_first, high_second, dim=-1).mean()
        high_negative = F.cosine_similarity(
            high_first, high_second.roll(shifts=1, dims=0), dim=-1
        ).mean()
        low_positive = F.cosine_similarity(low_first, low_second, dim=-1).mean()

        centered_first = first - model.decoder_bias
        centered_second = second - model.decoder_bias
        centered = torch.cat([centered_first, centered_second], dim=0).float()
        high_reconstruction = torch.cat(
            [
                first_output["high_reconstruction"],
                second_output["high_reconstruction"],
            ],
            dim=0,
        ).float()
        low_reconstruction = torch.cat(
            [
                first_output["low_reconstruction"],
                second_output["low_reconstruction"],
            ],
            dim=0,
        ).float()
        denominator = centered.norm(dim=-1).clamp_min(1e-8)
        high_ratio = (high_reconstruction.norm(dim=-1) / denominator).mean()
        low_ratio = (low_reconstruction.norm(dim=-1) / denominator).mean()

        swapped_first = (
            model.decoder_bias
            + second_output["high_reconstruction"]
            + first_output["low_reconstruction"]
        )
        swapped_second = (
            model.decoder_bias
            + first_output["high_reconstruction"]
            + second_output["low_reconstruction"]
        )
        swap_nmse = 0.5 * (
            normalized_mse(swapped_first, first)
            + normalized_mse(swapped_second, second)
        )

        batch_size = first.shape[0]
        values = {
            "loss": reconstruction + contrastive_weight * contrastive,
            "reconstruction_nmse": reconstruction,
            "contrastive": contrastive,
            "high_positive_cosine": high_positive,
            "high_negative_cosine": high_negative,
            "semantic_gap": high_positive - high_negative,
            "low_positive_cosine": low_positive,
            "high_contribution_ratio": high_ratio,
            "low_contribution_ratio": low_ratio,
            "swap_nmse": swap_nmse,
            "high_active_mean": 0.5
            * (
                (high_first > 0).sum(dim=-1).float().mean()
                + (high_second > 0).sum(dim=-1).float().mean()
            ),
            "low_active_mean": 0.5
            * (
                (low_first > 0).sum(dim=-1).float().mean()
                + (low_second > 0).sum(dim=-1).float().mean()
            ),
        }
        for name, value in values.items():
            weighted_sums[name] = weighted_sums.get(name, 0.0) + tensor_float(
                value
            ) * batch_size

        high_feature_counts += (high_first > 0).sum(dim=0).to(torch.long)
        high_feature_counts += (high_second > 0).sum(dim=0).to(torch.long)
        low_feature_counts += (low_first > 0).sum(dim=0).to(torch.long)
        low_feature_counts += (low_second > 0).sum(dim=0).to(torch.long)
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Evaluation received no batch containing at least two pairs")

    result = {
        name: value / total_examples for name, value in weighted_sums.items()
    }
    result["high_dead_fraction"] = tensor_float(
        (high_feature_counts == 0).float().mean()
    )
    result["low_dead_fraction"] = tensor_float(
        (low_feature_counts == 0).float().mean()
    )
    result["evaluated_pairs"] = float(total_examples)
    return result

