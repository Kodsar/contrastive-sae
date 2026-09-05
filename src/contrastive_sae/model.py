from __future__ import annotations

from typing import Any, Mapping, TypedDict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SAEForwardOutput(TypedDict):
    reconstruction: Tensor
    high_code: Tensor
    low_code: Tensor
    high_reconstruction: Tensor
    low_reconstruction: Tensor
    residual: Tensor


def topk_relu(pre_activations: Tensor, k: int) -> Tensor:
    """Apply ReLU and retain the largest ``k`` values in each final dimension."""
    if pre_activations.ndim < 2:
        raise ValueError("topk_relu expects at least two dimensions")
    if not 0 < k <= pre_activations.shape[-1]:
        raise ValueError(
            f"k must be in [1, {pre_activations.shape[-1]}], received {k}"
        )

    positive = F.relu(pre_activations)
    values, indices = torch.topk(positive, k=k, dim=-1, sorted=False)
    sparse = torch.zeros_like(positive)
    return sparse.scatter(-1, indices, values)


class SparseBranch(nn.Module):
    """One independently parameterized TopK encoder/linear decoder branch."""

    def __init__(self, input_dim: int, feature_dim: int, k: int) -> None:
        super().__init__()
        if not 0 < k <= feature_dim:
            raise ValueError("k must be positive and no larger than feature_dim")

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.k = k
        self.encoder = nn.Linear(input_dim, feature_dim, bias=True)
        self.decoder = nn.Linear(feature_dim, input_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            directions = torch.randn(
                self.input_dim,
                self.feature_dim,
                device=self.decoder.weight.device,
                dtype=self.decoder.weight.dtype,
            )
            directions = F.normalize(directions, dim=0)
            self.decoder.weight.copy_(directions)
            self.encoder.weight.copy_(directions.T)
            self.encoder.bias.zero_()

    def encode(self, x: Tensor) -> Tensor:
        return topk_relu(self.encoder(x), self.k)

    def decode(self, code: Tensor) -> Tensor:
        return self.decoder(code)

    @torch.no_grad()
    def normalize_decoder_columns_(self) -> None:
        self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))


class SplitTopKSAE(nn.Module):
    """High-level semantic branch plus low-level residual-correction branch.

    The high branch sees the centered input. The low branch sees what remains
    after the high reconstruction. Both branches participate in the final
    reconstruction; callers should apply contrastive losses only to
    ``high_code``.
    """

    def __init__(
        self,
        input_dim: int,
        high_features: int,
        low_features: int,
        high_topk: int,
        low_topk: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.high = SparseBranch(input_dim, high_features, high_topk)
        self.low = SparseBranch(input_dim, low_features, low_topk)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SplitTopKSAE":
        return cls(
            input_dim=int(config["input_dim"]),
            high_features=int(config["high_features"]),
            low_features=int(config["low_features"]),
            high_topk=int(config["high_topk"]),
            low_topk=int(config["low_topk"]),
        )

    def forward(self, x: Tensor) -> SAEForwardOutput:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected final dimension {self.input_dim}, got {x.shape[-1]}"
            )

        centered = x - self.decoder_bias
        high_code = self.high.encode(centered)
        high_reconstruction = self.high.decode(high_code)

        residual = centered - high_reconstruction
        low_code = self.low.encode(residual)
        low_reconstruction = self.low.decode(low_code)

        reconstruction = (
            self.decoder_bias + high_reconstruction + low_reconstruction
        )
        return {
            "reconstruction": reconstruction,
            "high_code": high_code,
            "low_code": low_code,
            "high_reconstruction": high_reconstruction,
            "low_reconstruction": low_reconstruction,
            "residual": residual,
        }

    @torch.no_grad()
    def normalize_decoder_columns_(self) -> None:
        self.high.normalize_decoder_columns_()
        self.low.normalize_decoder_columns_()

    @property
    def high_topk(self) -> int:
        return self.high.k

    @property
    def low_topk(self) -> int:
        return self.low.k
