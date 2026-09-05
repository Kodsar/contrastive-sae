"""Minimal contrastive high/low sparse autoencoder package."""

from .losses import normalized_mse, split_sae_loss, symmetric_info_nce
from .model import SplitTopKSAE, topk_relu

__all__ = [
    "SplitTopKSAE",
    "normalized_mse",
    "split_sae_loss",
    "symmetric_info_nce",
    "topk_relu",
]

__version__ = "0.1.0"

