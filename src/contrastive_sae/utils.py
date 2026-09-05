from __future__ import annotations

import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ContextManager

import numpy as np
import torch
from torch import Tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def autocast_context(device: torch.device, precision: str) -> ContextManager[Any]:
    if device.type != "cuda" or precision == "float32":
        return nullcontext()
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported mixed precision mode: {precision}")


def contrastive_weight_at_step(
    step: int,
    target_weight: float,
    start_step: int,
    ramp_steps: int,
) -> float:
    if step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return target_weight
    progress = min(1.0, (step - start_step + 1) / ramp_steps)
    return target_weight * progress


def cosine_warmup_lambda(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-8)
    denominator = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / denominator))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_torch_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def tensor_float(value: Tensor) -> float:
    return float(value.detach().float().cpu())

