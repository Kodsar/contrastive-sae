from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration and validate required sections."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")

    required_sections = ("paths", "data", "activation", "sae", "training")
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")

    validate_config(config)
    config["_config_path"] = str(config_path.resolve())
    return config


def validate_config(config: dict[str, Any]) -> None:
    sae = config["sae"]
    training = config["training"]
    data = config["data"]
    activation = config["activation"]

    positive_integer_fields = {
        "sae.input_dim": sae.get("input_dim"),
        "sae.high_features": sae.get("high_features"),
        "sae.low_features": sae.get("low_features"),
        "sae.high_topk": sae.get("high_topk"),
        "sae.low_topk": sae.get("low_topk"),
        "training.batch_size": training.get("batch_size"),
        "training.steps": training.get("steps"),
        "training.log_every": training.get("log_every"),
        "training.validate_every": training.get("validate_every"),
        "training.validation_batches": training.get("validation_batches"),
        "training.save_every": training.get("save_every"),
        "activation.max_length": activation.get("max_length"),
        "activation.extraction_batch_size": activation.get(
            "extraction_batch_size"
        ),
    }
    for name, value in positive_integer_fields.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")

    if sae["high_topk"] > sae["high_features"]:
        raise ValueError("sae.high_topk cannot exceed sae.high_features")
    if sae["low_topk"] > sae["low_features"]:
        raise ValueError("sae.low_topk cannot exceed sae.low_features")

    if training.get("temperature", 0.0) <= 0:
        raise ValueError("training.temperature must be positive")
    if training.get("learning_rate", 0.0) <= 0:
        raise ValueError("training.learning_rate must be positive")
    if training.get("contrastive_weight", -1.0) < 0:
        raise ValueError("training.contrastive_weight cannot be negative")
    if int(training.get("num_workers", 0)) < 0:
        raise ValueError("training.num_workers cannot be negative")
    if int(training.get("warmup_steps", 0)) < 0:
        raise ValueError("training.warmup_steps cannot be negative")
    if int(training.get("contrastive_start_step", 0)) < 0:
        raise ValueError("training.contrastive_start_step cannot be negative")
    if int(training.get("contrastive_ramp_steps", 0)) < 0:
        raise ValueError("training.contrastive_ramp_steps cannot be negative")
    if not 0 <= float(data.get("validation_fraction", -1)) < 1:
        raise ValueError("data.validation_fraction must be in [0, 1)")
    if activation.get("pooling") not in {"masked_mean", "last_non_padding"}:
        raise ValueError(
            "activation.pooling must be 'masked_mean' or 'last_non_padding'"
        )
    if data.get("source") not in {"huggingface", "jsonl"}:
        raise ValueError("data.source must be 'huggingface' or 'jsonl'")
    layer_index = activation.get("layer_index")
    if not isinstance(layer_index, int) or layer_index < 0:
        raise ValueError("activation.layer_index must be a non-negative integer")


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a checkpoint-safe copy without private runtime fields."""
    result = deepcopy(config)
    for key in list(result):
        if key.startswith("_"):
            result.pop(key)
    return result


def resolve_project_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve paths relative to the project root containing ``configs/``."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    config_path = Path(config["_config_path"])
    project_root = config_path.parent.parent
    return (project_root / path).resolve()
