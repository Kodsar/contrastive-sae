from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch import Tensor, nn
from tqdm import tqdm

from .config import load_config, public_config, resolve_project_path
from .utils import choose_device, seed_everything


@dataclass(frozen=True)
class TextPair:
    first: str
    second: str


class RunningActivationStats:
    """Accumulate a vector mean and one global centered RMS scale."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.count = 0
        self.sum = np.zeros(dimension, dtype=np.float64)
        self.sum_of_squares = 0.0

    def update(self, batch: np.ndarray) -> None:
        batch64 = np.asarray(batch, dtype=np.float64)
        if batch64.ndim != 2 or batch64.shape[1] != self.dimension:
            raise ValueError(
                f"Expected [batch, {self.dimension}], got {batch64.shape}"
            )
        self.count += batch64.shape[0]
        self.sum += batch64.sum(axis=0)
        self.sum_of_squares += float(np.square(batch64).sum())

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("Cannot compute statistics from zero activations")
        mean = self.sum / self.count
        mean_square = self.sum_of_squares / (self.count * self.dimension)
        centered_mean_square = mean_square - float(np.square(mean).mean())
        scale = np.sqrt(max(centered_mean_square, 1e-12))
        return mean.astype(np.float32), np.asarray(scale, dtype=np.float32)


def _clean_pair(first: Any, second: Any) -> TextPair | None:
    if not isinstance(first, str) or not isinstance(second, str):
        return None
    first = first.strip()
    second = second.strip()
    if not first or not second:
        return None
    return TextPair(first, second)


def load_text_pairs(
    data_config: dict[str, Any], seed: int, max_pairs: int | None
) -> list[TextPair]:
    first_column = data_config["text1_column"]
    second_column = data_config["text2_column"]
    pairs: list[TextPair] = []

    if data_config["source"] == "huggingface":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "The 'datasets' package is required for a Hugging Face source"
            ) from exc

        dataset_name = data_config["dataset_name"]
        dataset_config = data_config.get("dataset_config")
        split = data_config.get("split", "train")
        cache_dir_value = data_config.get("cache_dir")
        cache_dir: str | None = None
        if cache_dir_value:
            cache_path = Path(cache_dir_value).expanduser()
            cache_path.mkdir(parents=True, exist_ok=True)
            cache_dir = str(cache_path)

        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            cache_dir=cache_dir,
        )
        if data_config.get("shuffle", True):
            dataset = dataset.shuffle(seed=seed)

        required = {first_column, second_column}
        missing = required.difference(dataset.column_names)
        if missing:
            raise ValueError(
                f"Dataset is missing columns {sorted(missing)}; available columns: "
                f"{dataset.column_names}"
            )

        for row in dataset:
            pair = _clean_pair(row[first_column], row[second_column])
            if pair is not None:
                pairs.append(pair)
            if max_pairs is not None and len(pairs) >= max_pairs:
                break
    else:
        jsonl_path = Path(data_config["jsonl_path"]).expanduser()
        if not jsonl_path.is_absolute():
            jsonl_path = Path.cwd() / jsonl_path
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {jsonl_path}:{line_number}"
                    ) from exc
                pair = _clean_pair(row.get(first_column), row.get(second_column))
                if pair is not None:
                    pairs.append(pair)
                if max_pairs is not None and len(pairs) >= max_pairs:
                    break

        if data_config.get("shuffle", True):
            random.Random(seed).shuffle(pairs)

    if len(pairs) < 4:
        raise ValueError(
            f"At least four valid pairs are required; found {len(pairs)}"
        )
    return pairs


def find_decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    candidates = [model, getattr(model, "model", None)]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            layers = getattr(candidate, "layers")
            if isinstance(layers, (nn.ModuleList, list, tuple)):
                return layers
    raise AttributeError(
        "Could not locate transformer decoder layers. Expected model.layers or "
        "model.model.layers."
    )


class ResidualActivationExtractor:
    """Capture one decoder block's output and pool it into sentence vectors."""

    def __init__(
        self,
        model_name: str,
        layer_index: int,
        max_length: int,
        pooling: str,
        device: torch.device,
    ) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' package is required for activation extraction"
            ) from exc

        if device.type == "cuda":
            model_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            model_dtype = torch.float32

        self.device = device
        self.max_length = max_length
        self.pooling = pooling
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)

        layers = find_decoder_layers(self.model)
        if not 0 <= layer_index < len(layers):
            raise ValueError(
                f"layer_index={layer_index} is outside [0, {len(layers) - 1}]"
            )

        self.layer_index = layer_index
        self.hidden_size = int(self.model.config.hidden_size)
        self._captured: Tensor | None = None
        self._hook_handle = layers[layer_index].register_forward_hook(self._hook)

    def _hook(
        self,
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: Tensor | tuple[Any, ...],
    ) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, Tensor):
            raise TypeError("Decoder layer hook did not receive a tensor output")
        self._captured = hidden.detach()

    def close(self) -> None:
        self._hook_handle.remove()

    def __enter__(self) -> "ResidualActivationExtractor":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> Tensor:
        if not texts:
            return torch.empty(0, self.hidden_size)

        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        self._captured = None
        self.model(**encoded, use_cache=False, return_dict=True)
        if self._captured is None:
            raise RuntimeError("Layer hook did not capture an activation")

        hidden = self._captured.float()
        mask = encoded["attention_mask"].to(hidden.device)
        if self.pooling == "masked_mean":
            expanded_mask = mask.unsqueeze(-1).to(hidden.dtype)
            denominator = expanded_mask.sum(dim=1).clamp_min(1.0)
            pooled = (hidden * expanded_mask).sum(dim=1) / denominator
        elif self.pooling == "last_non_padding":
            final_indices = mask.sum(dim=1).sub(1).clamp_min(0)
            batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
            pooled = hidden[batch_indices, final_indices]
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")

        return pooled.cpu()


def batched(items: Sequence[TextPair], batch_size: int) -> Iterable[list[TextPair]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    expected_files = [
        "train_x1.npy",
        "train_x2.npy",
        "validation_x1.npy",
        "validation_x2.npy",
        "mean.npy",
        "scale.npy",
        "manifest.json",
    ]
    existing = [
        output_dir / name
        for name in expected_files
        if (output_dir / name).exists()
    ]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Output files already exist ({names}). Pass --overwrite to replace them."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()


def extract_split(
    name: str,
    pairs: Sequence[TextPair],
    extractor: ResidualActivationExtractor,
    output_dir: Path,
    batch_size: int,
    storage_dtype: np.dtype[Any],
    stats: RunningActivationStats | None,
) -> dict[str, Any]:
    dimension = extractor.hidden_size
    first_name = f"{name}_x1.npy"
    second_name = f"{name}_x2.npy"
    first_array = open_memmap(
        output_dir / first_name,
        mode="w+",
        dtype=storage_dtype,
        shape=(len(pairs), dimension),
    )
    second_array = open_memmap(
        output_dir / second_name,
        mode="w+",
        dtype=storage_dtype,
        shape=(len(pairs), dimension),
    )

    offset = 0
    progress = tqdm(total=len(pairs), desc=f"Extracting {name}", unit="pair")
    for batch_pairs in batched(pairs, batch_size):
        texts = [pair.first for pair in batch_pairs] + [
            pair.second for pair in batch_pairs
        ]
        pooled = extractor.encode(texts).numpy()
        pair_count = len(batch_pairs)
        first = pooled[:pair_count].astype(storage_dtype, copy=False)
        second = pooled[pair_count:].astype(storage_dtype, copy=False)
        next_offset = offset + pair_count
        first_array[offset:next_offset] = first
        second_array[offset:next_offset] = second
        if stats is not None:
            stats.update(first)
            stats.update(second)
        offset = next_offset
        progress.update(pair_count)
    progress.close()

    first_array.flush()
    second_array.flush()
    return {"pairs": len(pairs), "x1": first_name, "x2": second_name}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--max-pairs", type=int, default=None, help="Override data.max_pairs"
    )
    parser.add_argument("--device", default=None, help="Override activation.device")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing cache"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    seed_everything(seed)

    data_config = config["data"]
    activation_config = config["activation"]
    max_pairs = args.max_pairs
    if max_pairs is None:
        max_pairs = int(data_config["max_pairs"])
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")

    if data_config["source"] == "jsonl":
        jsonl_value = data_config.get("jsonl_path")
        if not jsonl_value:
            raise ValueError("data.jsonl_path is required for a JSONL source")
        data_config = dict(data_config)
        data_config["jsonl_path"] = str(resolve_project_path(config, jsonl_value))
    elif data_config.get("cache_dir"):
        data_config = dict(data_config)
        data_config["cache_dir"] = str(
            resolve_project_path(config, data_config["cache_dir"])
        )

    pairs = load_text_pairs(data_config, seed=seed, max_pairs=max_pairs)
    validation_fraction = float(data_config["validation_fraction"])
    validation_count = max(2, int(round(len(pairs) * validation_fraction)))
    if validation_count >= len(pairs):
        raise ValueError("Validation split leaves no training examples")
    validation_pairs = pairs[:validation_count]
    train_pairs = pairs[validation_count:]

    output_dir = resolve_project_path(config, config["paths"]["activation_dir"])
    prepare_output_directory(output_dir, overwrite=args.overwrite)

    requested_device = args.device or activation_config.get("device", "auto")
    device = choose_device(requested_device)
    storage_dtype_name = activation_config.get("storage_dtype", "float16")
    storage_types: dict[str, np.dtype[Any]] = {
        "float16": np.dtype(np.float16),
        "float32": np.dtype(np.float32),
    }
    if storage_dtype_name not in storage_types:
        raise ValueError("activation.storage_dtype must be float16 or float32")
    storage_dtype = storage_types[storage_dtype_name]

    with ResidualActivationExtractor(
        model_name=activation_config["model_name"],
        layer_index=int(activation_config["layer_index"]),
        max_length=int(activation_config["max_length"]),
        pooling=activation_config["pooling"],
        device=device,
    ) as extractor:
        configured_dim = int(config["sae"]["input_dim"])
        if extractor.hidden_size != configured_dim:
            raise ValueError(
                f"Model hidden size is {extractor.hidden_size}, but "
                f"sae.input_dim is {configured_dim}"
            )

        stats = RunningActivationStats(extractor.hidden_size)
        train_info = extract_split(
            "train",
            train_pairs,
            extractor,
            output_dir,
            int(activation_config["extraction_batch_size"]),
            storage_dtype,
            stats,
        )
        validation_info = extract_split(
            "validation",
            validation_pairs,
            extractor,
            output_dir,
            int(activation_config["extraction_batch_size"]),
            storage_dtype,
            stats=None,
        )

    mean, scale = stats.finalize()
    np.save(output_dir / "mean.npy", mean)
    np.save(output_dir / "scale.npy", scale)

    manifest = {
        "format_version": 1,
        "activation_dim": int(config["sae"]["input_dim"]),
        "storage_dtype": storage_dtype_name,
        "normalization": {"mean": "mean.npy", "scale": "scale.npy"},
        "splits": {"train": train_info, "validation": validation_info},
        "source": {
            "type": data_config["source"],
            "cache_dir": data_config.get("cache_dir"),
            "dataset_name": data_config.get("dataset_name"),
            "dataset_config": data_config.get("dataset_config"),
            "split": data_config.get("split"),
            "text1_column": data_config["text1_column"],
            "text2_column": data_config["text2_column"],
        },
        "activation": {
            "model_name": activation_config["model_name"],
            "layer_index": int(activation_config["layer_index"]),
            "hook": "decoder_block_output_resid_post",
            "pooling": activation_config["pooling"],
            "max_length": int(activation_config["max_length"]),
        },
        "config": public_config(config),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(f"Saved {len(train_pairs):,} training pairs to {output_dir}")
    print(f"Saved {len(validation_pairs):,} validation pairs")
    print(f"Activation scale: {float(scale):.6g}")


if __name__ == "__main__":
    main()
