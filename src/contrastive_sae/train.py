from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import load_config, public_config, resolve_project_path
from .data import (
    PairedActivationDataset,
    load_normalization_stats,
    normalize_batch,
)
from .losses import split_sae_loss
from .metrics import evaluate_model
from .model import SplitTopKSAE
from .utils import (
    append_jsonl,
    atomic_torch_save,
    autocast_context,
    choose_device,
    contrastive_weight_at_step,
    cosine_warmup_lambda,
    seed_everything,
    tensor_float,
)


def make_loader(
    dataset: PairedActivationDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    effective_batch_size = min(batch_size, len(dataset))
    if effective_batch_size < 2:
        raise ValueError("A data split must contain at least two pairs")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
        generator=generator,
    )


def checkpoint_payload(
    model: SplitTopKSAE,
    config: dict[str, Any],
    step: int,
    validation: dict[str, float] | None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "step": step,
        "model": model.state_dict(),
        "config": public_config(config),
        "validation": validation,
    }


def prepare_checkpoint_directory(path: Path, overwrite: bool) -> None:
    tracked = [
        path / "best.pt",
        path / "final.pt",
        path / "latest.pt",
        path / "metrics.jsonl",
        path / "run_summary.json",
    ]
    existing = [item for item in tracked if item.exists()]
    if existing and not overwrite:
        names = ", ".join(item.name for item in existing)
        raise FileExistsError(
            f"Training outputs already exist ({names}). Pass --overwrite to "
            "start a new run."
        )
    path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for item in existing:
            item.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    seed_everything(seed)

    training = config["training"]
    total_steps = args.steps if args.steps is not None else int(training["steps"])
    if total_steps <= 0:
        raise ValueError("Training steps must be positive")
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(training.get("num_workers", 0))
    )
    requested_device = args.device or config["activation"].get("device", "auto")
    device = choose_device(requested_device)

    activation_dir = resolve_project_path(config, config["paths"]["activation_dir"])
    checkpoint_dir = resolve_project_path(config, config["paths"]["checkpoint_dir"])
    prepare_checkpoint_directory(checkpoint_dir, overwrite=args.overwrite)

    train_dataset = PairedActivationDataset(activation_dir, "train")
    validation_dataset = PairedActivationDataset(activation_dir, "validation")
    if train_dataset.expected_dim != int(config["sae"]["input_dim"]):
        raise ValueError("Cached activation dimension does not match SAE config")

    train_loader = make_loader(
        train_dataset,
        int(training["batch_size"]),
        num_workers,
        shuffle=True,
        seed=seed,
        device=device,
    )
    validation_loader = make_loader(
        validation_dataset,
        int(training["batch_size"]),
        num_workers,
        shuffle=False,
        seed=seed + 1,
        device=device,
    )

    mean, scale = load_normalization_stats(activation_dir)
    mean = mean.to(device)
    model = SplitTopKSAE.from_config(config["sae"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training["learning_rate"])
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_warmup_lambda(
            step,
            warmup_steps=int(training["warmup_steps"]),
            total_steps=total_steps,
            min_ratio=float(training.get("min_learning_rate_ratio", 0.1)),
        ),
    )

    precision = training.get("mixed_precision", "bf16")
    use_scaler = device.type == "cuda" and precision == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    train_iterator = iter(train_loader)
    best_validation = float("inf")
    last_validation: dict[str, float] | None = None

    print(
        f"Training on {device} with {len(train_dataset):,} pairs, "
        f"batch={train_loader.batch_size}, steps={total_steps:,}"
    )

    for step in range(1, total_steps + 1):
        try:
            raw_first, raw_second = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            raw_first, raw_second = next(train_iterator)

        first = normalize_batch(
            raw_first.to(device, non_blocking=True), mean, scale
        )
        second = normalize_batch(
            raw_second.to(device, non_blocking=True), mean, scale
        )
        current_contrastive_weight = contrastive_weight_at_step(
            step,
            target_weight=float(training["contrastive_weight"]),
            start_step=int(training.get("contrastive_start_step", 0)),
            ramp_steps=int(training.get("contrastive_ramp_steps", 0)),
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, precision):
            first_output = model(first)
            second_output = model(second)
            losses = split_sae_loss(
                first_output,
                second_output,
                first,
                second,
                contrastive_weight=current_contrastive_weight,
                temperature=float(training["temperature"]),
            )

        scaler.scale(losses["loss"]).backward()
        if float(training.get("gradient_clip_norm", 0.0)) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
        scaler.step(optimizer)
        scaler.update()
        model.normalize_decoder_columns_()
        scheduler.step()

        if step == 1 or step % int(training["log_every"]) == 0:
            high_active = 0.5 * (
                (first_output["high_code"] > 0).sum(dim=-1).float().mean()
                + (second_output["high_code"] > 0).sum(dim=-1).float().mean()
            )
            low_active = 0.5 * (
                (first_output["low_code"] > 0).sum(dim=-1).float().mean()
                + (second_output["low_code"] > 0).sum(dim=-1).float().mean()
            )
            record = {
                "split": "train",
                "step": step,
                "loss": tensor_float(losses["loss"]),
                "reconstruction_nmse": tensor_float(losses["reconstruction"]),
                "contrastive": tensor_float(losses["contrastive"]),
                "contrastive_weight": current_contrastive_weight,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "high_active_mean": tensor_float(high_active),
                "low_active_mean": tensor_float(low_active),
            }
            append_jsonl(checkpoint_dir / "metrics.jsonl", record)
            print(
                f"step={step:>6} loss={record['loss']:.4f} "
                f"rec={record['reconstruction_nmse']:.4f} "
                f"ctr={record['contrastive']:.4f} "
                f"lambda={current_contrastive_weight:.3f}"
            )

        should_validate = step % int(training["validate_every"]) == 0
        if should_validate or step == total_steps:
            last_validation = evaluate_model(
                model,
                validation_loader,
                mean,
                scale,
                device,
                temperature=float(training["temperature"]),
                contrastive_weight=float(training["contrastive_weight"]),
                precision=precision,
                max_batches=int(training["validation_batches"]),
            )
            validation_record: dict[str, Any] = {
                "split": "validation",
                "step": step,
                **last_validation,
            }
            append_jsonl(checkpoint_dir / "metrics.jsonl", validation_record)
            print(
                f"validation step={step:>6} "
                f"loss={last_validation['loss']:.4f} "
                f"rec={last_validation['reconstruction_nmse']:.4f} "
                f"gap={last_validation['semantic_gap']:.4f}"
            )
            if last_validation["loss"] < best_validation:
                best_validation = last_validation["loss"]
                atomic_torch_save(
                    checkpoint_payload(
                        model, config, step, last_validation
                    ),
                    checkpoint_dir / "best.pt",
                )

        if step % int(training["save_every"]) == 0:
            atomic_torch_save(
                checkpoint_payload(
                    model, config, step, last_validation
                ),
                checkpoint_dir / "latest.pt",
            )

    atomic_torch_save(
        checkpoint_payload(
            model, config, total_steps, last_validation
        ),
        checkpoint_dir / "final.pt",
    )
    with (checkpoint_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "steps": total_steps,
                "best_validation_loss": best_validation,
                "last_validation": last_validation,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"Saved final checkpoint to {checkpoint_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
