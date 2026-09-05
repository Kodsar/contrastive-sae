from __future__ import annotations

import argparse
import json

from torch.utils.data import DataLoader

from .config import load_config, resolve_project_path
from .data import PairedActivationDataset, load_normalization_stats
from .metrics import evaluate_model
from .model import SplitTopKSAE
from .utils import choose_device, load_torch_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split", choices=["train", "validation"], default="validation"
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    seed_everything(seed)

    requested_device = args.device or config["activation"].get("device", "auto")
    device = choose_device(requested_device)
    activation_dir = resolve_project_path(config, config["paths"]["activation_dir"])
    checkpoint_path = resolve_project_path(config, args.checkpoint)

    dataset = PairedActivationDataset(activation_dir, args.split)
    configured_batch_size = int(config["training"]["batch_size"])
    batch_size = args.batch_size or configured_batch_size
    batch_size = min(batch_size, len(dataset))
    if batch_size < 2:
        raise ValueError("Evaluation requires at least two pairs")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    mean, scale = load_normalization_stats(activation_dir)
    model = SplitTopKSAE.from_config(config["sae"]).to(device)
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"], strict=True)

    metrics = evaluate_model(
        model,
        loader,
        mean,
        scale,
        device,
        temperature=float(config["training"]["temperature"]),
        contrastive_weight=float(config["training"]["contrastive_weight"]),
        precision=config["training"].get("mixed_precision", "bf16"),
        max_batches=args.max_batches,
    )
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "split": args.split,
        "metrics": metrics,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = resolve_project_path(config, args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
