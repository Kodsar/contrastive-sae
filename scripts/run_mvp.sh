#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/mvp.yaml}"

csae-extract --config "$CONFIG_PATH"
csae-train --config "$CONFIG_PATH"
csae-evaluate \
  --config "$CONFIG_PATH" \
  --checkpoint checkpoints/mvp/best.pt

