# Contrastive Activation SAE

A minimal research implementation of a **high/low Split TopK sparse
autoencoder** trained on sentence-level Gemma residual activations.

The implementation intentionally has only two objectives:

1. Both branches jointly reconstruct the original activation.
2. Only the high-level sparse code receives a symmetric InfoNCE loss.

The low-level branch receives **no contrastive, adversarial, orthogonality, or
pair-difference loss**. It learns solely by reconstructing the residual left by
the high-level branch.

## Model

For a normalized, mean-pooled layer activation `x`:

```text
x -> High TopK SAE -> x_high
                      |
                      v
              residual = x - x_high
                      |
                      v
                Low TopK SAE -> x_low

x_hat = x_high + x_low
```

For a paraphrase pair `(x1, x2)`:

```text
loss = reconstruction_nmse(x1, xhat1)
     + reconstruction_nmse(x2, xhat2)
     + contrastive_weight * symmetric_info_nce(z_high1, z_high2)
```

The high and low dictionaries, encoders, and TopK budgets are separate. They
share only the final reconstruction objective.

## What is extracted

The default configuration freezes `google/gemma-2-2b`, captures the output of
decoder block index 12 (`resid_post`), and performs a masked mean over its token
dimension:

```text
[batch, sequence, 2304] -> [batch, 2304]
```

This project trains on the residual activation itself. It does **not** train on
attention weights or on the difference between pre- and post-attention states.

## Setup

Python 3.10+ and a CUDA GPU are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Accept the Gemma model license on Hugging Face and authenticate if your
environment requires it:

```bash
huggingface-cli login
```

## 1. Cache paired activations

The default dataset is the positive-pair subset of Quora Duplicate Questions.
It exposes `anchor` and `positive` columns and contains about 149k positive
pairs. Dataset licensing and attribution remain the responsibility of the
researcher.

The MVP configuration downloads and caches the Hugging Face dataset
automatically. No local dataset file is required. All dataset artifacts are
kept below the configured shared directory:

```text
/ehsan-rw/kosar/
├── dataset/
│   ├── huggingface/      # Raw/downloaded Hugging Face dataset cache
│   └── activations/mvp/  # Extracted paired Gemma activations
└── checkpoints/mvp/      # Trained SAE checkpoints and metrics
```

The extractor creates these subdirectories automatically. The user running the
command must have write permission on `/ehsan-rw/kosar/dataset`.

```bash
csae-extract --config configs/mvp.yaml
```

For a short extraction smoke test:

```bash
csae-extract --config configs/mvp.yaml --max-pairs 1000
```

To use a local JSONL file instead, copy the configuration and set:

```yaml
data:
  source: jsonl
  jsonl_path: examples/pairs.jsonl
  text1_column: sentence1
  text2_column: sentence2
```

The extractor writes memory-mapped NumPy arrays plus training-set
normalization statistics to the configured absolute activation path:

```text
/ehsan-rw/kosar/dataset/activations/mvp/
├── train_x1.npy
├── train_x2.npy
├── validation_x1.npy
├── validation_x2.npy
├── mean.npy
├── scale.npy
└── manifest.json
```

## 2. Train

```bash
csae-train --config configs/mvp.yaml
```

Short training smoke test:

```bash
csae-train --config configs/mvp.yaml --steps 50 --num-workers 0
```

Pass `--overwrite` to either command when intentionally replacing an existing
activation cache or training run.

Training outputs are written to the configured shared checkpoint directory,
`/ehsan-rw/kosar/checkpoints/mvp/`:

```text
best.pt
final.pt
latest.pt
metrics.jsonl
```

## 3. Evaluate

```bash
csae-evaluate \
  --config configs/mvp.yaml \
  --checkpoint /ehsan-rw/kosar/checkpoints/mvp/best.pt
```

Evaluation reports:

- reconstruction NMSE;
- symmetric InfoNCE;
- high-code paraphrase cosine similarity;
- high-code in-batch-negative cosine similarity;
- semantic similarity gap;
- low-code paraphrase cosine similarity;
- high/low reconstruction contribution ratios;
- swap reconstruction error;
- dead-feature fractions and average active counts.

## 4. Tests

```bash
pytest
```

The gradient-routing tests verify the central design contract:

- reconstruction updates both high and low branches;
- contrastive loss updates only the high encoder;
- each branch obeys its own TopK budget.

## Recommended experiment sequence

1. Overfit a tiny cached dataset and confirm reconstruction decreases.
2. Run 100k positive pairs with the MVP configuration.
3. Compare a vanilla TopK SAE, the split SAE with contrastive weight zero, and
   the full split contrastive SAE.
4. Only after the MVP is healthy, add hard negatives such as PAWS examples.

## Important limitation

Calling the branches “high-level” and “low-level” is an operational hypothesis,
not a guarantee. The high branch is the branch constrained to be stable across
paraphrases; the low branch is a limited-capacity residual corrector. Feature
inspection and causal evaluation are still required before making semantic or
mechanistic claims.

## References behind the minimal design

- Gao et al., *Scaling and evaluating sparse autoencoders* (TopK SAE).
- Bussmann et al., *BatchTopK Sparse Autoencoders* (possible later upgrade).
- Rajamanoharan et al., *Jumping Ahead* (possible later upgrade).
- Gao et al., *SimCSE* (symmetric contrastive sentence representations).
- Lieberum et al., *Gemma Scope* (evaluation and decoder-normalization practice).
