# PatchTST — Long-term Time Series Forecasting with Transformers

Implementation and study of the paper **"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"**, applied to the ETT (Electricity Transformer Temperature) benchmark datasets.

> 📓 [Google Colab Notebook](https://colab.research.google.com/drive/11GJyRLQ5xPXld0j9332I0OIzc1Nj_Fy_#scrollTo=WbuslqS_wZpe)

---

## Overview

PatchTST is a Transformer-based framework for multivariate time series forecasting. Rather than processing individual time steps, it treats the input series as a sequence of **patches** (overlapping sub-sequences), enabling the model to capture local semantic patterns while drastically reducing the quadratic cost of self-attention.

This implementation follows the original architecture with one adaptation for computational efficiency: `d_model=128` instead of the standard 512, optimised for CPU execution while maintaining the standard depth of 3 encoder layers.

---

## Architecture

### Patching & Channel Independence

The input series `[Batch, Channels, Seq_Len]` is segmented into overlapping patches via a `Patcher` module. Each channel is processed **independently** by the Transformer backbone — a design choice called *Channel Independence* that prevents cross-channel leakage during training and improves generalisation.

Patch parameters (configurable in `configs/default.yaml`):
- `patch_len`: length of each patch (default: 16)
- `stride`: step between consecutive patches (default: 8)
- `n_patches`: computed as `floor((seq_len - patch_len) / stride) + 2`

The final patch uses padding (last observed value repeated) to ensure coverage of the full input window.

### Reversible Instance Normalisation (RevIN)

To handle the non-stationary nature of time series, the backbone applies **RevIN**: the input is normalised to zero mean and unit variance before the Transformer, and the output is de-normalised before returning predictions. This mitigates distribution shift between train and test windows.

### Transformer Backbone

The backbone uses a stack of `TransformerBatchNormEncoderLayer` modules — a variant of the standard encoder layer that replaces LayerNorm with **BatchNorm1d** applied along the feature dimension. Each layer consists of:

1. Multi-head self-attention (`nn.MultiheadAttention`)
2. Dropout + residual connection
3. BatchNorm1d (transposed to `[B, D, L]` for compatibility)
4. Two-layer feed-forward network (Linear → GELU → Dropout → Linear)
5. Dropout + residual connection
6. BatchNorm1d

Learnable positional embeddings `W_pos` (shape `[1, n_patches, latent_dim]`, initialised with uniform noise in `[-0.02, 0.02]`) are added to patch projections before encoding.

### Dual-Head Design

The backbone supports two operating modes controlled by `mask_ratio`:

| Mode | `mask_ratio` | Head used | Output |
|---|---|---|---|
| Forecasting | `0.0` | `head_forecast` (Linear) | `[B, C, pred_len]` |
| Pre-training | `> 0.0` | `head_pretrain` (Linear) | `[B, C, n_patches, patch_len]` + mask |

---

## Training Pipeline

### Self-Supervised Pre-training

Inspired by BERT's Masked Language Modelling and Masked Autoencoders in Vision, the model is first trained to **reconstruct randomly masked patches** without any labels. A random fraction (`mask_ratio=0.4` by default) of patches is zeroed out; the model must recover the original normalised values.

The loss is computed only over masked positions:

```
loss = sum(MSE(output, target) * mask) / sum(mask)
```

### Supervised Fine-tuning (Linear Probing)

After pre-training, the backbone weights are loaded and **frozen**. Only the `head_forecast` linear layer is trained on the labelled forecasting task. This *linear probing* strategy is efficient and helps prevent overfitting when labelled data is scarce.

The fine-tuning phase uses:
- **Optimiser**: Adam (filtered to trainable parameters only)
- **Scheduler**: OneCycleLR with 30% warmup
- **Early stopping**: patience of 10 epochs, best checkpoint saved automatically

---

## Project Structure

```
.
├── configs/
│   └── default.yaml          # All hyperparameters and phase configs
├── src/
│   ├── patcher.py             # Patch extraction with optional masking
│   ├── transformer_blackbone.py  # Full model (backbone + heads)
│   ├── dataset.py             # ETT dataset loader with train/val/test splits
│   └── train.py               # Training loop, EarlyStopping, config loader
├── run_benchmark.py           # Benchmark runner across datasets/horizons
└── requirements.txt
```

---

## Configuration

All hyperparameters are managed in `configs/default.yaml`:

```yaml
data:
  root_path: "./dataset/ETT-small/"
  data_path: "ETTh1.csv"
  n_channels: 7

model:
  seq_len: 336
  pred_len: 96
  patch_len: 16
  stride: 8
  d_model: 128
  dropout: 0.2

training:
  batch_size: 32
  patience: 10
  learning_rate: 0.0001

phases:
  pretrain:
    mask_ratio: 0.4
    epochs: 50
  forecast:
    mask_ratio: 0.0
    epochs: 20
```

---

## Dataset

The model is evaluated on the **ETT (Electricity Transformer Temperature)** dataset family, which contains measurements from electricity transformers at 1-hour (`ETTh1`, `ETTh2`) and 15-minute (`ETTm1`, `ETTm2`) resolution. All datasets have 7 channels and use a standard 12/4/4 month train/val/test split.

Data is automatically scaled using `StandardScaler` fitted on the training split only.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and PyTorch 2.x. GPU support is available but not required — the default config targets CPU.

---

## Usage

**Run the full pre-train → fine-tune pipeline:**

```bash
python run_benchmark.py
```

**Run a single training phase directly:**

```bash
python src/train.py
```

**Run model sanity checks:**

```bash
python src/transformer_blackbone.py
python src/patcher.py
```

---

## Results

The notebook includes a comparative analysis of:
- Model trained **from scratch** (supervised only)
- Model **fine-tuned** from a self-supervised pre-trained backbone

All experiments are run on the ETT datasets across prediction horizons of 96, 192, 336, and 720 steps.

---

## License

Distributed under the Apache License 2.0. See `LICENSE` for details.
