# CoG-Bio

CoG-Bio combines **Co**nsistency, **G**eometry and **Bio**metry. The existing
`clingeo` Python import path is retained for compatibility.

Implementation of CoG-Bio for structured ultrasound biometry. The final
23.67 px configuration consists of a frozen DINOv3 encoder, merge-refined predictor (MRP),
independently adapted predictor (IAP), fixed flip/crop views, bounded consistency
fusion, conditional cardiac shape repair and HC/FA caliper projection.

This repository contains source code, configurations and tests. Manuscripts,
publication figures, medical images and neural checkpoints are not included.

See [中文说明](README_zh.md), [data format](docs/data_format.md),
[reproduction notes](docs/reproduction.md) and [verification](VERIFICATION.md).

## Installation

Use Python 3.11 or 3.12 in a dedicated environment. Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Run the commands below from this directory. Scripts resolve their default
configuration files relative to their own location. No command downloads images,
models or code automatically. CUDA inference uses bfloat16 encoder features and
float32 geometric heads. `--device cpu --workers 0` supports diagnostic runs.

## Required external assets

1. Obtain the challenge images and annotations through the organizers. Preserve
   the corrected FA endpoint ordering and the released image dimensions.
2. Download `model.safetensors` from the pinned
   `timm/vit_base_patch16_dinov3.lvd1689m` model repository, revision
   `c6a5fb7d12bbd3cf3b0079253141c3332aaed7da`, under its applicable terms.
   Place it in an encoder directory, then run:

   ```bash
   python scripts/prepare_encoder.py --encoder /path/to/encoder
   ```

3. Supply the learned MRP and IAP checkpoints as `mrp.pth` and `iap.pth` in one
   weights directory, or train them with the commands below. Each checkpoint
   contains `model_config` and `state_dict`; the frozen encoder is loaded
   separately. The package contains the fitted aggregate shape prior but no
   neural checkpoints, dataset images, individual labels or case predictions.

## Inference and evaluation

```bash
python scripts/predict.py \
  --manifest /path/to/images.csv --image-root /path/to/image_root \
  --encoder /path/to/encoder --weights /path/to/adaptations \
  --output runs/prediction --device cuda:0

python scripts/evaluate.py \
  --predictions runs/prediction/predictions.json \
  --labels /path/to/labels.csv --output runs/score.json
```

Inference reads only the image manifest. It writes four view predictions, final
`predictions.json` and a receipt. Evaluation reads labels separately and computes
the mean over valid landmarks within each image, the mean over images within
each task, then the equally weighted mean over the nine tasks. The units are
original image pixels. Missing, duplicate or malformed predictions are errors.

`--verify-paper-checksums` checks whether the supplied checkpoints are the exact
recorded paper artifacts. Omit it for newly trained checkpoints. The encoder
artifact is always verified. Output directories must be new.

To apply only the fusion and geometric stages to saved view predictions:

```bash
python scripts/fuse.py \
  --views runs/prediction/mrp_flip_mean.json runs/prediction/mrp_crop_mean.json \
          runs/prediction/iap_flip_mean.json runs/prediction/iap_crop_mean.json \
  --manifest /path/to/images.csv --output runs/fused.json
```

## Training the final configuration

The following stages use the fixed lengths selected for the final training
partition. Supply the same ordered, identity-excluded training manifest to each
stage. `--exclude` can additionally check sample and identity-group separation
from an excluded manifest. The exact paper score depends on the original split
and learned artifacts; generating a random split does not reproduce it.

```bash
COGBIO_TRAIN=/path/to/train.csv
COGBIO_DATA=/path/to/image_root
COGBIO_ENCODER=/path/to/encoder

# Rank-8 parent: warmup and continuation.
python scripts/train.py --config configs/train_mrp_base_final.json \
  --manifest "$COGBIO_TRAIN" --image-root "$COGBIO_DATA" \
  --encoder "$COGBIO_ENCODER" --output runs/mrp_base

# Grow the parent to rank 32 and refine its spatial layout.
python scripts/train.py --config configs/train_mrp_layout_final.json \
  --manifest "$COGBIO_TRAIN" --image-root "$COGBIO_DATA" \
  --encoder "$COGBIO_ENCODER" --initial runs/mrp_base/selected_checkpoint.pth \
  --output runs/mrp_layout

# The merged rank-40 residual is the mean of the two parent residuals.
python scripts/merge_adapters.py \
  --first runs/mrp_base/selected_checkpoint.pth \
  --second runs/mrp_layout/selected_checkpoint.pth --output runs/mrp_merged.pth

python scripts/train.py --config configs/train_mrp_final.json \
  --manifest "$COGBIO_TRAIN" --image-root "$COGBIO_DATA" \
  --encoder "$COGBIO_ENCODER" --initial runs/mrp_merged.pth --output runs/mrp

# IAP starts independently from the same frozen encoder.
python scripts/train.py --config configs/train_iap_final.json \
  --manifest "$COGBIO_TRAIN" --image-root "$COGBIO_DATA" \
  --encoder "$COGBIO_ENCODER" --output runs/iap

python scripts/fit_shape_prior.py --manifest "$COGBIO_TRAIN" \
  --output runs/shape_prior.json
```

Copy the final two selected checkpoints to your weights directory using the
names in `configs/inference.json`. When retraining on a different partition,
pass the training-fitted prior with `predict.py --prior runs/shape_prior.json`.
Never fit the prior or select training lengths with evaluation annotations.

The development MRP/IAP recipes require a separate `--validation` manifest.
MRP development additionally requires an appropriate development-only merged
parent; the final-fit parent must not be reused for development evaluation.
Final recipes refuse a validation manifest and save the predetermined state.

## Source organization

| Path | Purpose |
|---|---|
| `clingeo/model.py`, `precision.py`, `models/` | Frozen encoder, LoRA and task-conditioned heads |
| `clingeo/data/`, `images.py` | Image loading, augmentation and coordinate mapping |
| `clingeo/training/`, `losses/` | Stage training, losses, adapter growth/merge, EMA |
| `clingeo/views.py`, `fusion.py` | Fixed views and image-level model weighting |
| `clingeo/shape.py`, `projection.py` | Training-fitted repair and caliper geometry |
| `clingeo/inference.py`, `evaluation.py`, `io.py` | Portable inference, metrics and validation |
| `configs/` | Selected recipes, task schema and aggregate shape prior |
| `scripts/`, `tests/` | Command-line entry points and synthetic regression tests |

The schema retains its original task-index order, including clinical aliases,
to load the recorded checkpoints correctly. The evaluation task list contains
the nine dataset domains, not every alias entry in the schema.
