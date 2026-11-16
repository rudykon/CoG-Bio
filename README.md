<a id="top"></a>

<div align="center">

<h1>🩺 CoG-Bio</h1>

<p><strong>Consistency + Geometry for Structured Ultrasound Biometry</strong></p>
<p><strong>Co</strong>nsistency · <strong>G</strong>eometry · <strong>Bio</strong>metry</p>

<p>
  <a href="#installation"><img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Recommended Python versions: 3.11 or 3.12"></a>
  <a href="requirements.txt"><img src="https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?style=flat-square&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch 2.6.0"></a>
  <a href="#highlights"><img src="https://img.shields.io/badge/Encoder-DINOv3-7462B2?style=flat-square" alt="DINOv3 encoder"></a>
  <a href="configs/landmark_schema.yaml"><img src="https://img.shields.io/badge/Ultrasound_tasks-9-278C91?style=flat-square" alt="Nine ultrasound tasks"></a>
</p>

<p>
  <a href="#installation">🚀 Installation</a> ·
  <a href="#inference">🎯 Inference</a> ·
  <a href="#training">🏋️ Training</a> ·
  <a href="docs/data_format.md">📋 Data format</a> ·
  <a href="VERIFICATION.md">✅ Verification</a> ·
  <a href="README_zh.md">🌐 中文</a>
</p>

</div>

---

Implementation of CoG-Bio for structured ultrasound biometry. The final
23.67 px configuration consists of a frozen DINOv3 encoder, merge-refined predictor (MRP),
independently adapted predictor (IAP), fixed flip/crop views, bounded consistency
fusion, conditional cardiac shape repair and HC/FA caliper projection.

This repository contains source code, configurations and tests. Manuscripts,
publication figures, medical images and neural checkpoints are not included.

<a id="highlights"></a>

## ✨ Highlights

| 🧠 Shared foundation | 🔀 Complementary predictors | 📐 Structured geometry |
|---|---|---|
| Frozen DINOv3 features with low-rank adaptation across nine tasks. | MRP and IAP combine distinct training histories with bounded, image-level consistency weights. | Training-fitted shape repair and caliper projection enforce the annotation geometry. |

<a id="results"></a>

## 📊 Evaluation snapshot

| Measure | Result |
|---|---:|
| Ultrasound tasks | **9** |
| Released benchmark images | **1,281** |
| Task-macro MRE in original image pixels | **23.67 px** |
| Saved-prediction replay: maximum coordinate difference | **0 px** |

See [reproduction notes](docs/reproduction.md) for the fixed configuration and
[verification](VERIFICATION.md) for numerical checks and their scope.

<a id="installation"></a>

## 🚀 Installation

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

<a id="assets"></a>

## 📦 Required external assets

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

<a id="inference"></a>

## 🎯 Inference and evaluation

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

<a id="training"></a>

## 🏋️ Training the final configuration

The following stages use the fixed lengths selected for the final training
partition. Supply the same ordered, identity-excluded training manifest to each
stage. `--exclude` can additionally check sample and identity-group separation
from an excluded manifest. The exact paper score depends on the original split
and learned artifacts; generating a random split does not reproduce it.

<details>
<summary><strong>Expand the complete training sequence: MRP parents → merge → refinement, IAP and shape prior</strong></summary>

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

</details>

Copy the final two selected checkpoints to your weights directory using the
names in `configs/inference.json`. When retraining on a different partition,
pass the training-fitted prior with `predict.py --prior runs/shape_prior.json`.
Never fit the prior or select training lengths with evaluation annotations.

The development MRP/IAP recipes require a separate `--validation` manifest.
MRP development additionally requires an appropriate development-only merged
parent; the final-fit parent must not be reused for development evaluation.
Final recipes refuse a validation manifest and save the predetermined state.

<a id="source"></a>

## 🗂️ Source organization

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

The existing `clingeo` Python import path is retained for compatibility.

## ✅ Checks and documentation

The recorded local suite has **96 passing tests**, with **1 CUDA-only test skipped**
when CUDA is unavailable. Run `make test` or `make coverage` to check your setup.

| Resource | Contents |
|---|---|
| 📋 [Data format](docs/data_format.md) | Image manifests, labels and endpoint records |
| 🔁 [Reproduction notes](docs/reproduction.md) | Training stages, fusion settings and coordinate conventions |
| 🔎 [Verification report](VERIFICATION.md) | Test coverage, forward parity and saved-prediction replay |
| 📚 [External components](THIRD_PARTY.md) | Pretrained encoder, dataset and third-party terms |

<p align="center"><a href="#top">⬆ Back to top</a></p>
