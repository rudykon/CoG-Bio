"""Command-line entry points. All data and neural weights are supplied locally."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--schema", type=Path, default=ROOT / "configs/landmark_schema.yaml")
    return p


def predict_main():
    from .inference import predict
    p = parser("Run the fixed CoG-Bio MRP/IAP inference pipeline")
    for name in ("manifest", "image-root", "encoder", "weights", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/inference.json")
    p.add_argument("--prior", type=Path, default=ROOT / "configs/shape_prior.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--verify-paper-checksums", action="store_true")
    a = p.parse_args()
    predict(a.manifest, a.image_root, a.encoder, a.weights, a.output, a.schema,
            a.config, a.prior, a.device, a.workers, a.batch_size, a.verify_paper_checksums)


def fuse_main():
    from .io import read_json, read_rows, write_json
    from .data.task_schema import load_landmark_schema
    from .pipeline import combine_views
    p = parser("Fuse saved MRP flip/crop and IAP flip/crop records, then correct geometry")
    p.add_argument("--views", nargs=4, type=Path, required=True,
                   metavar=("MRP_FLIP", "MRP_CROP", "IAP_FLIP", "IAP_CROP"))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/inference.json")
    p.add_argument("--prior", type=Path, default=ROOT / "configs/shape_prior.json")
    a = p.parse_args()
    predictions, _ = combine_views([read_json(f) for f in a.views], read_rows(a.manifest),
        load_landmark_schema(a.schema), read_json(a.config), read_json(a.prior))
    if a.output.exists():
        raise FileExistsError(a.output)
    write_json(a.output, predictions)


def evaluate_main():
    from .io import read_json, read_rows, write_json
    from .data.task_schema import load_landmark_schema
    from .evaluation import score_predictions
    p = parser("Evaluate task-macro MRE in original image pixels")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-subset", action="store_true", help="Diagnostic only; not the paper's nine-task score")
    a = p.parse_args()
    result = score_predictions(read_json(a.predictions), read_rows(a.labels),
        load_landmark_schema(a.schema), require_all_tasks=not a.allow_subset)
    if a.output.exists():
        raise FileExistsError(a.output)
    write_json(a.output, result)
    print(f"Task-macro MRE: {result['macro_mre_pixels']:.6f} px ({result['images']} images)")


def train_main():
    from .training.runner import train
    p = parser("Train one fixed CoG-Bio stage from an explicit manifest")
    for name in ("config", "manifest", "image-root", "encoder", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("initial", "validation", "exclude"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    train(a.config, a.manifest, a.image_root, a.encoder, a.schema, a.output,
          a.device, a.initial, a.validation, a.exclude, a.workers)


def merge_main():
    import torch
    from .training.merge import merge_lora_states, merged_model_config
    from .training.state import save_checkpoint
    p = argparse.ArgumentParser(description="Merge adapters: A=[A1;A2], B=0.5[B1,B2]")
    p.add_argument("--first", type=Path, required=True)
    p.add_argument("--second", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    first, second = [torch.load(f, map_location="cpu", weights_only=True) for f in (a.first, a.second)]
    configs = [{**v["model_config"], "weight_dir": "ENCODER_DIR"} for v in (first, second)]
    payload = {"model_config": merged_model_config(*configs),
               "state_dict": merge_lora_states(first["state_dict"], second["state_dict"]),
               "selected_state": "mean_adaptation_residuals"}
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(a.output, payload)


def fit_prior_main():
    import pandas as pd
    from .shape import fit_prior
    from .io import write_json
    from .data.task_schema import load_landmark_schema
    from .training.runner import validate_training_frame, disjoint
    p = parser("Fit the cardiac shape prior from training labels only")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--exclude", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    frame = validate_training_frame(pd.read_csv(a.manifest), load_landmark_schema(a.schema))
    if a.exclude:
        disjoint(frame, pd.read_csv(a.exclude))
    if a.output.exists():
        raise FileExistsError(a.output)
    write_json(a.output, fit_prior(frame))


def prepare_encoder_main():
    from .model import ENCODER_NAME, ENCODER_REVISION, ENCODER_SHA256, sha256_file
    from .io import write_json
    p = argparse.ArgumentParser(description="Verify a locally downloaded DINOv3 artifact; no network access")
    p.add_argument("--encoder", type=Path, required=True, help="Directory containing model.safetensors")
    a = p.parse_args()
    if sha256_file(a.encoder / "model.safetensors") != ENCODER_SHA256:
        raise ValueError("encoder does not match the pinned paper artifact")
    target = a.encoder / "provenance.json"
    if target.exists():
        raise FileExistsError("provenance.json already exists; keep the existing verified provenance")
    write_json(target, {"repo_id": f"timm/{ENCODER_NAME}", "model_name": ENCODER_NAME,
        "revision": ENCODER_REVISION, "files": {"model.safetensors": {"sha256": ENCODER_SHA256}}})
