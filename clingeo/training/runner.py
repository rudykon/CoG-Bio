"""Portable fixed-stage training recipes corresponding to the paper's predictors.

Development mode selects checkpoints on the supplied development split. Final
mode runs the already selected lengths and never reads evaluation labels.
"""
from contextlib import nullcontext
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from clingeo.data.dataset import BiometryDataset, collate_biometry
from clingeo.data.task_schema import load_landmark_schema
from clingeo.evaluation import score_predictions
from clingeo.images import original_coordinates
from clingeo.io import read_json, write_json
from clingeo.model import FoundationBiometry
from clingeo.precision import PreciseFoundationBiometry
from clingeo.losses.landmark_loss import LandmarkCriterion
from clingeo.views import FixedViewMean
from .exposure import ExposureDataset, exposure_indices, to_device
from .visible import VisibleExposureDataset
from .state import LearnedEMA, sampled_exposures, schedule_factor, save_checkpoint
from .growth import grow_lora_state
from .mre_loss import mre_objective
from .pixel_weighting import weight_in_original_pixels
from .spatial_loss import SpatialLandmarkCriterion


def validate_training_frame(frame, schema):
    required = {"domain", "image_path", "case_id", "manifest_row_id", "hidden_identity_group",
                "landmarks", "valid_mask", "width", "height"}
    if not required <= set(frame) or frame.empty:
        raise ValueError("training manifest is empty or missing documented columns")
    if frame.manifest_row_id.isna().any() or frame.manifest_row_id.astype(str).eq("").any() or frame.manifest_row_id.duplicated().any():
        raise ValueError("manifest_row_id must be present and unique")
    if frame.hidden_identity_group.isna().any() or frame.hidden_identity_group.astype(str).eq("").any():
        raise ValueError("resolved identity groups are required")
    if not set(frame.domain) <= set(schema.tasks):
        raise ValueError("unknown training task")
    return frame


def disjoint(train, other):
    if set(train.manifest_row_id) & set(other.manifest_row_id) or set(train.hidden_identity_group) & set(other.hidden_identity_group):
        raise ValueError("identity or sample overlap between training and excluded/evaluation rows")


@torch.no_grad()
def evaluate(model, dataset, schema, device, batch_size, view):
    wrapped = FixedViewMean(model, schema, view) if view != "native" else model
    wrapped.eval()
    predictions, labels = [], []
    for start in range(0, len(dataset), batch_size):
        indices = range(start, min(start + batch_size, len(dataset)))
        batch = to_device(collate_biometry([dataset[i] for i in indices]), device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = wrapped(batch["image"], batch["task_id"])["coords"].float().cpu().numpy()
        for j, i in enumerate(indices):
            row = dataset.df.iloc[i]
            identifier = str(row.manifest_row_id)
            labels.append({"sample_id": identifier, "task_id": row.domain,
                           "landmarks": row.landmarks, "valid_mask": row.valid_mask})
            predictions.append({"image_path": identifier, "task_id": row.domain,
                "predicted_points_pixels": original_coordinates(output[j], batch["affine_inv"][j].cpu().numpy(),
                    int(batch["width"][j]), int(batch["height"][j]), int(batch["num_landmarks"][j]))})
    return score_predictions(predictions, labels, schema)


def optimize(model, dataset, frame, phase, schema, device, seed, workers, on_evaluate=None):
    training = phase["training"]
    batch_size, accumulation = training["batch_size"], training["accumulation_steps"]
    steps = training["optimizer_steps"]
    if min(batch_size, accumulation, steps) <= 0:
        raise ValueError("positive batch size, accumulation and optimizer steps required")
    groups = []
    for kind, rate in [(True, training["lora_lr"]), (False, training["head_lr"])]:
        params = [p for n, p in model.named_parameters() if p.requires_grad and ((".lora_" in n) == kind)]
        groups.append({"params": params, "lr": rate, "base_lr": rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=training["weight_decay"])
    ema = LearnedEMA(model, training["ema_decay"])
    count = steps * batch_size * accumulation
    exposures = ([(index, pos) for pos, index in enumerate(exposure_indices(frame, count, seed, smoke=False))]
                 if phase.get("legacy_sampling") else sampled_exposures(frame, count, seed, 1.))
    kwargs = {"multiprocessing_context": "spawn"} if workers else {}
    loader = DataLoader(dataset, sampler=exposures, batch_size=batch_size, num_workers=workers,
                        collate_fn=collate_biometry, pin_memory=device.type == "cuda", **kwargs)
    iterator = iter(loader)
    loss_type = phase["loss_type"]
    if loss_type not in {"mre", "landmark", "spatial"}:
        raise ValueError("unknown loss_type")
    criterion = (SpatialLandmarkCriterion if loss_type == "spatial" else LandmarkCriterion)(schema, weights=phase["loss"]) if loss_type != "mre" else None
    history = []
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        factor = 1. if phase.get("constant_lr") else schedule_factor(step, training["schedule_steps"], training["warmup_steps"], training["lr_floor"])
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        total = 0.
        for _ in range(accumulation):
            batch = to_device(next(iterator), device)
            if phase.get("equal_image_loss"):
                batch["sample_weight"] = batch["sample_weight"] / batch["valid_mask"].sum(-1).clamp_min(1)
            if phase.get("original_pixel_weighting"):
                weight_in_original_pixels(batch)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(batch["image"], batch["task_id"])
            loss, _ = (mre_objective(outputs, batch, phase["loss"]) if loss_type == "mre"
                       else criterion({k: v.float() for k, v in outputs.items()}, batch))
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            (loss / accumulation).backward()
            total += float(loss.detach()) / accumulation
        if any(p.grad is not None for n, p in model.named_parameters() if n.startswith("backbone.encoder.") and ".lora_" not in n):
            raise AssertionError("frozen encoder received a gradient")
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()
        ema.update(model)
        history.append({"step": step + 1, "loss": total})
        if on_evaluate and ((step + 1) % training["evaluate_every"] == 0 or step + 1 == steps):
            with ema.applied(model):
                on_evaluate(step + 1, model)
    with ema.applied(model) if phase.get("selected_state", "ema") == "ema" else nullcontext():
        state = model.research_state_dict()
    return state, history


def train(config_path, manifest, image_root, encoder, schema_path, output, device="cuda:0",
          initial=None, validation=None, exclude=None, workers=4):
    config = read_json(config_path)
    schema = load_landmark_schema(schema_path)
    frame = validate_training_frame(pd.read_csv(manifest), schema)
    is_development = config["data_role"] == "development"
    if is_development != (validation is not None):
        raise ValueError("development requires --validation; final training must not use it")
    val_frame = validate_training_frame(pd.read_csv(validation), schema) if validation else None
    if val_frame is not None:
        disjoint(frame, val_frame)
    if exclude:
        excluded = pd.read_csv(exclude)
        disjoint(frame, excluded)
    if config["initialization"] != "fresh" and initial is None:
        raise ValueError("this stage requires --initial")
    if config["initialization"] == "fresh" and initial is not None:
        raise ValueError("fresh adaptation must not receive --initial")
    device = torch.device(device)
    torch.set_num_threads(4)
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    model_config = {"candidate": "dino_lora", "input_size": config["input_size"],
                    "lora_rank": config["lora_rank"], "weight_dir": str(encoder)}
    common = dict(data_root=image_root, schema=schema, input_size=config["input_size"], cache_dir=None,
                  ultrasound_aug="strong", geometry_overrides=config["geometry_overrides"], preprocess={"ch2": "global_he"})
    data = BiometryDataset(frame, training=True, **common)
    dataset = (VisibleExposureDataset if config["visible_augmentation"] else ExposureDataset)(data, config["seed"])
    val_data = BiometryDataset(val_frame, training=False, **common) if validation else None
    state = torch.load(initial, map_location="cpu", weights_only=True)["state_dict"] if initial else None
    best, best_step = float("inf"), None
    clean_model_config = {**model_config, "weight_dir": "ENCODER_DIR"}
    def save(state, step):
        save_checkpoint(out / "selected_checkpoint.pth", {"model_config": clean_model_config,
            "state_dict": state, "step": step, "selected_state": "ema" if is_development else config.get("selected_state", "ema")})
    def select(step, model):
        nonlocal best, best_step
        result = evaluate(model, val_data, schema, device, 8, config["selection_view"])
        score = result["macro_mre_pixels"]
        if score < best:
            best, best_step = score, step
            save(model.research_state_dict(), step)
            write_json(out / "development_score.json", result)
    history = []
    for phase_index, overrides in enumerate(config.get("phases", [{}])):
        phase = {**config, **overrides}
        cls = PreciseFoundationBiometry if phase.get("precision_head", True) else FoundationBiometry
        model = cls(schema, model_config).to(device)
        if state is not None:
            if config["initialization"] == "grow" and phase_index == 0:
                state = grow_lora_state(state, model.research_state_dict())
            model.load_research_state_dict(state)
        if is_development and initial:
            select(0, model)
        state, rows = optimize(model, dataset, frame, phase, schema, device, config["seed"], workers,
                               on_evaluate=select if is_development else None)
        history.extend({"phase": phase_index, **row} for row in rows)
        del model
    if not is_development:
        best_step = sum(row["training"]["optimizer_steps"] for row in config.get("phases", [config]))
        save(state, best_step)
    write_json(out / "history.json", history)
    write_json(out / "recipe.json", config)
    write_json(out / "result.json", {"status": "complete", "training_images": len(frame),
        "selected_step": best_step, "data_role": config["data_role"],
        "best_development_mre_pixels": best if is_development else None})
