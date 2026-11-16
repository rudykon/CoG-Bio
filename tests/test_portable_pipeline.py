import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from clingeo import inference
from clingeo.io import read_json, read_rows, write_json, validate_predictions
from clingeo.pipeline import combine_views
from clingeo.training.runner import train, optimize, validate_training_frame, disjoint
from clingeo.data.dataset import BiometryDataset
from clingeo.training.exposure import ExposureDataset

ROOT = Path(__file__).resolve().parents[1]


class SmallPredictor(torch.nn.Module):
    """Small trainable fixture; no pretrained artifact or patient data needed."""
    def __init__(self, schema, config):
        super().__init__()
        self.input_size = config["input_size"]
        self.max_landmarks = schema.max_landmarks
        self.head = torch.nn.Linear(3, self.max_landmarks * 2)
        self.adapter = torch.nn.Module()
        self.adapter.lora_a = torch.nn.Linear(3, 3, bias=False)
        self.base = torch.nn.Linear(3, 3).requires_grad_(False)

    def forward(self, image, task_id):
        mean = image.mean((-2, -1))
        features = self.base(mean) + self.adapter.lora_a(mean)
        points = self.head(features).sigmoid().view(-1, self.max_landmarks, 2) * (self.input_size - 1)
        heatmap = points[..., 0, None, None].expand(-1, -1, self.input_size // 4, self.input_size // 4)
        return dict(coords=points, coord_heatmap=points, coord_query=points, heatmap=heatmap)

    def research_state_dict(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_research_state_dict(self, state):
        with torch.no_grad():
            for n, p in self.named_parameters():
                if p.requires_grad: p.copy_(state[n])


def test_label_free_full_inference_writes_valid_json(nine_images, schema, monkeypatch):
    root, rows, _, _ = nine_images
    monkeypatch.setattr(inference, "load_predictor", lambda *a, **kw: SmallPredictor(schema, {"input_size": 32}).eval())
    # A label-free manifest is the only manifest passed to inference.
    result = inference.predict(root / "images.csv", root, root, root, root / "output",
        ROOT / "configs/landmark_schema.yaml", ROOT / "configs/inference.json",
        ROOT / "configs/shape_prior.json", device="cpu", workers=0, batch_size=3)
    assert len(result) == 9
    validate_predictions(read_json(root / "output/predictions.json"), rows, schema)
    assert read_json(root / "output/inference.json")["labels_read"] is False
    saved = [read_json(root / "output" / f"{model}_{view}_mean.json")
             for model in ("mrp", "iap") for view in ("flip", "crop")]
    recomputed, _ = combine_views(saved, rows, schema, read_json(ROOT / "configs/inference.json"), read_json(ROOT / "configs/shape_prior.json"))
    assert result == recomputed
    with pytest.raises(ValueError): combine_views(saved[:2], rows, schema, {}, {})


def test_synthetic_training_checkpoint_and_frozen_base(nine_images, schema, monkeypatch):
    import clingeo.training.runner as runner
    root, _, _, frame = nine_images
    config = read_json(ROOT / "configs/train_iap_final.json")
    config["input_size"] = 32
    config["training"].update(optimizer_steps=2, schedule_steps=2, warmup_steps=0, batch_size=3, accumulation_steps=1)
    write_json(root / "train_recipe.json", config)
    captured = []
    def build(s, c):
        m = SmallPredictor(s, c); captured.append((m, m.base.weight.detach().clone(), m.head.weight.detach().clone()))
        return m
    monkeypatch.setattr(runner, "PreciseFoundationBiometry", build)
    train(root / "train_recipe.json", root / "train.csv", root, root,
          ROOT / "configs/landmark_schema.yaml", root / "run", device="cpu", workers=0)
    model, base, head = captured[0]
    assert torch.equal(model.base.weight, base) and model.base.weight.grad is None
    assert not torch.equal(model.head.weight, head)
    saved = torch.load(root / "run/selected_checkpoint.pth", weights_only=True)
    assert saved["step"] == 2 and saved["model_config"]["weight_dir"] == "ENCODER_DIR"
    assert len(read_json(root / "run/history.json")) == 2
    assert all(np.isfinite(r["loss"]) for r in read_json(root / "run/history.json"))


def test_identity_overlap_is_rejected(nine_images, schema):
    root, _, _, frame = nine_images
    validate_training_frame(frame, schema)
    with pytest.raises(ValueError): disjoint(frame, frame.copy())
    with pytest.raises(ValueError): validate_training_frame(frame.drop(columns="hidden_identity_group"), schema)
    other = frame.copy(); other["manifest_row_id"] = "duplicate"
    with pytest.raises(ValueError): validate_training_frame(other, schema)
    other = frame.copy(); other["hidden_identity_group"] = None
    with pytest.raises(ValueError): validate_training_frame(other, schema)
    other = frame.copy(); other.loc[0, "domain"] = "unknown"
    with pytest.raises(ValueError): validate_training_frame(other, schema)
