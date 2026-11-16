import copy
import json
import numpy as np
import pandas as pd
import pytest
from clingeo.fusion import fuse, infer_floor
from clingeo.shape import fit_prior, apply_prior, CARDIAC
from clingeo.projection import project_ellipse
from clingeo.postprocess.coordinate_transform import apply_affine, make_letterbox_affine, clamp_coords


def test_fit_prior_and_repair_use_training_geometry(nine_images):
    _, rows, labels, frame = nine_images
    frames = []
    for shift in (-2, 0, 2, 4):
        local = frame.copy()
        local["landmarks"] = [json.dumps((np.asarray(json.loads(s)) + shift).tolist()) for s in local.landmarks]
        frames.append(local)
    training = pd.concat(frames)
    prior = fit_prior(training)
    assert set(prior["tasks"]) == set(CARDIAC)
    for task in CARDIAC:
        assert prior["tasks"][task]["training_cases"] == 4
        assert np.linalg.eigvalsh(prior["tasks"][task]["covariance"]).min() > 0
    malformed = training.copy()
    malformed.loc[malformed.index == 0, "valid_mask"] = json.dumps([0] * len(json.loads(malformed.iloc[0].valid_mask)))
    with pytest.raises(ValueError): fit_prior(malformed)
    with pytest.raises(ValueError): fit_prior(frame)
    records = [dict(image_path=r["sample_id"], task_id=r["task_id"],
        predicted_points_pixels=np.asarray(json.loads(r["landmarks"])).reshape(-1).tolist()) for r in labels]
    metadata = {r["sample_id"]: r for r in rows}
    for variant in ("unchanged", "shape", "length", "shape_length"):
        result, _ = apply_prior(records, metadata, prior, variant, "image_path")
        assert all(np.isfinite(r["predicted_points_pixels"]).all() for r in result)
    with pytest.raises(ValueError): apply_prior(records, metadata, prior, "invalid", "image_path")
    records[-1]["predicted_points_pixels"][0] = float("nan")
    with pytest.raises(ValueError): apply_prior(records, metadata, prior, "unchanged", "image_path")


def test_fixed_fusion_scale_and_error_guards(nine_images):
    _, _, labels, _ = nine_images
    base = [dict(image_path=r["sample_id"], task_id=r["task_id"], predicted_points_pixels=[0., 0.]) for r in labels]
    shifted = [{**r, "predicted_points_pixels": [4., 0.]} for r in base]
    assert infer_floor([base, shifted, base, shifted], "image_path") == 4.
    assert fuse([base, shifted, shifted, shifted], "new_single", 1., "image_path") == shifted
    with pytest.raises(ValueError): infer_floor([base, shifted[:-1], base, shifted], "image_path")
    with pytest.raises(ValueError): infer_floor([base[:1]] * 4, "image_path")
    with pytest.raises(ValueError): fuse([base] * 4, "unknown", 1., "image_path")
    with pytest.raises(ValueError): fuse([base] * 3, "case_consistency", 1., "image_path")


def test_coordinate_guard_and_bounds():
    with pytest.raises(ValueError): make_letterbox_affine(0, 12, 32)
    assert make_letterbox_affine(10, 20, (32, 40)).output_size == (32, 40)
    with pytest.raises(ValueError): apply_affine(np.ones((2, 3)), np.eye(3))
    assert clamp_coords([[-1, 9]], 5, 5).tolist() == [[0., 4.]]
    with pytest.raises(ValueError): project_ellipse(np.zeros((3, 2)))
