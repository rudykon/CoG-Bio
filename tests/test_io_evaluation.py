import copy
import json
import numpy as np
import pandas as pd
import pytest
from clingeo.io import image_rows, read_json, read_rows, write_json, validate_predictions
from clingeo.images import original_coordinates
from clingeo.evaluation import score_predictions


def records(labels, offset=0.):
    return [dict(image_path=r["sample_id"], task_id=r["task_id"],
        predicted_points_pixels=(np.asarray(json.loads(r["landmarks"])) + [offset, 0]).reshape(-1).tolist()) for r in labels]


def test_json_manifest_and_perfect_score(nine_images, schema):
    root, rows, labels, _ = nine_images
    loaded = image_rows(root / "images.csv", schema, root)
    assert len(loaded) == 9 and loaded[0]["checksum"] is None
    out = root / "nested" / "predictions.json"
    write_json(out, records(labels))
    assert not out.with_suffix(".json.tmp").exists()
    result = score_predictions(read_json(out), read_rows(root / "labels.csv"), schema)
    assert result["macro_mre_pixels"] == 0. and result["tasks"] == 9
    with pytest.raises(ValueError): write_json(out, [float("nan")])


@pytest.mark.parametrize("change", ["empty", "duplicate", "unknown_task", "wrong_count", "bad_size", "outside", "absolute"])
def test_bad_image_manifest(nine_images, schema, change):
    root, rows, _, _ = nine_images
    if change == "empty": rows = []
    elif change == "duplicate": rows[1]["sample_id"] = rows[0]["sample_id"]
    elif change == "unknown_task": rows[0]["task_id"] = "UNKNOWN"
    elif change == "wrong_count": rows[0]["num_landmarks"] = 1
    elif change == "bad_size": rows[0]["width"] = 0
    elif change == "outside": rows[0]["image_relpath"] = "../outside.png"
    elif change == "absolute": rows[0]["image_relpath"] = "/outside.png"
    path = root / "invalid.csv"
    pd.DataFrame(rows, columns=list(nine_images[1][0])).to_csv(path, index=False)
    with pytest.raises(ValueError): image_rows(path, schema, root)


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown_id", "wrong_task", "wrong_count", "nan"])
def test_malformed_predictions(nine_images, schema, change):
    _, rows, labels, _ = nine_images
    values = records(labels)
    if change == "missing": values.pop()
    elif change == "duplicate": values[1] = copy.deepcopy(values[0])
    elif change == "unknown_id": values[0]["image_path"] = "unknown"
    elif change == "wrong_task": values[0]["task_id"] = "HC"
    elif change == "wrong_count": values[0]["predicted_points_pixels"].pop()
    elif change == "nan": values[0]["predicted_points_pixels"][0] = float("nan")
    with pytest.raises(ValueError): validate_predictions(values, rows, schema)


def test_task_macro_does_not_weight_large_tasks_more(nine_images, schema):
    _, _, labels, _ = nine_images
    first, second = labels[:2]
    labels = [{**first, "sample_id": f"first-{i}"} for i in range(4)] + [second]
    predicted = records(labels)
    predicted[-1] = records([second], 10)[0]
    result = score_predictions(predicted, labels, schema, require_all_tasks=False)
    assert result["macro_mre_pixels"] == 5
    with pytest.raises(ValueError): score_predictions(predicted, labels, schema)
    with pytest.raises(ValueError): score_predictions([], [], schema, require_all_tasks=False)


@pytest.mark.parametrize("change", ["coordinates", "mask_shape", "empty_mask", "nonfinite"])
def test_invalid_labels(nine_images, schema, change):
    _, _, labels, _ = nine_images
    predicted = records(labels)
    if change == "coordinates": labels[0]["landmarks"] = "[[0,0]]"
    elif change == "mask_shape": labels[0]["valid_mask"] = "[1]"
    elif change == "empty_mask": labels[0]["valid_mask"] = json.dumps([0] * labels[0]["num_landmarks"])
    else: labels[0]["landmarks"] = json.dumps([[float("nan"), 0]] * labels[0]["num_landmarks"])
    with pytest.raises(ValueError): score_predictions(predicted, labels, schema)


def test_coordinate_serialization_clips_and_rounds():
    assert original_coordinates([[-1, 2], [100, 3.123456]], np.eye(3), 80, 64, 2) == [0., 2., 79., 3.1235]
    with pytest.raises(ValueError): original_coordinates([[0, float("nan")]], np.eye(3), 80, 64, 1)
