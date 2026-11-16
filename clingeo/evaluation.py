"""Original-pixel MRE: valid landmarks, then images, then equally weighted tasks."""
import json
import numpy as np
from .io import validate_predictions

PAPER_TASKS = {"A4C", "AOP", "FA", "FUGC", "HC", "IVC", "PLAX", "PSAX", "fetal_femur"}


def score_predictions(records, labels, schema, require_all_tasks=True):
    validate_predictions(records, labels, schema)
    truth = {r["sample_id"]: r for r in labels}
    values = {}
    for record in records:
        row = truth[record["image_path"]]
        predicted = np.asarray(record["predicted_points_pixels"], dtype=np.float64).reshape(-1, 2)
        target = np.asarray(json.loads(row["landmarks"]), dtype=np.float64)
        mask = np.asarray(json.loads(row["valid_mask"]), dtype=bool)
        if target.shape != predicted.shape or mask.shape != (len(predicted),):
            raise ValueError("label coordinates/mask differ from the task schema")
        if not mask.any() or not np.isfinite(target[mask]).all():
            raise ValueError("each evaluated image needs finite valid ground truth")
        error = np.linalg.norm(predicted[mask] - target[mask], axis=-1).mean()
        values.setdefault(record["task_id"], []).append(float(error))
    if not values or (require_all_tasks and set(values) != PAPER_TASKS):
        raise ValueError("all nine tasks are required for the paper's task-macro score")
    per_task = {task: {"images": len(v), "mre_pixels": float(np.mean(v))} for task, v in sorted(values.items())}
    return {"macro_mre_pixels": float(np.mean([v["mre_pixels"] for v in per_task.values()])),
            "images": len(records), "tasks": len(values), "per_task": per_task,
            "metric": "equal-task macro mean radial error in original image pixels"}
