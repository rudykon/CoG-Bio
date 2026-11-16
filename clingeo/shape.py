"""Training-fitted cardiac shape consistency, with fixed literature-inspired settings.

Independent implementation of a conditional Gaussian landmark prior. The
motivation is the public Kev-Bio postprocessing description; no external code
or pretrained model is imported. Length limits use image-diagonal units so a
change of image resolution does not change the constraint.
"""
from __future__ import annotations
import json
import numpy as np

CARDIAC = ("A4C", "IVC", "PLAX", "PSAX")


def fit_prior(frame):
    fitted = {}
    for task in CARDIAC:
        shapes, lengths = [], []
        for row in frame[frame.domain == task].itertuples():
            points = np.asarray(json.loads(row.landmarks), dtype=np.float64)
            valid = np.asarray(json.loads(row.valid_mask), dtype=bool)
            if not valid.all() or not np.isfinite(points).all():
                continue
            size = np.array([row.width, row.height], dtype=np.float64)
            shapes.append((points / size).reshape(-1))
            pairs = points.reshape(-1, 2, 2)
            lengths.append(np.linalg.norm(pairs[:, 0] - pairs[:, 1], axis=-1) / np.linalg.norm(size))
        samples = np.asarray(shapes)
        if len(samples) < 3:
            raise ValueError(f"at least three complete training shapes required for {task}")
        covariance = np.cov(samples, rowvar=False)
        covariance = .8 * covariance + .2 * np.diag(np.diag(covariance))
        ridge = max(1e-8, float(np.trace(covariance) / len(covariance)) * 1e-4)
        covariance += np.eye(len(covariance)) * ridge
        fitted[task] = {
            "training_cases": len(samples), "mean": samples.mean(0).tolist(),
            "covariance": covariance.tolist(), "ridge": ridge,
            "length_bands_diagonal_units": np.percentile(lengths, [5., 95.], axis=0).T.tolist(),
        }
    return {"format_version": 22, "tasks": fitted, "shrinkage_diagonal": .2,
            "mahalanobis_threshold": 4., "maximum_updates": 3,
            "length_percentiles": [5., 95.], "reference": "https://arxiv.org/abs/2611.03360"}


def enforce_shape(points, size, prior, threshold=4., maximum_updates=3):
    mean = np.asarray(prior["mean"], dtype=np.float64)
    covariance = np.asarray(prior["covariance"], dtype=np.float64)
    values = (points / size).reshape(-1).copy()
    conditional = []
    for endpoint in range(len(points)):
        selected = np.array([2 * endpoint, 2 * endpoint + 1])
        remaining = np.array([j for j in range(len(values)) if j // 2 != endpoint])
        coefficient = np.linalg.solve(covariance[np.ix_(remaining, remaining)],
                                      covariance[np.ix_(remaining, selected)]).T
        residual_cov = covariance[np.ix_(selected, selected)] - coefficient @ covariance[np.ix_(remaining, selected)]
        conditional.append((selected, remaining, coefficient, np.linalg.inv(residual_cov)))
    for _ in range(maximum_updates):
        candidates = []
        for selected, remaining, coefficient, precision in conditional:
            estimate = mean[selected] + coefficient @ (values[remaining] - mean[remaining])
            residual = values[selected] - estimate
            distance = float(np.sqrt(max(0., residual @ precision @ residual)))
            candidates.append((distance, selected, estimate))
        distance, selected, estimate = max(candidates, key=lambda item: item[0])
        if distance <= threshold:
            break
        values[selected] = estimate
    return values.reshape(-1, 2) * size


def enforce_lengths(points, size, prior):
    result = points.copy()
    for pair, (low, high) in enumerate(prior["length_bands_diagonal_units"]):
        indices = [2 * pair, 2 * pair + 1]
        displacement = result[indices[1]] - result[indices[0]]
        distance = float(np.linalg.norm(displacement))
        # A collapsed prediction supplies no anatomical direction: leave it to
        # the image model instead of inventing an arbitrary vertical caliper.
        if distance < 1e-6:
            continue
        desired = float(np.clip(distance / np.linalg.norm(size), low, high)) * np.linalg.norm(size)
        if abs(desired - distance) <= 1e-9:
            continue
        center = result[indices].mean(0)
        half = displacement * (desired / (2 * distance))
        result[indices[0]], result[indices[1]] = center - half, center + half
    return result


def apply_prior(records, metadata, prior, variant, id_key="manifest_row_id"):
    if variant not in {"unchanged", "shape", "length", "shape_length"}:
        raise ValueError("undeclared prior variant")
    output, changed = [], 0
    for row in records:
        points = np.asarray(row["predicted_points_pixels"], dtype=np.float64).reshape(-1, 2)
        before = points.copy()
        task = row["task_id"]
        if variant != "unchanged" and task in prior["tasks"]:
            info = metadata[row[id_key]]
            size = np.asarray([info["width"], info["height"]], dtype=np.float64)
            if "shape" in variant:
                points = enforce_shape(points, size, prior["tasks"][task], prior["mahalanobis_threshold"], prior["maximum_updates"])
            if "length" in variant:
                points = enforce_lengths(points, size, prior["tasks"][task])
            points = np.clip(points, [0., 0.], size - 1)
        if not np.isfinite(points).all():
            raise ValueError("nonfinite structured prediction")
        points = np.round(points, 4)
        changed += int(not np.array_equal(points, before))
        output.append({**row, "predicted_points_pixels": points.reshape(-1).tolist()})
    return output, changed
