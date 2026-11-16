"""Least-squares projection onto the observed HC/FA annotation geometry.

HC axes share a center and are orthogonal; FA axes additionally align with the
image axes. These conventions are audited on inner training labels before any
development result is computed. Endpoint identities are preserved.
"""
import numpy as np


def project_ellipse(points, axis_aligned=False):
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("four finite ellipse endpoints required")
    center = points.mean(0)
    first = (points[1] - points[0]) / 2
    second = (points[3] - points[2]) / 2
    if axis_aligned:
        first = np.array([0., first[1]])
        second = np.array([second[0], 0.])
    else:
        quarter_turn = np.array([[0., -1.], [1., 0.]])
        turned = quarter_turn.T @ second
        matrix = np.outer(first, first) + np.outer(turned, turned)
        _, directions = np.linalg.eigh(matrix)
        direction = directions[:, -1]
        perpendicular = quarter_turn @ direction
        first = direction * np.dot(direction, first)
        second = perpendicular * np.dot(perpendicular, second)
    return np.array([center - first, center + first, center - second, center + second])


def apply_projection(records):
    result = []
    for row in records:
        points = np.asarray(row["predicted_points_pixels"], dtype=np.float64).reshape(-1, 2)
        if row["task_id"] in {"HC", "FA"}:
            points = project_ellipse(points, row["task_id"] == "FA")
        result.append({**row, "predicted_points_pixels": np.round(points, 4).reshape(-1).tolist()})
    return result
