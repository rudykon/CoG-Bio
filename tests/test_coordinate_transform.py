import numpy as np

from clingeo.postprocess.coordinate_transform import make_letterbox_affine


def test_coordinate_affine_round_trip():
    affine = make_letterbox_affine(height=333, width=777, output_size=768)
    xy = np.array([[0.0, 0.0], [100.5, 200.25], [776.0, 332.0]], dtype=np.float64)
    mapped = affine.apply(xy)
    restored = affine.invert(mapped)
    assert np.max(np.abs(restored - xy)) < 1e-4
