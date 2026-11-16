import numpy as np
from clingeo.shape import enforce_lengths, enforce_shape


def test_conditional_outlier_and_unchanged_inlier():
    mean = np.array([.2, .3, .4, .3])
    prior = {"mean": mean.tolist(), "covariance": (np.eye(4) * .001).tolist()}
    size = np.array([100., 200.])
    points = mean.reshape(-1, 2) * size
    np.testing.assert_allclose(enforce_shape(points, size, prior), points)
    corrupted = points.copy(); corrupted[1, 0] = 95.
    np.testing.assert_allclose(enforce_shape(corrupted, size, prior), points)


def test_length_constraint_preserves_center_direction_and_scale():
    size = np.array([300., 400.])
    points = np.array([[10., 20.], [40., 60.]])
    prior = {"length_bands_diagonal_units": [[.04, .08]]}
    corrected = enforce_lengths(points, size, prior)
    np.testing.assert_allclose(corrected.mean(0), points.mean(0))
    np.testing.assert_allclose(corrected[1] - corrected[0], [.8 * 30, .8 * 40])
    np.testing.assert_allclose(enforce_lengths(points * 2, size * 2, prior), corrected * 2)
    collapsed = np.ones((2, 2))
    np.testing.assert_array_equal(enforce_lengths(collapsed, size, prior), collapsed)
