import numpy as np
from clingeo.projection import project_ellipse


def test_projection_is_idempotent_and_similarity_equivariant():
    points = np.array([[20., 8.], [18., 42.], [2., 28.], [42., 31.]])
    result = project_ellipse(points)
    np.testing.assert_allclose(result[:2].mean(0), result[2:].mean(0))
    assert abs(np.dot(result[1] - result[0], result[3] - result[2])) < 1e-8
    np.testing.assert_allclose(project_ellipse(result), result)
    angle = .7; rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    np.testing.assert_allclose(project_ellipse(points @ rotation.T * 2 + 5), result @ rotation.T * 2 + 5)


def test_axis_aligned_projection_conventions():
    points = np.array([[21., 10.], [19., 40.], [2., 27.], [38., 23.]])
    result = project_ellipse(points, True)
    np.testing.assert_allclose(result, [[20., 10.], [20., 40.], [2., 25.], [38., 25.]])
