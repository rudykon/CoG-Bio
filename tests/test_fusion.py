import numpy as np
from clingeo.fusion import fuse


def records(x):
    return [{"manifest_row_id":"a","task_id":"HC","predicted_points_pixels":[x,0.]}]


def test_consistency_prefers_stable_model_and_caps_weight():
    views=[records(0.),records(0.),records(4.),records(12.)]
    # Stable old model receives at most 0.75; the other model's mean is 8.
    for kind in ["case_consistency","point_consistency"]:
        out=fuse(views,kind,1.)
        assert np.allclose(out[0]["predicted_points_pixels"],[2.,0.])


def test_fusion_preserves_identical_predictions_and_is_model_symmetric():
    for kind in ["equal_mean","case_consistency","point_consistency","geometric_median"]:
        assert fuse([records(3.)]*4,kind,1.)[0]["predicted_points_pixels"]==[3.,0.]
        a=[records(0.),records(2.),records(6.),records(10.)]
        assert fuse(a,kind,1.)[0]["predicted_points_pixels"]==fuse(a[2:]+a[:2],kind,1.)[0]["predicted_points_pixels"]
