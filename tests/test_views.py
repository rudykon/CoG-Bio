import torch
from clingeo.views import predicted_crop_box, undo_crop, undo_flip, average_records


def test_crop_inverse_preserves_corners_and_stays_inside_image():
    for points in [torch.tensor([[0.,0.],[20.,30.]]),torch.tensor([[480.,480.],[511.,511.]])]:
        left,top,side=predicted_crop_box(points,512)
        assert 0<=left<=512-side and 0<=top<=512-side and 410<=side<=512
        assert torch.allclose(undo_crop(torch.tensor([[0.,0.],[511.,511.]]),(left,top,side),512),
                              torch.tensor([[left,top],[left+side-1,top+side-1]],dtype=torch.float32))


def test_flip_inverse_is_involution_with_semantic_pairs():
    points=torch.tensor([[2.,3.],[7.,9.],[10.,4.],[13.,8.]])
    assert torch.equal(undo_flip(undo_flip(points,512,[(2,3)]),512,[(2,3)]),points)


def test_view_mean_matches_ids_and_keeps_task_coordinates():
    first=[{"manifest_row_id":"a","task_id":"HC","predicted_points_pixels":[2.,4.]},
           {"manifest_row_id":"b","task_id":"AOP","predicted_points_pixels":[8.,10.]}]
    second=[{**first[1],"predicted_points_pixels":[10.,12.]},
            {**first[0],"predicted_points_pixels":[4.,6.]}]
    result=average_records([first,second])
    assert [r["predicted_points_pixels"] for r in result]==[[3.,5.],[9.,11.]]
