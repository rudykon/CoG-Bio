from __future__ import annotations

import numpy as np
import pytest
import torch

from clingeo.postprocess.coordinate_transform import (
    apply_affine,
    apply_affine_torch,
    make_letterbox_affine,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_apply_affine_torch_shared_and_batched_matrices_match_numpy(dtype: torch.dtype):
    xy = torch.tensor(
        [
            [[0.0, 1.0], [4.0, -2.0]],
            [[3.0, 7.0], [-5.0, 6.0]],
        ],
        dtype=dtype,
    )
    matrix = torch.tensor(
        [[1.5, 0.25, 3.0], [-0.5, 2.0, -4.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64 if dtype == torch.float32 else torch.float32,
    )

    shared = apply_affine_torch(xy, matrix)
    batched = apply_affine_torch(xy, matrix.unsqueeze(0).repeat(2, 1, 1))
    expected = apply_affine(xy.numpy(), matrix.numpy())

    assert shared.dtype == dtype
    assert shared.device == xy.device
    assert torch.allclose(shared, batched)
    assert np.allclose(shared.detach().numpy(), expected, rtol=1e-5, atol=1e-6)


def test_apply_affine_torch_uses_each_batch_matrix_without_cross_broadcasting():
    xy = torch.zeros((2, 2, 2), dtype=torch.float32)
    matrices = torch.eye(3).repeat(2, 1, 1)
    matrices[0, :2, 2] = torch.tensor([5.0, -7.0])
    matrices[1, :2, 2] = torch.tensor([-5.0, 7.0])

    mapped = apply_affine_torch(xy, matrices)

    assert torch.equal(mapped[0], torch.tensor([[5.0, -7.0], [5.0, -7.0]]))
    assert torch.equal(mapped[1], torch.tensor([[-5.0, 7.0], [-5.0, 7.0]]))


def test_apply_affine_torch_batched_round_trip_preserves_gradients_and_inputs():
    xy = torch.tensor(
        [[[1.0, 2.0], [8.0, -3.0]], [[-4.0, 5.0], [9.0, 6.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    xy_before = xy.detach().clone()
    matrices = torch.tensor(
        [
            [[2.0, 0.0, 3.0], [0.0, 0.5, -1.0], [0.0, 0.0, 1.0]],
            [[0.5, 0.0, -2.0], [0.0, 4.0, 6.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    matrices_before = matrices.clone()

    mapped = apply_affine_torch(xy, matrices)
    restored = apply_affine_torch(mapped, torch.linalg.inv(matrices))
    restored.sum().backward()

    assert torch.allclose(restored, xy)
    assert torch.equal(xy.detach(), xy_before)
    assert torch.equal(matrices, matrices_before)
    assert xy.grad is not None
    assert torch.allclose(xy.grad, torch.ones_like(xy))


def test_non_square_letterbox_restores_five_input_pixels_to_ten_original_pixels():
    transform = make_letterbox_affine(height=200, width=400, output_size=200)
    original = torch.tensor([[100.0, 80.0]], dtype=torch.float64)
    input_xy = apply_affine_torch(original, torch.from_numpy(transform.matrix))
    shifted_input = input_xy + torch.tensor([[5.0, 0.0]], dtype=torch.float64)
    restored = apply_affine_torch(shifted_input, torch.from_numpy(transform.inverse))

    assert transform.matrix[0, 0] == pytest.approx(0.5)
    assert restored[0, 0] - original[0, 0] == pytest.approx(10.0)
    assert restored[0, 1] == pytest.approx(original[0, 1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_apply_affine_torch_preserves_cuda_device_and_dtype():
    xy = torch.tensor([[1.0, 2.0]], dtype=torch.float32, device="cuda")
    matrix = torch.eye(3, dtype=torch.float64)

    result = apply_affine_torch(xy, matrix)

    assert result.dtype == xy.dtype
    assert result.device == xy.device


@pytest.mark.parametrize(
    ("xy", "matrix", "error_type", "message"),
    [
        (np.zeros((2, 2)), torch.eye(3), TypeError, "xy"),
        (torch.zeros((2, 2)), np.eye(3), TypeError, "matrix"),
        (torch.zeros((2, 2), dtype=torch.int64), torch.eye(3), TypeError, "floating"),
        (torch.zeros((2, 2)), torch.eye(3, dtype=torch.int64), TypeError, "floating"),
        (torch.zeros((2,)), torch.eye(3), ValueError, "xy.*shape"),
        (torch.zeros((2, 3)), torch.eye(3), ValueError, "xy.*shape"),
        (torch.zeros((1, 1, 1, 2)), torch.eye(3), ValueError, "xy.*shape"),
        (torch.zeros((2, 2)), torch.zeros((2, 3)), ValueError, "matrix.*shape"),
        (torch.zeros((2, 2)), torch.eye(4), ValueError, "matrix.*shape"),
        (torch.zeros((2, 2)), torch.eye(3).repeat(1, 1, 1), ValueError, "batch|shape"),
        (torch.zeros((2, 2, 2)), torch.eye(3).repeat(1, 1, 1), ValueError, "batch"),
        (torch.zeros((2, 2, 2)), torch.eye(3).repeat(3, 1, 1), ValueError, "batch"),
    ],
)
def test_apply_affine_torch_rejects_invalid_types_and_shapes(xy, matrix, error_type, message):
    with pytest.raises(error_type, match=message):
        apply_affine_torch(xy, matrix)


@pytest.mark.parametrize(
    ("xy", "matrix", "message"),
    [
        (torch.tensor([[float("nan"), 0.0]]), torch.eye(3), "xy.*finite"),
        (torch.tensor([[float("inf"), 0.0]]), torch.eye(3), "xy.*finite"),
        (torch.zeros((1, 2)), torch.tensor([[1.0, 0.0, float("nan")], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "matrix.*finite"),
        (torch.zeros((1, 2)), torch.diag(torch.tensor([1.0, 1.0, 2.0])), "affine"),
        (torch.zeros((1, 2)), torch.tensor([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]]), "singular"),
    ],
)
def test_apply_affine_torch_rejects_nonfinite_nonaffine_and_singular_inputs(xy, matrix, message):
    with pytest.raises(ValueError, match=message):
        apply_affine_torch(xy, matrix)
