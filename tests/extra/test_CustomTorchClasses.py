import pytest
import torch

from configilm.extra.CustomTorchClasses import MyGaussianNoise
from configilm.extra.CustomTorchClasses import MyRotateTransform


def test_rotate_single():
    rt = MyRotateTransform([90])
    t = torch.Tensor([[1, 0], [0, 2]]).unsqueeze(dim=0)
    t = rt(t).squeeze()
    assert torch.equal(t, torch.Tensor([[0, 2], [1, 0]]))


def test_rotate_multiple():
    rt = MyRotateTransform([0, 90, 180, 270])
    t = torch.Tensor([[1, 0], [0, 2]]).unsqueeze(dim=0)
    t = rt(t).squeeze()

    valid_results = [
        torch.Tensor([[1, 0], [0, 2]]),
        torch.Tensor([[0, 1], [2, 0]]),
        torch.Tensor([[2, 0], [0, 1]]),
        torch.Tensor([[0, 2], [1, 0]]),
    ]

    res_arr = []
    for v in valid_results:
        res_arr.append(torch.equal(t, v))
    assert sum(res_arr), (
        f"Only exactly 1 rotation should be equal " f"but was {sum(res_arr)}"
    )


def test_empty_sequence():
    with pytest.raises(AssertionError):
        _ = MyRotateTransform([])


def test_gaussian():
    gt = MyGaussianNoise(sigma=1)
    t = torch.Tensor([0])
    assert not torch.equal(gt(t), t), "No noise added"
    assert not torch.equal(gt(t), gt(t)), "Same noise used twice in a row"


def test_gaussian_zero():
    gt = MyGaussianNoise(sigma=0)
    t = torch.Tensor([0])
    assert torch.equal(gt(t), t), "No noise should be added"


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float,
        torch.double,
        torch.half,
        torch.bfloat16,
        torch.complex32,
        torch.complex128,
        torch.uint8,
        torch.int32,
        torch.bool,
    ],
)
def test_gaussian_dtype(dtype):
    gt = MyGaussianNoise(sigma=0)
    t = torch.ones([10, 10], dtype=dtype)
    assert gt(t).dtype == t.dtype, "Datatype changed"
