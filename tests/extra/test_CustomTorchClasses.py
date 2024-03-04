import warnings

import pytest
import torch

from configilm.extra.CustomTorchClasses import LinearWarmupCosineAnnealingLR
from configilm.extra.CustomTorchClasses import MyGaussianNoise
from configilm.extra.CustomTorchClasses import MyRotateTransform


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def control_lr(lr_is: float, lr_should: float, epoch):
    assert torch.isclose(
        torch.Tensor([lr_is]), torch.tensor([lr_should]), atol=1e-4
    ), f"LR is {lr_is} but should be {lr_should} after {epoch} epochs."


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
    assert sum(res_arr), f"Only exactly 1 rotation should be equal " f"but was {sum(res_arr)}"


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
    if dtype in [torch.complex32, torch.complex128]:
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message=r"ComplexHalf support is experimental and many " r"operators don't support it yet.+",
        )
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message=r"Casting complex values to real discards the " r"imaginary part.+",
        )
        t = torch.ones([10, 10], dtype=dtype)
    else:
        t = torch.ones([10, 10], dtype=dtype)
    assert gt(t).dtype == t.dtype, "Datatype changed"


@pytest.mark.filterwarnings('ignore:To get the last learning rate computed by the scheduler,')
def test_lwcalr_scheduler_basic():
    warmup_epochs = 10
    max_epochs = 1000
    warmup_start_lr = 0.0
    eta_min = 0.0
    optim = torch.optim.SGD(MockModel().parameters(), lr=1)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optim,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )

    for i in range(max_epochs):
        lr = scheduler.get_lr()[0]
        if i == 0:
            control_lr(lr, 0.0, i)
        if i == 10:
            control_lr(lr, 1.0, i)
        if i == 504:
            control_lr(lr, 0.5, i)
        if i == 999:
            control_lr(lr, 0.0, i)
        # just to supress warnings that optimizer should step before scheduler
        optim.step()
        scheduler.step()


@pytest.mark.filterwarnings('ignore:To get the last learning rate computed by the scheduler,')
def test_lwcalr_scheduler_non_zero_start():
    warmup_epochs = 10
    max_epochs = 1000
    warmup_start_lr = 0.5
    eta_min = 0.0
    optim = torch.optim.SGD(MockModel().parameters(), lr=1)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optim,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )

    for i in range(max_epochs):
        lr = scheduler.get_lr()[0]
        if i == 0:
            control_lr(lr, 0.5, i)
        if i == 10:
            control_lr(lr, 1.0, i)
        if i == 504:
            control_lr(lr, 0.5, i)
        if i == 999:
            control_lr(lr, 0.0, i)
        # just to supress warnings that optimizer should step before scheduler
        optim.step()
        scheduler.step()


@pytest.mark.filterwarnings('ignore:To get the last learning rate computed by the scheduler,')
def test_lwcalr_scheduler_non_zero_end():
    warmup_epochs = 10
    max_epochs = 1000
    warmup_start_lr = 0.0
    eta_min = 0.5
    optim = torch.optim.SGD(MockModel().parameters(), lr=1)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optim,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )

    for i in range(max_epochs):
        lr = scheduler.get_lr()[0]
        if i == 0:
            control_lr(lr, 0.0, i)
        if i == 10:
            control_lr(lr, 1.0, i)
        if i == 504:
            control_lr(lr, 0.75, i)
        if i == 999:
            control_lr(lr, 0.5, i)
        # just to supress warnings that optimizer should step before scheduler
        optim.step()
        scheduler.step()


@pytest.mark.filterwarnings('ignore:To get the last learning rate computed by the scheduler,')
@pytest.mark.filterwarnings('ignore:The epoch parameter in ')
def test_lwcalr_scheduler_closed_form():
    warmup_epochs = 10
    max_epochs = 1000
    warmup_start_lr = 0.0
    eta_min = 0.0
    optim = torch.optim.SGD(MockModel().parameters(), lr=1)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optim,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )

    for i in range(max_epochs):
        lr = scheduler.get_lr()[0]
        if i == 0:
            control_lr(lr, 0.0, i)
        if i == 10:
            control_lr(lr, 1.0, i)
        if i == 504:
            control_lr(lr, 0.5, i)
        if i == 999:
            control_lr(lr, 0.0, i)
        # just to supress warnings that optimizer should step before scheduler
        optim.step()
        scheduler.step()

    # just to supress warnings that optimizer should step before scheduler
    optim.step()
    scheduler.step(4)
    lr = scheduler.get_lr()[0]
    control_lr(lr, 0.5555, epoch="4 second time")

    # just to supress warnings that optimizer should step before scheduler
    optim.step()
    scheduler.step(504)
    lr = scheduler.get_lr()[0]
    control_lr(lr, 0.5, epoch="504 second time")
