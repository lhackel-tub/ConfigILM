import random
from typing import Sequence

import torch
import torchvision.transforms.functional as TF


class MyRotateTransform:
    # Rotates an image by one of the given degrees,
    # instead of a degree in the given range as transforms.RandomRotate does
    # Rotation is counter-clockwise.
    # Code credit to https://github.com/pytorch/vision/issues/566
    def __init__(self, angles: Sequence[int]):
        assert len(angles) > 0, (
            "Choice of no angles not possible. "
            "Please provide a sequence of at least length 1"
        )
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class MyGaussianNoise:
    # Adds gaussian noise with sigma in the specified range
    # Code credit to https://github.com/pytorch/vision/issues/6192
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, x):
        dtype = x.dtype
        if not x.is_floating_point():
            x = x.to(torch.float32)

        out = x + self.sigma * torch.randn_like(x)

        if out.dtype != dtype:
            out = out.to(dtype)

        return out
