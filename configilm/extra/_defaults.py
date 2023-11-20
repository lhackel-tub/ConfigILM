import torchvision.transforms as transforms

from configilm.extra.CustomTorchClasses import MyGaussianNoise
from configilm.extra.CustomTorchClasses import MyRotateTransform


def default_train_transform(img_size, mean, std):
    return transforms.Compose(
        [
            transforms.Resize(img_size, antialias=True),
            MyGaussianNoise(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            MyRotateTransform([0, 90, 180, 270]),
            transforms.Normalize(mean, std),
        ]
    )


def default_transform(img_size, mean, std):
    return transforms.Compose(
        [
            transforms.Resize(img_size, antialias=True),
            transforms.Normalize(mean, std),
        ]
    )
