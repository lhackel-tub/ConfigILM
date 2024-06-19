import torchvision.transforms as transforms

from configilm.extra.CustomTorchClasses import MyGaussianNoise
from configilm.extra.CustomTorchClasses import MyRotateTransform


def default_train_transform(img_size, mean, std):
    return transforms.Compose(
        [
            transforms.Resize(img_size, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean, std),
        ]
    )


def default_train_transform_with_noise(img_size, mean, std, gaussian_sigma: float = 20.0):
    """
    Returns a default training transform with gaussian noise added.
    Also contains additional rotation.
    """
    return transforms.Compose(
        [
            transforms.Resize(img_size, antialias=True),
            MyGaussianNoise(sigma=gaussian_sigma),
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
