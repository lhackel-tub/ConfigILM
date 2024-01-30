import warnings
from typing import Sequence, Union

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset


def dataset_ok(
        dataset: Union[Dataset, None],
        expected_image_shape: Sequence,
        expected_question_length: int,
        expected_length: Union[int, None],
        classes: int,
):
    assert dataset is not None
    if expected_length is not None:
        assert len(dataset) == expected_length

    if len(dataset) > 0:
        for i in [0, 100, 2000, 5000, 10000]:
            i = i % len(dataset)
            sample = dataset[i]
            assert len(sample) == 3
            v, q, a = sample
            assert v.shape == expected_image_shape
            assert len(q) == expected_question_length
            assert list(a.size()) == [classes]


def dataloaders_ok(
        dm: LightningDataModule,
        expected_image_shape: Sequence,
        expected_question_length: int,
        classes: int,
):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Validation and Test set are equal in this " "Dataset.",
        )
        dm.setup(stage=None)
    dataloaders = [
        dm.train_dataloader(),
        dm.val_dataloader(),
        dm.test_dataloader(),
    ]
    for dl in dataloaders:
        max_batch = len(dl) // expected_image_shape[0]
        for i, batch in enumerate(dl):
            if i == 5 or i >= max_batch:
                break
            v, _q, a = batch
            q = torch.stack(_q).T
            assert v.shape == expected_image_shape
            assert q.shape == (
                expected_image_shape[0],
                expected_question_length,
            )
            assert a.shape == (expected_image_shape[0], classes)


def _test_dm_shuffle_false(dm: LightningDataModule):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Validation and Test set are equal in this " "Dataset.",
        )
        dm.setup(None)
    # should not be equal due to transforms being random!
    assert not torch.equal(
        next(iter(dm.train_dataloader()))[0],
        next(iter(dm.train_dataloader()))[0],
    )
    assert torch.equal(
        next(iter(dm.val_dataloader()))[0], next(iter(dm.val_dataloader()))[0]
    )
    assert torch.equal(
        next(iter(dm.test_dataloader()))[0], next(iter(dm.test_dataloader()))[0]
    )


def _test_dm_shuffle_true(dm: LightningDataModule):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Validation and Test set are equal in this " "Dataset.",
        )
        dm.setup(None)
    assert not torch.equal(
        next(iter(dm.train_dataloader()))[0],
        next(iter(dm.train_dataloader()))[0],
    )
    assert not torch.equal(
        next(iter(dm.val_dataloader()))[0], next(iter(dm.val_dataloader()))[0]
    )
    assert not torch.equal(
        next(iter(dm.test_dataloader()))[0], next(iter(dm.test_dataloader()))[0]
    )


def _test_dm_shuffle_none(dm: LightningDataModule):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Validation and Test set are equal in this " "Dataset.",
        )
        dm.setup(None)
    assert not torch.equal(
        next(iter(dm.train_dataloader()))[0],
        next(iter(dm.train_dataloader()))[0],
    )
    assert torch.equal(
        next(iter(dm.val_dataloader()))[0], next(iter(dm.val_dataloader()))[0]
    )
    assert torch.equal(
        next(iter(dm.test_dataloader()))[0], next(iter(dm.test_dataloader()))[0]
    )
