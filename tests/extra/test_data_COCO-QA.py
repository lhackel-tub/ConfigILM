import warnings
from typing import Sequence
from typing import Union

import pytest
import torch

from configilm.extra.COCOQA_DataModule import COCOQADataModule
from configilm.extra.COCOQA_DataModule import COCOQADataSet
from configilm.extra.COCOQA_DataModule import resolve_cocoqa_data_dir


@pytest.fixture
def data_dir():
    return resolve_cocoqa_data_dir(None, force_mock=True)


def dataset_ok(
    dataset: Union[COCOQADataSet, None],
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
    dm: COCOQADataModule,
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


@pytest.mark.parametrize("split", ["train", "test", None])
def test_ds_default(data_dir, split):
    img_size = (3, 120, 120)
    ds = COCOQADataSet(data_dir, split=split)
    length = {"train": 78_736, "test": 38_948, None: 78_736 + 38_948}[split]
    mocked_datadir = "mock" in data_dir
    length = (
        25 if mocked_datadir and split is not None else 50 if mocked_datadir else length
    )

    dataset_ok(
        dataset=ds,
        expected_image_shape=img_size,
        expected_question_length=64,
        expected_length=length,
        classes=430,
    )


@pytest.mark.parametrize(
    "max_img_idx",
    [None, -1, 1, 5, 20, 50, 200, 2_000, 20_000, 117_683, 117_684, 117_685, 200_000],
)
def test_ds_max_img_idx(data_dir, max_img_idx):
    img_size = (3, 120, 120)
    ds = COCOQADataSet(data_dir, split=None, max_img_idx=max_img_idx)
    mocked_datadir = "mock" in data_dir
    max_len = 50 if mocked_datadir else 78_736 + 38_948
    length = (
        max_len
        if max_img_idx is None or max_img_idx > max_len or max_img_idx == -1
        else max_img_idx
    )

    dataset_ok(
        dataset=ds,
        expected_image_shape=img_size,
        expected_question_length=64,
        expected_length=length,
        classes=430,
    )


dataset_params = ["train", "val", "test", None]


@pytest.mark.parametrize("split", dataset_params)
def test_dm_default(data_dir, split: str):
    dm = COCOQADataModule(data_dir=data_dir)
    split2stage = {"train": "fit", "val": "fit", "test": "test", None: None}

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Validation and Test set are equal in this " "Dataset.",
        )
        dm.setup(stage=split2stage[split])
    dm.prepare_data()
    if split in ["train", "val"]:
        dataset_ok(
            dm.train_ds,
            expected_image_shape=(3, 120, 120),
            expected_length=None,
            classes=430,
            expected_question_length=64,
        )
        dataset_ok(
            dm.val_ds,
            expected_image_shape=(3, 120, 120),
            expected_length=None,
            classes=430,
            expected_question_length=64,
        )
        assert dm.test_ds is None
    elif split == "test":
        dataset_ok(
            dm.test_ds,
            expected_image_shape=(3, 120, 120),
            expected_length=None,
            classes=430,
            expected_question_length=64,
        )
        assert dm.train_ds is None
        assert dm.val_ds is None
    elif split is None:
        for ds in [dm.train_ds, dm.val_ds, dm.test_ds]:
            dataset_ok(
                ds,
                expected_image_shape=(3, 120, 120),
                expected_length=None,
                classes=430,
                expected_question_length=64,
            )
    else:
        ValueError(f"split {split} unknown")


@pytest.mark.parametrize("bs", [1, 2, 3, 4, 16, 32])
def test_dm_dataloader(data_dir, bs: int):
    dm = COCOQADataModule(data_dir=data_dir, batch_size=bs)
    dataloaders_ok(
        dm,
        expected_image_shape=(bs, 3, 120, 120),
        expected_question_length=64,
        classes=430,
    )


def test_dm_shuffle_false(data_dir):
    dm = COCOQADataModule(data_dir=data_dir, shuffle=False)

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


def test_dm_shuffle_none(data_dir):
    dm = COCOQADataModule(data_dir=data_dir, shuffle=None)

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


def test_dm_shuffle_true(data_dir):
    dm = COCOQADataModule(data_dir=data_dir, shuffle=True)

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
