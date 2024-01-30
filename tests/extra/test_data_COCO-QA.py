import warnings

import pytest

from configilm.extra.DataModules.COCOQA_DataModule import COCOQADataModule
from configilm.extra.DataSets.COCOQA_DataSet import COCOQADataSet
from configilm.extra.DataSets.COCOQA_DataSet import resolve_data_dir
from . import test_data_common


@pytest.fixture
def data_dirs():
    return resolve_data_dir(None, allow_mock=True, force_mock=True)


@pytest.mark.parametrize("split", ["train", "test", None])
def test_ds_default(data_dirs, split):
    img_size = (3, 120, 120)
    ds = COCOQADataSet(data_dirs, split=split)
    length = {"train": 10, "test": 10, None: 20}[split]

    test_data_common.dataset_ok(
        dataset=ds,
        expected_image_shape=img_size,
        expected_question_length=64,
        expected_length=length,
        classes=430,
    )


@pytest.mark.parametrize(
    "max_len",
    [None, -1, 1, 5, 20, 50, 200, 2_000, 20_000, 117_683, 117_684, 117_685, 200_000],
)
def test_ds_max_img_idx(data_dirs, max_len):
    img_size = (3, 120, 120)
    ds = COCOQADataSet(data_dirs, split=None, max_len=max_len)

    length = 20 if max_len in [None, -1] else min(max_len, 20)

    test_data_common.dataset_ok(
        dataset=ds,
        expected_image_shape=img_size,
        expected_question_length=64,
        expected_length=length,
        classes=430,
    )


dataset_params = ["train", "val", "test", None]


@pytest.mark.parametrize("split", dataset_params)
def test_dm_default(data_dirs, split: str):
    dm = COCOQADataModule(data_dirs=data_dirs)
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
        test_data_common.dataset_ok(
            dm.train_ds,
            expected_image_shape=(3, 120, 120),
            expected_length=None,
            classes=430,
            expected_question_length=64,
        )
        test_data_common.dataset_ok(
            dm.val_ds,
            expected_image_shape=(3, 120, 120),
            expected_length=None,
            classes=430,
            expected_question_length=64,
        )
        assert dm.test_ds is None
    elif split == "test":
        test_data_common.dataset_ok(
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
            test_data_common.dataset_ok(
                ds,
                expected_image_shape=(3, 120, 120),
                expected_length=None,
                classes=430,
                expected_question_length=64,
            )
    else:
        ValueError(f"split {split} unknown")


@pytest.mark.parametrize("bs", [1, 2, 3, 4, 16, 32])
def test_dm_dataloader(data_dirs, bs: int):
    dm = COCOQADataModule(
        data_dirs=data_dirs, batch_size=bs, num_workers_dataloader=0, pin_memory=False
    )
    test_data_common.dataloaders_ok(
        dm,
        expected_image_shape=(bs, 3, 120, 120),
        expected_question_length=64,
        classes=430,
    )


def test_dm_shuffle_false(data_dirs):
    dm = COCOQADataModule(
        data_dirs=data_dirs, shuffle=False, num_workers_dataloader=0, pin_memory=False
    )
    test_data_common._test_dm_shuffle_false(dm)


def test_dm_shuffle_none(data_dirs):
    dm = COCOQADataModule(
        data_dirs=data_dirs, shuffle=None, num_workers_dataloader=0, pin_memory=False
    )
    test_data_common._test_dm_shuffle_none(dm)


def test_dm_shuffle_true(data_dirs):
    dm = COCOQADataModule(
        data_dirs=data_dirs, shuffle=True, num_workers_dataloader=0, pin_memory=False
    )
    test_data_common._test_dm_shuffle_true(dm)
