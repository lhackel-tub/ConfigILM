from typing import Tuple

import pytest

from . import test_data_common
from configilm.extra.DataModules.RSVQALR_DataModule import RSVQALRDataModule
from configilm.extra.DataSets.RSVQALR_DataSet import resolve_data_dir
from configilm.extra.DataSets.RSVQALR_DataSet import RSVQALRDataSet


@pytest.fixture
def data_dirs():
    return resolve_data_dir(None, allow_mock=True, force_mock=True)


dataset_params = ["train", "val", "test", None]

class_number = [10, 100, 250, 1000, 1234]
img_sizes = [60, 120, 128, 144, 256]
channels_pass = [3]  # accepted channel configs
channels_fail = [1, 5, 2, 0, -1, 10, 12, 13]  # not accepted configs
img_shapes_pass = [(c, hw, hw) for c in channels_pass for hw in img_sizes]
img_shapes_fail = [(c, hw, hw) for c in channels_fail for hw in img_sizes]
max_img_idxs = [0, 1, 100, 1_000]
max_img_idxs_too_large = [600_000, 1_000_000]


@pytest.mark.parametrize("split, classes", [(s, c) for s in dataset_params for c in class_number])
def test_basic_dataset_splits(data_dirs, split: str, classes: int):
    img_size = (3, 256, 256)
    seq_length = 32

    ds = RSVQALRDataSet(
        data_dirs=data_dirs,
        split=split,
        img_size=img_size,
        num_classes=classes,
        seq_length=seq_length,
    )

    test_data_common.dataset_ok(
        dataset=ds,
        expected_image_shape=img_size,
        expected_length=None,
        classes=classes,
        expected_question_length=seq_length,
    )


@pytest.mark.parametrize("img_size", img_shapes_pass)
def test_ds_imgsize_pass(data_dirs, img_size: Tuple[int, int, int]):
    ds = RSVQALRDataSet(data_dirs=data_dirs, split="val", img_size=img_size, num_classes=9, seq_length=32)

    test_data_common.dataset_ok(
        dataset=ds,
        expected_image_shape=img_size,
        expected_length=None,
        classes=9,
        expected_question_length=32,
    )


@pytest.mark.parametrize("img_size", img_shapes_fail)
def test_ds_imgsize_fail(data_dirs, img_size: Tuple[int, int, int]):
    with pytest.raises(AssertionError):
        _ = RSVQALRDataSet(
            data_dirs=data_dirs,
            split="val",
            img_size=img_size,
            num_classes=9,
            seq_length=32,
        )


@pytest.mark.parametrize("max_len", [1, 16, 74, 1_199, 1_200, None, -1])
def test_ds_max_img_idx(data_dirs, max_len: int):
    ds = RSVQALRDataSet(data_dirs=data_dirs, max_len=max_len)
    expected_len = 1_200
    len_ds = expected_len if max_len is None or max_len > expected_len or max_len == -1 else max_len
    test_data_common.dataset_ok(
        dataset=ds,
        expected_image_shape=(3, 256, 256),
        expected_length=len_ds,
        classes=9,
        expected_question_length=64,
    )


@pytest.mark.parametrize("max_len", [1_201, 20_000, 100_000, 10_000_000])
def test_ds_max_img_idx_too_large(data_dirs, max_len: int):
    ds = RSVQALRDataSet(data_dirs=data_dirs, max_len=max_len)
    assert len(ds) < max_len


@pytest.mark.parametrize("classes", [1, 5, 10, 50, 100, 1000, 2345, 5000, 15000, 25000])
def test_ds_classes(data_dirs, classes: int):
    ds = RSVQALRDataSet(data_dirs=data_dirs, num_classes=classes, split="train", quantize_answers=True)
    assert len(ds.answers) == classes
    max_classes_mock_set = 7  # number of classes in the mock data
    if classes <= max_classes_mock_set:
        for i in range(classes):
            assert ds.answers[i] != "INVALID"
    else:
        for i in range(max_classes_mock_set):
            assert ds.answers[i] != "INVALID"
        for i in range(max_classes_mock_set, classes):
            assert ds.answers[i] == "INVALID"


@pytest.mark.parametrize("classes", [1, 5, 10, 50, 100, 1000, 2345, 5000, 15000, 25000])
def test_ds_classes_unbucketed(data_dirs, classes: int):
    ds = RSVQALRDataSet(data_dirs=data_dirs, num_classes=classes, split="train", quantize_answers=False)
    assert len(ds.answers) == classes
    max_classes_mock_set = 36  # number of classes in the mock data
    if classes <= max_classes_mock_set:
        for i in range(classes):
            assert ds.answers[i] != "INVALID"
    else:
        for i in range(max_classes_mock_set):
            assert ds.answers[i] != "INVALID"
        for i in range(max_classes_mock_set, classes):
            assert ds.answers[i] == "INVALID"


@pytest.mark.parametrize("split", dataset_params)
def test_dm_default(data_dirs, split: str):
    dm = RSVQALRDataModule(data_dirs=data_dirs)
    split2stage = {"train": "fit", "val": "fit", "test": "test", None: None}
    dm.setup(stage=split2stage[split])
    dm.prepare_data()
    if split in ["train", "val"]:
        test_data_common.dataset_ok(
            dm.train_ds,
            expected_image_shape=(3, 256, 256),
            expected_length=None,
            classes=9,
            expected_question_length=64,
        )
        test_data_common.dataset_ok(
            dm.val_ds,
            expected_image_shape=(3, 256, 256),
            expected_length=None,
            classes=9,
            expected_question_length=64,
        )
        assert dm.test_ds is None
    elif split == "test":
        test_data_common.dataset_ok(
            dm.test_ds,
            expected_image_shape=(3, 256, 256),
            expected_length=None,
            classes=9,
            expected_question_length=64,
        )
        assert dm.train_ds is None
        assert dm.val_ds is None
    elif split is None:
        for ds in [dm.train_ds, dm.val_ds, dm.test_ds]:
            test_data_common.dataset_ok(
                ds,
                expected_image_shape=(3, 256, 256),
                expected_length=None,
                classes=9,
                expected_question_length=64,
            )
    else:
        ValueError(f"split {split} unknown")


@pytest.mark.parametrize("bs", [1, 2, 4, 8, 16, 32, 13, 27])
def test_dm_dataloaders(data_dirs, bs: int):
    dm = RSVQALRDataModule(data_dirs=data_dirs, batch_size=bs, num_workers_dataloader=0, pin_memory=False)
    test_data_common.dataloaders_ok(
        dm,
        expected_image_shape=(bs, 3, 256, 256),
        expected_question_length=64,
        classes=9,
    )


def test_dm_shuffle_false(data_dirs):
    dm = RSVQALRDataModule(data_dirs=data_dirs, shuffle=False, num_workers_dataloader=0, pin_memory=False)
    test_data_common._test_dm_shuffle_false(dm)


def test_dm_shuffle_none(data_dirs):
    dm = RSVQALRDataModule(data_dirs=data_dirs, shuffle=None, num_workers_dataloader=0, pin_memory=False)
    test_data_common._test_dm_shuffle_none(dm)


def test_dm_shuffle_true(data_dirs):
    dm = RSVQALRDataModule(data_dirs=data_dirs, shuffle=True, num_workers_dataloader=0, pin_memory=False)
    test_data_common._test_dm_shuffle_true(dm)
