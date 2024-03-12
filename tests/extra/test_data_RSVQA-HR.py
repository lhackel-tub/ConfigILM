from typing import Tuple

import pytest
import torch

from . import test_data_common
from configilm.extra.DataModules.RSVQAHR_DataModule import RSVQAHRDataModule
from configilm.extra.DataSets.RSVQAHR_DataSet import resolve_data_dir
from configilm.extra.DataSets.RSVQAHR_DataSet import RSVQAHRDataSet


@pytest.fixture
def data_dirs():
    return resolve_data_dir(None, allow_mock=True, force_mock=True)


dataset_params = ["train", "val", "test", "test_phili", None]

class_number = [10, 100, 250, 1000, 1234]
img_sizes = [60, 120, 128, 144, 256]
channels_pass = [3]  # accepted channel configs
channels_fail = [1, 5, 2, 0, -1, 10, 12, 13]  # not accepted configs
img_shapes_pass = [(c, hw, hw) for c in channels_pass for hw in img_sizes]
img_shapes_fail = [(c, hw, hw) for c in channels_fail for hw in img_sizes]
max_img_idxs = [0, 1, 100, 1_000]
max_img_idxs_too_large = [600_000, 1_000_000]


@pytest.mark.parametrize("split, classes", [(s, c) for s in dataset_params for c in class_number])
@pytest.mark.filterwarnings("ignore:No tokenizer was provided,")
def test_basic_dataset_splits(data_dirs, split: str, classes: int):
    img_size = (3, 256, 256)
    seq_length = 32

    ds = RSVQAHRDataSet(
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
@pytest.mark.filterwarnings("ignore:No tokenizer was provided,")
def test_ds_imgsize_pass(data_dirs, img_size: Tuple[int, int, int]):
    ds = RSVQAHRDataSet(data_dirs=data_dirs, split="val", img_size=img_size, num_classes=94, seq_length=32)

    test_data_common.dataset_ok(
        dataset=ds,
        expected_image_shape=img_size,
        expected_length=None,
        classes=94,
        expected_question_length=32,
    )


@pytest.mark.parametrize("img_size", img_shapes_fail)
@pytest.mark.filterwarnings("ignore:No tokenizer was provided,")
def test_ds_imgsize_fail(data_dirs, img_size: Tuple[int, int, int]):
    with pytest.raises(AssertionError):
        _ = RSVQAHRDataSet(
            data_dirs=data_dirs,
            split="val",
            img_size=img_size,
            num_classes=94,
            seq_length=64,
        )


@pytest.mark.parametrize("max_len", [1, 16, 74, 1200, 1201, None, -1])
@pytest.mark.filterwarnings("ignore:No tokenizer was provided,")
def test_ds_max_img_idx(data_dirs, max_len):
    ds = RSVQAHRDataSet(data_dirs=data_dirs, max_len=max_len)
    expected_len = 1201
    len_ds = expected_len if max_len is None or max_len > expected_len or max_len == -1 else max_len
    test_data_common.dataset_ok(
        dataset=ds,
        expected_image_shape=(3, 256, 256),
        expected_length=len_ds,
        classes=94,
        expected_question_length=64,
    )


@pytest.mark.parametrize("max_len", [1202, 20_000, 100_000, 10_000_000])
@pytest.mark.filterwarnings("ignore:No tokenizer was provided,")
def test_ds_max_img_idx_too_large(data_dirs, max_len: int):
    ds = RSVQAHRDataSet(data_dirs=data_dirs, max_len=max_len)
    assert len(ds) < max_len


@pytest.mark.parametrize("classes", [1, 5, 10, 50, 100, 1000, 2345, 5000, 15000, 25000])
@pytest.mark.filterwarnings("ignore:No tokenizer was provided,")
def test_ds_classes(data_dirs, classes: int):
    ds = RSVQAHRDataSet(data_dirs=data_dirs, num_classes=classes, split="train")
    max_classes_mock_set = 14  # number of classes in the mock data
    test_data_common._assert_classes_beyond_border_invalid(ds, classes, max_classes_mock_set)


@pytest.mark.parametrize("classes", [1, 5, 10, 50, 100, 1000, 2345, 5000, 15000, 25000])
@pytest.mark.filterwarnings("ignore:No tokenizer was provided,")
def test_ds_classes_no_buckets(data_dirs, classes: int):
    ds = RSVQAHRDataSet(data_dirs=data_dirs, num_classes=classes, split="train", quantize_answers=False)
    max_classes_mock_set = 26  # number of classes in the mock data
    test_data_common._assert_classes_beyond_border_invalid(ds, classes, max_classes_mock_set)


@pytest.mark.filterwarnings("ignore:No tokenizer was provided,")
def test_ben_dm_lightning(data_dirs):
    dm = RSVQAHRDataModule(data_dirs=data_dirs)
    test_data_common._assert_dm_correct_lightning_version(dm)


@pytest.mark.parametrize("split", dataset_params)
@pytest.mark.filterwarnings("ignore:No tokenizer was provided,")
def test_dm_default(data_dirs, split: str):
    dm = RSVQAHRDataModule(data_dirs=data_dirs)
    split2stage = {
        "train": "fit",
        "val": "fit",
        "test": "test",
        "test_phili": "test",
        None: None,
    }
    dm.setup(stage=split2stage[split])
    dm.prepare_data()
    test_data_common._assert_datasets_set_correctly_for_split(dm, split, (3, 256, 256), None, 64, 94)


@pytest.mark.parametrize("bs", [1, 2, 4, 8, 16, 32, 13, 27])
@pytest.mark.filterwarnings("ignore:No tokenizer was provided,")
def test_dm_dataloaders(data_dirs, bs: int):
    dm = RSVQAHRDataModule(data_dirs=data_dirs, batch_size=bs, num_workers_dataloader=0, pin_memory=False)
    test_data_common.dataloaders_ok(
        dm,
        expected_image_shape=(bs, 3, 256, 256),
        expected_question_length=64,
        classes=94,
    )


@pytest.mark.filterwarnings("ignore:Shuffle was set to False.")
@pytest.mark.filterwarnings("ignore:No tokenizer was provided,")
def test_dm_shuffle_false(data_dirs):
    dm = RSVQAHRDataModule(data_dirs=data_dirs, shuffle=False, num_workers_dataloader=0, pin_memory=False)
    test_data_common._test_dm_shuffle_false(dm)


@pytest.mark.filterwarnings("ignore:No tokenizer was provided,")
def test_dm_shuffle_none(data_dirs):
    dm = RSVQAHRDataModule(data_dirs=data_dirs, shuffle=None, num_workers_dataloader=0, pin_memory=False)
    test_data_common._test_dm_shuffle_none(dm)


@pytest.mark.filterwarnings("ignore:Shuffle was set to True.")
@pytest.mark.filterwarnings("ignore:No tokenizer was provided,")
def test_dm_shuffle_true(data_dirs):
    dm = RSVQAHRDataModule(data_dirs=data_dirs, shuffle=True, num_workers_dataloader=0, pin_memory=False)
    test_data_common._test_dm_shuffle_true(dm)


@pytest.mark.filterwarnings("ignore:No tokenizer was provided,")
def test_different_test_splits(data_dirs):
    dm = RSVQAHRDataModule(
        data_dirs=data_dirs,
        use_phili_test=False,
        num_workers_dataloader=0,
        pin_memory=False,
    )
    dm.setup("test")
    dm_p = RSVQAHRDataModule(
        data_dirs=data_dirs,
        use_phili_test=True,
        num_workers_dataloader=0,
        pin_memory=False,
    )
    dm_p.setup("test")

    assert not torch.equal(next(iter(dm.test_dataloader()))[0], next(iter(dm_p.test_dataloader()))[0])
