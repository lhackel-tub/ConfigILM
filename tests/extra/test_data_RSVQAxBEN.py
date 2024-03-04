from typing import Tuple
import warnings

import pytest

from . import test_data_common
from configilm.extra.DataModules.RSVQAxBEN_DataModule import RSVQAxBENDataModule
from configilm.extra.DataSets.RSVQAxBEN_DataSet import resolve_data_dir
from configilm.extra.DataSets.RSVQAxBEN_DataSet import RSVQAxBENDataSet


@pytest.fixture
def data_dirs():
    return resolve_data_dir(None, allow_mock=True, force_mock=True)


dataset_params = ["train", "val", "test", None]

class_number = [10, 100, 250, 1000, 1234]
img_sizes = [60, 120, 128, 144, 256]
channels_pass = [2, 3, 4, 10, 12]  # accepted channel configs
channels_fail = [5, 1, 0, -1, 13]  # not accepted configs
img_shapes_pass = [(c, hw, hw) for c in channels_pass for hw in img_sizes]
img_shapes_fail = [(c, hw, hw) for c in channels_fail for hw in img_sizes]
max_img_idxs = [0, 1, 100, 10000]
max_img_idxs_too_large = [600_000, 1_000_000]

# these are part of the names used in mock data
# to see all names, open the lmdb env, create transaction and read
# [name for name, _ in txn.cursor()]
mock_s2_names = [
    "S2A_MSIL2A_20170613T101031_0_45",
    "S2A_MSIL2A_20170613T101031_26_57",
    "S2A_MSIL2A_20170613T101031_34_32",
    "S2A_MSIL2A_20170613T101031_39_24",
    "S2A_MSIL2A_20170617T113321_48_5",
    "S2A_MSIL2A_20170701T093031_2_52",
    "S2A_MSIL2A_20170701T093031_31_31",
    "S2A_MSIL2A_20170701T093031_43_67",
    "S2A_MSIL2A_20170701T093031_50_51",
    "S2A_MSIL2A_20170701T093031_63_35",
    "S2A_MSIL2A_20170701T093031_77_24",
    "S2A_MSIL2A_20170717T113321_82_31",
    "S2A_MSIL2A_20170816T095031_79_10",
    "S2A_MSIL2A_20170818T103021_43_48",
    "S2A_MSIL2A_20171002T112112_10_57",
    "S2A_MSIL2A_20171002T112112_34_76",
]

mock_data_dict = {
    i: {
        "type": "LC",
        "question": "What is the question?",
        "answer": f"{i % 2345}",
        "S2_name": mock_s2_names[i % len(mock_s2_names)],
    }
    for i in range(15000)
}


@pytest.mark.parametrize("split, classes", [(s, c) for s in dataset_params for c in class_number])
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_4c_ben_dataset_splits(data_dirs, split: str, classes: int):
    img_size = (4, 120, 120)
    seq_length = 32

    ds = RSVQAxBENDataSet(
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
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_ds_imgsize_pass(data_dirs, img_size: Tuple[int, int, int]):
    ds = RSVQAxBENDataSet(data_dirs=data_dirs, split="val", img_size=img_size, num_classes=1000, seq_length=32)

    test_data_common.dataset_ok(
        dataset=ds,
        expected_image_shape=img_size,
        expected_length=None,
        classes=1000,
        expected_question_length=32,
    )


@pytest.mark.parametrize("img_size", img_shapes_fail)
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_ds_imgsize_fail(data_dirs, img_size: Tuple[int, int, int]):
    with pytest.raises(AssertionError):
        _ = RSVQAxBENDataSet(
            data_dirs=data_dirs,
            split="val",
            img_size=img_size,
            num_classes=1000,
            seq_length=32,
        )


@pytest.mark.parametrize("max_len", [1, 16, 29, 30, None, -1])
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_ds_max_img_idx(data_dirs, max_len: int):
    ds = RSVQAxBENDataSet(data_dirs=data_dirs, max_len=max_len)
    expected_len = 30
    len_ds = expected_len if max_len is None or max_len > expected_len or max_len == -1 else max_len
    test_data_common.dataset_ok(
        dataset=ds,
        expected_image_shape=(12, 120, 120),
        expected_length=len_ds,
        classes=1000,
        expected_question_length=64,
    )


@pytest.mark.parametrize("max_img_index", [32, 20000, 100_000, 10_000_000])
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_ds_max_img_idx_too_large(data_dirs, max_img_index: int):
    ds = RSVQAxBENDataSet(data_dirs=data_dirs, max_len=max_img_index)
    assert len(ds) < max_img_index


@pytest.mark.parametrize("classes", [1, 5, 10, 50, 100, 1000, 2345, 5000, 15000, 25000])
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_ds_classes(data_dirs, classes: int):
    ds = RSVQAxBENDataSet(data_dirs=data_dirs, num_classes=classes, split="train")
    test_data_common._assert_classes_beyond_border_invalid(ds, classes, 2)


@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_ben_dm_lightning(data_dirs):
    dm = RSVQAxBENDataModule(data_dirs=data_dirs)
    test_data_common._assert_dm_correct_lightning_version(dm)


@pytest.mark.parametrize("split", dataset_params)
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_dm_default(data_dirs, split: str):
    dm = RSVQAxBENDataModule(data_dirs=data_dirs)
    split2stage = {"train": "fit", "val": "fit", "test": "test", None: None}
    dm.setup(stage=split2stage[split])
    dm.prepare_data()
    test_data_common._assert_datasets_set_correctly_for_split(dm, split, (12, 120, 120), None, 64, 1000)


@pytest.mark.parametrize("bs", [1, 2, 4, 8, 16, 32, 13, 27])
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_dm_dataloaders(data_dirs, bs: int):
    dm = RSVQAxBENDataModule(data_dirs=data_dirs, batch_size=bs, num_workers_dataloader=0, pin_memory=False)
    test_data_common.dataloaders_ok(
        dm,
        expected_image_shape=(bs, 12, 120, 120),
        expected_question_length=64,
        classes=1000,
    )


@pytest.mark.filterwarnings('ignore:Shuffle was set to False.')
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_dm_shuffle_false(data_dirs):
    dm = RSVQAxBENDataModule(data_dirs=data_dirs, shuffle=False, num_workers_dataloader=0, pin_memory=False)
    test_data_common._test_dm_shuffle_false(dm)


@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_dm_shuffle_none(data_dirs):
    dm = RSVQAxBENDataModule(data_dirs=data_dirs, shuffle=None, num_workers_dataloader=0, pin_memory=False)
    test_data_common._test_dm_shuffle_none(dm)


@pytest.mark.filterwarnings('ignore:Shuffle was set to True.')
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_dm_shuffle_true(data_dirs):
    dm = RSVQAxBENDataModule(data_dirs=data_dirs, shuffle=True, num_workers_dataloader=0, pin_memory=False)
    test_data_common._test_dm_shuffle_true(dm)
