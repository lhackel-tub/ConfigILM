from typing import Sequence
from typing import Tuple
from typing import Union

import pytest
import torch

from configilm.extra.HRVQA_DataModule import HRVQADataModule
from configilm.extra.HRVQADataSet import HRVQADataSet
from configilm.extra.HRVQADataSet import resolve_data_dir


@pytest.fixture
def data_dir():
    return resolve_data_dir(None, allow_mock=True, force_mock=True)


dataset_params = ["train", "val", None]  # this dataset does not support test split

class_number = [10, 100, 250, 1000, 1234]
img_sizes = [60, 120, 128, 144, 256, 1024]
channels_pass = [1, 3]  # accepted channel configs
channels_fail = [2, 4, 0, -1, 10, 12]  # not accepted configs
img_shapes_pass = [(c, hw, hw) for c in channels_pass for hw in img_sizes]
img_shapes_fail = [(c, hw, hw) for c in channels_fail for hw in img_sizes]
max_img_idxs = [0, 1, 100, 10_000]
max_img_idxs_too_large = [600_000, 1_000_000]

# these are part of the names used in mock data
mock_s2_names = [
    "6991",
    "7731",
    "9342",
    "11806",
    "11844",
    "12900",
    "14652",
    "20116",
    "21817",
    "23843",
    "27286",
    "27600",
    "28070",
    "29400",
    "30808",
    "30813",
    "30988",
    "31203",
    "32675",
    "32822",
    "33973",
    "41277",
    "42569",
    "43759",
    "45386",
    "46738",
    "47903",
    "49129",
    "51504",
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


def dataset_ok(
    dataset: Union[HRVQADataSet, None],
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
    dm: HRVQADataModule,
    expected_image_shape: Sequence,
    expected_question_length: int,
    classes: int,
):
    dm.setup(stage=None)
    dataloaders = [
        dm.train_dataloader(),
        dm.val_dataloader(),
        # dm.test_dataloader(),  # no test DL for this dataset
    ]
    for dl in dataloaders:
        max_batch = len(dl) // expected_image_shape[0]
        for i, batch in enumerate(dl):
            if i == 5 or i >= max_batch:
                break
            v, q, a = batch
            assert v.shape == expected_image_shape
            assert len(q) == expected_question_length
            assert len(q[0]) == expected_image_shape[0]
            assert a.shape == (expected_image_shape[0], classes)


def test_ds_default(data_dir):
    ds = HRVQADataSet(root_dir=data_dir)

    dataset_ok(
        dataset=ds,
        expected_image_shape=(3, 1024, 1024),
        expected_length=None,
        classes=1_000,
        expected_question_length=32,
    )


@pytest.mark.parametrize(
    "split, classes", [(s, c) for s in dataset_params for c in class_number]
)
def test_3c_dataset_splits(data_dir, split: str, classes: int):
    img_size = (3, 128, 128)
    seq_length = 32

    ds = HRVQADataSet(
        root_dir=data_dir,
        split=split,
        img_size=img_size,
        classes=classes,
        seq_length=seq_length,
    )

    dataset_ok(
        dataset=ds,
        expected_image_shape=img_size,
        expected_length=None,
        classes=classes,
        expected_question_length=seq_length,
    )


@pytest.mark.parametrize("classes", class_number)
def test_3c_dataset_splits_test(data_dir, classes: int):
    img_size = (3, 128, 128)
    seq_length = 32
    with pytest.raises(AssertionError):
        # test not supported by this DS
        _ = HRVQADataSet(
            root_dir=data_dir,
            split="test",
            img_size=img_size,
            classes=classes,
            seq_length=seq_length,
        )


@pytest.mark.parametrize("img_size", img_shapes_pass)
def test_ds_imgsize_pass(data_dir, img_size: Tuple[int, int, int]):
    ds = HRVQADataSet(
        root_dir=data_dir, split="val", img_size=img_size, classes=1000, seq_length=32
    )

    dataset_ok(
        dataset=ds,
        expected_image_shape=img_size,
        expected_length=None,
        classes=1000,
        expected_question_length=32,
    )


@pytest.mark.parametrize("img_size", img_shapes_fail)
def test_ds_imgsize_fail(data_dir, img_size: Tuple[int, int, int]):
    with pytest.raises(AssertionError):
        _ = HRVQADataSet(
            root_dir=data_dir,
            split="val",
            img_size=img_size,
            classes=1000,
            seq_length=32,
        )


@pytest.mark.parametrize("max_img_index", [1, 16, 74, 75, None, -1])
def test_ds_max_img_idx(data_dir, max_img_index: int):
    ds = HRVQADataSet(root_dir=data_dir, max_img_idx=max_img_index)
    max_len = 20
    len_ds = (
        max_len
        if max_img_index is None or max_img_index > max_len or max_img_index == -1
        else max_img_index
    )
    dataset_ok(
        dataset=ds,
        expected_image_shape=(3, 1024, 1024),
        expected_length=len_ds,
        classes=1000,
        expected_question_length=32,
    )


@pytest.mark.parametrize("idx", [0, 1, 5, 10, 19, 20, 50, 100, 1_000, 10_000, 100_000])
def test_ds_img_access_by_index(data_dir, idx: int):
    ds = HRVQADataSet(root_dir=data_dir)
    expected_len = 20
    assert len(ds) == expected_len, f"There should be {expected_len} samples exactly."
    if idx < len(ds):
        _ = ds[idx]
    else:
        with pytest.raises(IndexError):
            _ = ds[idx]


@pytest.mark.parametrize("max_img_index", [76, 20000, 100_000, 10_000_000])
def test_ds_max_img_idx_too_large(data_dir, max_img_index: int):
    ds = HRVQADataSet(root_dir=data_dir, max_img_idx=max_img_index)
    assert len(ds) < max_img_index


@pytest.mark.parametrize("classes", [1, 5, 10, 50, 100, 1000, 2345, 5000, 15000, 25000])
def test_ds_classes(data_dir, classes: int):
    ds = HRVQADataSet(root_dir=data_dir, classes=classes, split="train")
    gt_classes = 8  # number of classes in the mock data
    assert ds.classes == classes
    assert len(ds.selected_answers) == classes
    if classes <= gt_classes:
        for i in range(classes):
            assert ds.selected_answers[i] != "INVALID"
    else:
        for i in range(gt_classes):
            assert ds.selected_answers[i] != "INVALID"
        for i in range(gt_classes, classes):
            assert ds.selected_answers[i] == "INVALID"


@pytest.mark.parametrize("split", dataset_params)
def test_dm_default(data_dir, split: str):
    dm = HRVQADataModule(data_dir=data_dir)
    split2stage = {"train": "fit", "val": "fit", "test": "test", None: None}
    dm.setup(stage=split2stage[split])
    dm.prepare_data()
    if split in ["train", "val"]:
        dataset_ok(
            dm.train_ds,
            expected_image_shape=(3, 1024, 1024),
            expected_length=None,
            classes=1_000,
            expected_question_length=32,
        )
        dataset_ok(
            dm.val_ds,
            expected_image_shape=(3, 1024, 1024),
            expected_length=None,
            classes=1000,
            expected_question_length=32,
        )
        assert dm.test_ds is None
    elif split is None:
        for ds in [dm.train_ds, dm.val_ds]:
            dataset_ok(
                ds,
                expected_image_shape=(3, 1024, 1024),
                expected_length=None,
                classes=1000,
                expected_question_length=32,
            )
    else:
        ValueError(f"split {split} unknown")


@pytest.mark.parametrize("bs", [1, 2, 4, 8, 16, 32, 13, 27])
def test_dm_dataloaders_bs(data_dir, bs: int):
    dm = HRVQADataModule(data_dir=data_dir, batch_size=bs)
    dataloaders_ok(
        dm,
        expected_image_shape=(bs, 3, 1024, 1024),
        expected_question_length=32,
        classes=1000,
    )


@pytest.mark.parametrize("img_size", [[1], [1, 2], [1, 2, 3, 4]])
def test_dm_dataloaders_img_size(data_dir, img_size):
    with (pytest.raises(ValueError)):
        _ = HRVQADataModule(data_dir=data_dir, img_size=img_size)


@pytest.mark.parametrize("stage", [None, "fit"])
def test_dm_dataloaders_setup(data_dir, stage):
    dm = HRVQADataModule(data_dir=data_dir)
    dm.setup(stage)
    assert dm.train_dataloader() is not None, "Train Dataloader should not be None"
    assert dm.val_dataloader() is not None, "Val Dataloader should not be None"
    assert dm.test_dataloader() is None, "Test Dataloader should be None"


def test_dm_shuffle_false(data_dir):
    dm = HRVQADataModule(data_dir=data_dir, shuffle=False)
    dm.setup(None)
    # should not be equal due to transforms being random!
    assert not torch.equal(
        next(iter(dm.train_dataloader()))[0],
        next(iter(dm.train_dataloader()))[0],
    )
    assert torch.equal(
        next(iter(dm.val_dataloader()))[0], next(iter(dm.val_dataloader()))[0]
    )


def test_dm_shuffle_none(data_dir):
    dm = HRVQADataModule(data_dir=data_dir, shuffle=None)
    dm.setup(None)
    assert not torch.equal(
        next(iter(dm.train_dataloader()))[0],
        next(iter(dm.train_dataloader()))[0],
    )
    assert torch.equal(
        next(iter(dm.val_dataloader()))[0], next(iter(dm.val_dataloader()))[0]
    )


def test_dm_shuffle_true(data_dir):
    dm = HRVQADataModule(data_dir=data_dir, shuffle=True)
    dm.setup(None)
    assert not torch.equal(
        next(iter(dm.train_dataloader()))[0],
        next(iter(dm.train_dataloader()))[0],
    )
    assert not torch.equal(
        next(iter(dm.val_dataloader()))[0], next(iter(dm.val_dataloader()))[0]
    )


def test_dm_test_stage_setup(data_dir):
    dm = HRVQADataModule(data_dir=data_dir)
    with pytest.raises(NotImplementedError):
        dm.setup("test")


def test_dm_predict_stage_setup(data_dir):
    dm = HRVQADataModule(data_dir=data_dir)
    with pytest.raises(NotImplementedError):
        dm.setup("predict")
