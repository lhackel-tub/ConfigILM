from typing import Sequence
from typing import Tuple
from typing import Union

import pytest
import torch

from configilm.extra.DataModules.RSVQAHR_DataModule import RSVQAHRDataModule
from configilm.extra.DataSets.RSVQAHR_DataSet import resolve_data_dir
from configilm.extra.DataSets.RSVQAHR_DataSet import RSVQAHRDataSet


@pytest.fixture
def data_dir():
    return resolve_data_dir(None, allow_mock=True, force_mock=True)


dataset_params = ["train", "val", "test", "test_phili", None]

class_number = [10, 100, 250, 1000, 1234]
img_sizes = [60, 120, 128, 144, 256]
channels_pass = [1, 3]  # accepted channel configs
channels_fail = [5, 2, 0, -1, 10, 12, 13]  # not accepted configs
img_shapes_pass = [(c, hw, hw) for c in channels_pass for hw in img_sizes]
img_shapes_fail = [(c, hw, hw) for c in channels_fail for hw in img_sizes]
max_img_idxs = [0, 1, 100, 1_000]
max_img_idxs_too_large = [600_000, 1_000_000]


def dataset_ok(
    dataset: Union[RSVQAHRDataSet, None],
    expected_image_shape: Sequence,
    expected_question_length: int,
    expected_length: Union[int, None],
    classes: int,
):
    assert dataset is not None
    if expected_length is not None:
        assert len(dataset) == expected_length

    if len(dataset) > 0:
        for i in [0, 100, 2000, 5000, 10_000]:
            i = i % len(dataset)
            sample = dataset[i]
            assert len(sample) == 3
            v, q, a = sample
            assert v.shape == expected_image_shape
            assert len(q) == expected_question_length
            assert list(a.size()) == [classes]


def dataloaders_ok(
    dm: RSVQAHRDataModule,
    expected_image_shape: Sequence,
    expected_question_length: int,
    classes: int,
):
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
            v, q, a = batch
            assert v.shape == expected_image_shape
            assert len(q) == expected_question_length
            assert len(q[0]) == expected_image_shape[0]
            assert a.shape == (expected_image_shape[0], classes)


@pytest.mark.parametrize(
    "split, classes", [(s, c) for s in dataset_params for c in class_number]
)
def test_basic_dataset_splits(data_dir, split: str, classes: int):
    img_size = (3, 256, 256)
    seq_length = 32

    ds = RSVQAHRDataSet(
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


@pytest.mark.parametrize("img_size", img_shapes_pass)
def test_ds_imgsize_pass(data_dir, img_size: Tuple[int, int, int]):
    ds = RSVQAHRDataSet(
        root_dir=data_dir, split="val", img_size=img_size, classes=94, seq_length=32
    )

    dataset_ok(
        dataset=ds,
        expected_image_shape=img_size,
        expected_length=None,
        classes=94,
        expected_question_length=32,
    )


@pytest.mark.parametrize("img_size", img_shapes_fail)
def test_ds_imgsize_fail(data_dir, img_size: Tuple[int, int, int]):
    with pytest.raises(AssertionError):
        _ = RSVQAHRDataSet(
            root_dir=data_dir,
            split="val",
            img_size=img_size,
            classes=94,
            seq_length=32,
        )


@pytest.mark.parametrize("max_img_index", [1, 16, 74, 1200, 1201, None, -1])
def test_ds_max_img_idx(data_dir, max_img_index: int):
    ds = RSVQAHRDataSet(root_dir=data_dir, max_img_idx=max_img_index)
    max_len = 1201
    len_ds = (
        max_len
        if max_img_index is None or max_img_index > max_len or max_img_index == -1
        else max_img_index
    )
    dataset_ok(
        dataset=ds,
        expected_image_shape=(3, 256, 256),
        expected_length=len_ds,
        classes=94,
        expected_question_length=32,
    )


@pytest.mark.parametrize("max_img_index", [1202, 20_000, 100_000, 10_000_000])
def test_ds_max_img_idx_too_large(data_dir, max_img_index: int):
    ds = RSVQAHRDataSet(root_dir=data_dir, max_img_idx=max_img_index)
    assert len(ds) < max_img_index


@pytest.mark.parametrize("classes", [1, 5, 10, 50, 100, 1000, 2345, 5000, 15000, 25000])
def test_ds_classes(data_dir, classes: int):
    ds = RSVQAHRDataSet(root_dir=data_dir, classes=classes, split="train")
    assert ds.classes == classes
    assert len(ds.selected_answers) == classes
    max_classes_mock_set = 14  # number of classes in the mock data
    if classes <= max_classes_mock_set:
        for i in range(classes):
            assert ds.selected_answers[i] != "INVALID"
    else:
        for i in range(max_classes_mock_set):
            assert ds.selected_answers[i] != "INVALID"
        for i in range(max_classes_mock_set, classes):
            assert ds.selected_answers[i] == "INVALID"


@pytest.mark.parametrize("classes", [1, 5, 10, 50, 100, 1000, 2345, 5000, 15000, 25000])
def test_ds_classes_no_buckets(data_dir, classes: int):
    ds = RSVQAHRDataSet(
        root_dir=data_dir, classes=classes, split="train", quantize_answers=False
    )
    assert ds.classes == classes
    assert len(ds.selected_answers) == classes
    max_classes_mock_set = 26  # number of classes in the mock data
    if classes <= max_classes_mock_set:
        for i in range(classes):
            assert ds.selected_answers[i] != "INVALID"
    else:
        for i in range(max_classes_mock_set):
            assert ds.selected_answers[i] != "INVALID"
        for i in range(max_classes_mock_set, classes):
            assert ds.selected_answers[i] == "INVALID"


@pytest.mark.parametrize("split", dataset_params)
def test_dm_default(data_dir, split: str):
    dm = RSVQAHRDataModule(data_dir=data_dir)
    split2stage = {
        "train": "fit",
        "val": "fit",
        "test": "test",
        "test_phili": "test",
        None: None,
    }
    dm.setup(stage=split2stage[split])
    dm.prepare_data()
    if split in ["train", "val"]:
        dataset_ok(
            dm.train_ds,
            expected_image_shape=(3, 256, 256),
            expected_length=None,
            classes=94,
            expected_question_length=32,
        )
        dataset_ok(
            dm.val_ds,
            expected_image_shape=(3, 256, 256),
            expected_length=None,
            classes=94,
            expected_question_length=32,
        )
        assert dm.test_ds is None
    elif split == "test":
        dataset_ok(
            dm.test_ds,
            expected_image_shape=(3, 256, 256),
            expected_length=None,
            classes=94,
            expected_question_length=32,
        )
        assert dm.train_ds is None
        assert dm.val_ds is None
    elif split is None:
        for ds in [dm.train_ds, dm.val_ds, dm.test_ds]:
            dataset_ok(
                ds,
                expected_image_shape=(3, 256, 256),
                expected_length=None,
                classes=94,
                expected_question_length=32,
            )
    else:
        ValueError(f"split {split} unknown")


@pytest.mark.parametrize("bs", [1, 2, 4, 8, 16, 32, 13, 27])
def test_dm_dataloaders(data_dir, bs: int):
    dm = RSVQAHRDataModule(data_dir=data_dir, batch_size=bs)
    dataloaders_ok(
        dm,
        expected_image_shape=(bs, 3, 256, 256),
        expected_question_length=32,
        classes=94,
    )


@pytest.mark.parametrize("pi", [True, False])
def test_dm_print_on_setup(data_dir, pi):
    dm = RSVQAHRDataModule(data_dir=data_dir, print_infos=pi)
    dm.setup()


def test_dm_shuffle_false(data_dir):
    dm = RSVQAHRDataModule(data_dir=data_dir, shuffle=False)
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
    dm = RSVQAHRDataModule(data_dir=data_dir, shuffle=None)
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
    dm = RSVQAHRDataModule(data_dir=data_dir, shuffle=True)
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


def test_different_test_splits(data_dir):
    dm = RSVQAHRDataModule(data_dir=data_dir, use_phili_test=False)
    dm.setup("test")
    dm_p = RSVQAHRDataModule(data_dir=data_dir, use_phili_test=True)
    dm_p.setup("test")

    assert not torch.equal(
        next(iter(dm.test_dataloader()))[0], next(iter(dm_p.test_dataloader()))[0]
    )


def test_dm_unexposed_kwargs(data_dir):
    classes = 3
    dm = RSVQAHRDataModule(data_dir=data_dir, dataset_kwargs={"classes": classes})
    dm.setup(None)
    assert (
        dm.train_ds.classes == classes
    ), f"There should only be {classes} classes, but there are {dm.train_ds.classes}"
