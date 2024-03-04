import itertools
from typing import Sequence
from typing import Tuple

import pytest
import torch

from . import test_data_common
from configilm.extra.DataModules.HRVQA_DataModule import HRVQADataModule
from configilm.extra.DataSets.HRVQA_DataSet import HRVQADataSet
from configilm.extra.DataSets.HRVQA_DataSet import resolve_data_dir


@pytest.fixture
def data_dirs():
    return resolve_data_dir(None, allow_mock=True, force_mock=True)


# this dataset does not support test split natively
# instead the val split is reused for test
# for "x-div" the val split is split into two randomly based on a seed
dataset_splits = ["train", "val", "val-div", "test-div", "test"]
div_seeds = ["repeat", 0, 1, 42, 2023]
div_part = [0.1, 0.3, 0.3141592, 0.66, 0.7, 2, 4, 5]
dm_stages = [None, "fit", "test"]
stage_with_seeds = list(itertools.product(dm_stages, div_seeds))

class_number = [10, 100, 250, 1000, 1234]
img_sizes = [60, 120, 128, 144, 256, 1024]
channels_pass = [3]  # accepted channel configs
channels_fail = [1, 2, 4, 0, -1, 10, 12]  # not accepted configs
img_shapes_pass = [(c, hw, hw) for c in channels_pass for hw in img_sizes]
img_shapes_fail = [(c, hw, hw) for c in channels_fail for hw in img_sizes]

no_qa_full_val = 5  # number of samples in full val set


@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_ds_default(data_dirs):
    ds = HRVQADataSet(data_dirs=data_dirs)

    test_data_common.dataset_ok(
        dataset=ds,
        expected_image_shape=(3, 1024, 1024),
        expected_length=None,
        classes=1_000,
        expected_question_length=64,
    )


@pytest.mark.parametrize("split, classes", [(s, c) for s in dataset_splits for c in class_number])
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_3c_dataset_splits_classes(data_dirs, split: str, classes: int):
    img_size = (3, 128, 128)
    seq_length = 32

    ds = HRVQADataSet(
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


@pytest.mark.parametrize(
    "split, seed, div_part",
    [(s, se, d) for s in dataset_splits for se in div_seeds for d in div_part],
)
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_3c_dataset_splits_subdiv(data_dirs, split: str, seed, div_part):
    img_size = (3, 128, 128)
    seq_length = 32

    ds = HRVQADataSet(
        data_dirs=data_dirs,
        split=split,
        img_size=img_size,
        div_seed=seed,
        split_size=div_part,
        seq_length=seq_length,
    )

    div_part_i = div_part if isinstance(div_part, int) else int(div_part * no_qa_full_val)
    if split == "test-div":
        div_part_i = no_qa_full_val - div_part_i
    if "-div" not in split or seed == "repeat":
        div_part_i = no_qa_full_val

    test_data_common.dataset_ok(
        dataset=ds,
        expected_image_shape=img_size,
        expected_length=div_part_i,
        classes=1_000,
        expected_question_length=seq_length,
    )


@pytest.mark.parametrize(
    "split, seed, div_part",
    [(s, se, d) for s in ["val-div", "test-div"] for se in div_seeds for d in div_part],
)
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_3c_dataset_splits_subdiv_overlap(data_dirs, split: str, seed: int, div_part):
    img_size = (3, 128, 128)
    seq_length = 32
    inv_split = "val-div" if split == "test-div" else "test-div"

    ds = HRVQADataSet(
        data_dirs=data_dirs,
        split=split,
        img_size=img_size,
        div_seed=seed,
        split_size=div_part,
        seq_length=seq_length,
    )

    ds_i = HRVQADataSet(
        data_dirs=data_dirs,
        split=inv_split,
        img_size=img_size,
        div_seed=seed,
        split_size=div_part,
        seq_length=seq_length,
    )
    q = ds.qa_data
    qi = ds_i.qa_data

    if seed == "repeat":
        # both have to be same
        assert q == qi, "Questions sets are not equal but should be"
    else:
        q_set = set(q)
        qi_set = set(qi)
        assert q_set.intersection(qi_set) == set(), "There is an intersection between the val and test set"
        assert len(q_set.union(qi_set)) == no_qa_full_val, "Val and test set combined do not cover the full set"


@pytest.mark.parametrize("img_size", img_shapes_pass)
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_ds_imgsize_pass(data_dirs, img_size: Tuple[int, int, int]):
    ds = HRVQADataSet(data_dirs=data_dirs, split="val", img_size=img_size, num_classes=1000, seq_length=32)

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
        _ = HRVQADataSet(
            data_dirs=data_dirs,
            split="val",
            img_size=img_size,
            num_classes=1000,
            seq_length=64,
        )


@pytest.mark.parametrize("max_len", [1, 16, 74, 75, None, -1])
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_ds_max_img_idx(data_dirs, max_len: int):
    ds = HRVQADataSet(data_dirs=data_dirs, max_len=max_len)
    expected_len = 10
    len_ds = expected_len if max_len is None or max_len > expected_len or max_len == -1 else max_len
    test_data_common.dataset_ok(
        dataset=ds,
        expected_image_shape=(3, 1024, 1024),
        expected_length=len_ds,
        classes=1000,
        expected_question_length=64,
    )


@pytest.mark.parametrize("idx", [0, 1, 5, 10, 19, 20, 50, 100, 1_000, 10_000, 100_000])
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_ds_img_access_by_index(data_dirs, idx: int):
    ds = HRVQADataSet(data_dirs=data_dirs)
    expected_len = 10
    assert len(ds) == expected_len, f"There should be {expected_len} samples exactly."
    if idx < len(ds):
        _ = ds[idx]
    else:
        with pytest.raises(IndexError):
            _ = ds[idx]


@pytest.mark.parametrize("max_len", [76, 20000, 100_000, 10_000_000])
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_ds_max_img_idx_too_large(data_dirs, max_len: int):
    ds = HRVQADataSet(data_dirs=data_dirs, max_len=max_len)
    assert len(ds) < max_len


@pytest.mark.parametrize("classes", [1, 5, 10, 50, 100, 1000, 2345, 5000, 15000, 25000])
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_ds_classes(data_dirs, classes: int):
    ds = HRVQADataSet(data_dirs=data_dirs, num_classes=classes, split="train")
    test_data_common._assert_classes_beyond_border_invalid(ds, classes, 4)


@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_ben_dm_lightning(data_dirs):
    dm = HRVQADataModule(data_dirs=data_dirs)
    test_data_common._assert_dm_correct_lightning_version(dm)


@pytest.mark.parametrize("split", ["train", "val", "test", None])
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_dm_default(data_dirs, split: str):
    dm = HRVQADataModule(data_dirs=data_dirs)
    split2stage = {"train": "fit", "val": "fit", "test": "test", None: None}
    if split2stage[split] in ["test"]:
        with pytest.raises(NotImplementedError):
            dm.setup(stage=split2stage[split])
        # overwrite default
        dm = HRVQADataModule(data_dirs=data_dirs, test_splitting_seed=0, test_splitting_size=0.5)
        dm.setup(stage=split2stage[split])
    else:
        dm.setup(stage=split2stage[split])
    dm.prepare_data()
    if split in ["train", "val"]:
        test_data_common.dataset_ok(
            dm.train_ds,
            expected_image_shape=(3, 1024, 1024),
            expected_length=None,
            classes=1_000,
            expected_question_length=64,
        )
        test_data_common.dataset_ok(
            dm.val_ds,
            expected_image_shape=(3, 1024, 1024),
            expected_length=None,
            classes=1000,
            expected_question_length=64,
        )
        assert dm.test_ds is None, "Dataset for test should be None"
    elif split in ["test"]:
        assert dm.train_ds is None, "Dataset for train should be None"
        assert dm.val_ds is None, "Dataset for val should be None"
        test_data_common.dataset_ok(
            dm.test_ds,
            expected_image_shape=(3, 1024, 1024),
            expected_length=None,
            classes=1000,
            expected_question_length=64,
        )
    elif split is None:
        for ds in [dm.train_ds, dm.val_ds]:
            test_data_common.dataset_ok(
                ds,
                expected_image_shape=(3, 1024, 1024),
                expected_length=None,
                classes=1000,
                expected_question_length=64,
            )
        assert dm.test_ds is None, "Dataset for test should be None as we are not " "splitting"
    else:
        ValueError(f"split {split} unknown")


@pytest.mark.parametrize("bs", [1, 2, 4, 8, 16, 32, 13, 27])
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_dm_dataloaders_bs(data_dirs, bs: int):
    dm = HRVQADataModule(data_dirs=data_dirs, batch_size=bs, num_workers_dataloader=0, pin_memory=False)

    def _dataloaders_ok(
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

    _dataloaders_ok(
        dm,
        expected_image_shape=(bs, 3, 1024, 1024),
        expected_question_length=64,
        classes=1000,
    )


@pytest.mark.parametrize("img_size", [[1], [1, 2], [1, 2, 3, 4]])
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_dm_dataloaders_img_size(data_dirs, img_size):
    with (pytest.raises(AssertionError)):
        _ = HRVQADataModule(
            data_dirs=data_dirs,
            img_size=img_size,
            num_workers_dataloader=0,
            pin_memory=False,
        )


@pytest.mark.parametrize(
    "stage, seed, div",
    [(st, se, d) for st in dm_stages for se in div_seeds + [None] for d in div_part],
)
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_dm_dataloaders_with_splitting(data_dirs, stage, seed, div):
    dm = HRVQADataModule(
        data_dirs=data_dirs,
        test_splitting_seed=seed,
        test_splitting_size=div,
        num_workers_dataloader=0,
        pin_memory=False,
    )
    if stage == "test" and seed is None:
        with pytest.raises(NotImplementedError):
            dm.setup(stage)
        return

    dm.setup(stage)

    if dm.train_ds is not None:
        assert dm.train_dataloader() is not None, "Train Dataloader should not be None"

    if dm.val_ds is not None:
        assert dm.val_dataloader() is not None, "Val Dataloader should not be None"

    if dm.test_ds is not None:
        if seed is None:
            assert dm.test_dataloader() is None, "Test Dataloader should be None"
        else:
            assert dm.test_dataloader() is not None, "Test Dataloader should not be " "None"


@pytest.mark.filterwarnings('ignore:Shuffle was set to False.')
@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_dm_shuffle_false(data_dirs):
    dm = HRVQADataModule(data_dirs=data_dirs, shuffle=False, num_workers_dataloader=0, pin_memory=False)
    dm.setup(None)
    # should not be equal due to transforms being random!
    assert not torch.equal(
        next(iter(dm.train_dataloader()))[0],
        next(iter(dm.train_dataloader()))[0],
    )
    assert torch.equal(next(iter(dm.val_dataloader()))[0], next(iter(dm.val_dataloader()))[0])


@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_dm_shuffle_none(data_dirs):
    dm = HRVQADataModule(data_dirs=data_dirs, shuffle=None, num_workers_dataloader=0, pin_memory=False)
    dm.setup(None)
    assert not torch.equal(
        next(iter(dm.train_dataloader()))[0],
        next(iter(dm.train_dataloader()))[0],
    )
    assert torch.equal(next(iter(dm.val_dataloader()))[0], next(iter(dm.val_dataloader()))[0])


@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
@pytest.mark.filterwarnings('ignore:Shuffle was set to True.')
def test_dm_shuffle_true(data_dirs):
    dm = HRVQADataModule(data_dirs=data_dirs, shuffle=True, num_workers_dataloader=0, pin_memory=False)
    dm.setup(None)
    assert not torch.equal(
        next(iter(dm.train_dataloader()))[0],
        next(iter(dm.train_dataloader()))[0],
    )
    assert not torch.equal(next(iter(dm.val_dataloader()))[0], next(iter(dm.val_dataloader()))[0])


@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_dm_test_stage_setup(data_dirs):
    dm = HRVQADataModule(data_dirs=data_dirs, num_workers_dataloader=0, pin_memory=False)
    with pytest.raises(NotImplementedError):
        dm.setup("test")


@pytest.mark.filterwarnings('ignore:No tokenizer was provided,')
def test_dm_predict_stage_setup(data_dirs):
    dm = HRVQADataModule(data_dirs=data_dirs, num_workers_dataloader=0, pin_memory=False)
    with pytest.raises(NotImplementedError):
        dm.setup("predict")
