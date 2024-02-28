import warnings
from typing import Optional
from typing import Sequence

import torch

from configilm.extra.DataModules.ClassificationVQADataModule import ClassificationVQADataModule
from configilm.extra.DataSets.ClassificationVQADataset import ClassificationVQADataset


def dataset_ok(
    dataset: Optional[ClassificationVQADataset],
    expected_image_shape: Sequence,
    expected_question_length: int,
    expected_length: Optional[int],
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
            assert tuple(v.shape) == tuple(expected_image_shape)
            assert len(q) == expected_question_length
            assert list(a.size()) == [classes]


def dataloaders_ok(
    dm: ClassificationVQADataModule,
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


def _test_dm_shuffle_false(dm: ClassificationVQADataModule):
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
    assert torch.equal(next(iter(dm.val_dataloader()))[0], next(iter(dm.val_dataloader()))[0])
    assert torch.equal(next(iter(dm.test_dataloader()))[0], next(iter(dm.test_dataloader()))[0])


def _test_dm_shuffle_true(dm: ClassificationVQADataModule):
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
    assert not torch.equal(next(iter(dm.val_dataloader()))[0], next(iter(dm.val_dataloader()))[0])
    assert not torch.equal(next(iter(dm.test_dataloader()))[0], next(iter(dm.test_dataloader()))[0])


def _test_dm_shuffle_none(dm: ClassificationVQADataModule):
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
    assert torch.equal(next(iter(dm.val_dataloader()))[0], next(iter(dm.val_dataloader()))[0])
    assert torch.equal(next(iter(dm.test_dataloader()))[0], next(iter(dm.test_dataloader()))[0])


def _assert_datasets_set_correctly_for_split(
    dm: ClassificationVQADataModule,
    split: str,
    expected_image_shape: Sequence,
    expected_length: Optional[int],
    expected_question_length: int,
    classes: int,
):
    # check that the datamodule has the correct attributes
    assert hasattr(dm, "train_ds"), "DataModule has no train_ds attribute"
    assert hasattr(dm, "val_ds"), "DataModule has no train_ds attribute"
    assert hasattr(dm, "test_ds"), "DataModule has no train_ds attribute"

    if split == "train":
        dataset_ok(
            dm.train_ds,
            expected_image_shape=expected_image_shape,
            expected_length=expected_length,
            classes=classes,
            expected_question_length=expected_question_length,
        )
        # val may be None if no validation set is available or set if it is available
        assert dm.test_ds is None
    elif split == "val":
        # train may be None if no validation set is available or set if it is available
        dataset_ok(
            dm.val_ds,
            expected_image_shape=expected_image_shape,
            expected_length=expected_length,
            classes=classes,
            expected_question_length=expected_question_length,
        )

        assert dm.test_ds is None
    elif split == "test":
        dataset_ok(
            dm.test_ds,
            expected_image_shape=expected_image_shape,
            expected_length=expected_length,
            classes=classes,
            expected_question_length=expected_question_length,
        )
        assert dm.train_ds is None
        assert dm.val_ds is None
    elif split is None:
        for ds in [dm.train_ds, dm.val_ds, dm.test_ds]:
            dataset_ok(
                ds,
                expected_image_shape=expected_image_shape,
                expected_length=expected_length,
                classes=classes,
                expected_question_length=expected_question_length,
            )
    else:
        ValueError(f"split {split} unknown")


def _assert_classes_beyond_border_invalid(ds: ClassificationVQADataset, classes: int, max_classes_mock_set: int):
    assert hasattr(ds, "answers"), "Dataset has no answers attribute"
    assert len(ds.answers) == classes
    if classes <= max_classes_mock_set:
        for i in range(classes):
            assert ds.answers[i] != "INVALID"
    else:
        for i in range(max_classes_mock_set):
            assert ds.answers[i] != "INVALID"
        for i in range(max_classes_mock_set, classes):
            assert ds.answers[i] == "INVALID"


def _assert_dm_correct_lightning_version(dm):
    try:
        import lightning.pytorch

        assert isinstance(dm, lightning.pytorch.LightningDataModule), "DM should be a lightning DataModule"
    except ImportError:
        import pytorch_lightning

        assert isinstance(dm, pytorch_lightning.LightningDataModule), "DM should be a pytorch lightning DataModule"
