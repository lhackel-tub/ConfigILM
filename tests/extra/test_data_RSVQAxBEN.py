import json
import warnings

import pytest

with warnings.catch_warnings():
    warnings.filterwarnings(
        action="ignore", category=DeprecationWarning, message=".*distutils.*"
    )
    from configvlm.extra.RSVQAxBEN_DataModule_LMDB_Encoder import (
        RSVQAxBENDataSet,
        RSVQAxBENDataModule,
    )

from configvlm.extra.BEN_lmdb_utils import resolve_ben_data_dir
from typing import Sequence, Union


@pytest.fixture
def data_dir():
    return resolve_ben_data_dir(None, allow_mock=True)


dataset_params = ["train", "val", "test", None]

class_number = [10, 100, 250, 1000, 1234]
img_sizes = [60, 120, 128, 144, 256]
channels_pass = [2, 3, 4, 10, 12]  # accepted channel configs
channels_fail = [5, 1, 0, -1, 13]  # not accepted configs
img_shapes_pass = [(c, hw, hw) for c in channels_pass for hw in img_sizes]
img_shapes_fail = [(c, hw, hw) for c in channels_fail for hw in img_sizes]
max_img_idxs = [0, 1, 100, 10000]
max_img_idxs_too_large = [600_000, 1_000_000]

mock_data_dict = {
    i: {
        "type": "LC",
        "question": "What is the question?",
        "answer": f"{i % 2345}",
        "S2_name": f"S2A_MSIL2A_20170613T101031_0_{45 + i % 4}",
    }
    for i in range(15000)
}
redirected_json_load = json.load


def dataset_ok(
    dataset: RSVQAxBENDataSet,
    expected_image_shape: Sequence,
    expected_question_length: int,
    expected_length: Union[int, None],
    classes: int,
):
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
    dm: RSVQAxBENDataModule,
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
            assert q.shape == (
                expected_image_shape[0],
                expected_question_length,
            )
            assert a.shape == (expected_image_shape[0], classes)


def wrap_json_load(
    fp,
    *,
    cls=None,
    object_hook=None,
    parse_float=None,
    parse_int=None,
    parse_constant=None,
    object_pairs_hook=None,
    **kw,
):
    if (
        "RSVQAxBEN_QA_train.json" in fp.name
        or "RSVQAxBEN_QA_val.json" in fp.name
        or "RSVQAxBEN_QA_test.json" in fp.name
    ):
        # this call to load a json file will be redirected
        return mock_data_dict
    else:
        return redirected_json_load(
            fp=fp,
            cls=cls,
            object_hook=object_hook,
            parse_float=parse_float,
            parse_int=parse_int,
            parse_constant=parse_constant,
            object_pairs_hook=object_pairs_hook,
            **kw,
        )


@pytest.mark.parametrize(
    "split, classes", [(s, c) for s in dataset_params for c in class_number]
)
def test_4c_ben_dataset_splits(data_dir, split: str, classes: int):
    img_size = (4, 120, 120)
    json.load = wrap_json_load

    ds = RSVQAxBENDataSet(
        root_dir=data_dir, split=split, img_size=img_size, classes=classes
    )

    dataset_ok(
        dataset=ds,
        expected_image_shape=img_size,
        expected_length=None,
        classes=classes,
        expected_question_length=32,
    )
