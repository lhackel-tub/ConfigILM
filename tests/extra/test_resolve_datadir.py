from pathlib import Path

import pytest

from configilm.extra.BEN_lmdb_utils import resolve_data_dir as resolve_ben_data_dir
from configilm.extra.DataSets.COCOQA_DataSet import (
    resolve_data_dir as resolve_cocoqa_data_dir,
)
from configilm.extra.DataSets.HRVQA_DataSet import (
    resolve_data_dir as resolve_hrvqa_data_dir,
)

expected_paths = {
    "coco-qa": str(
        Path(__file__)
        .parent.parent.parent.joinpath("configilm")
        .joinpath("extra")
        .joinpath("mock_data")
        .joinpath("COCO-QA")
        .resolve(True)
    ),
    "bigearthnet": str(
        Path(__file__)
        .parent.parent.parent.joinpath("configilm")
        .joinpath("extra")
        .joinpath("mock_data")
        .resolve(True)
    ),
    "hrvqa": str(
        Path(__file__)
        .parent.parent.parent.joinpath("configilm")
        .joinpath("extra")
        .joinpath("mock_data")
        .joinpath("HRVQA")
        .resolve(True)
    ),
}


@pytest.mark.parametrize("data_dir", ["none", "None", None])
def test_mockdir_cocoqa_none(data_dir):
    with pytest.raises(AssertionError):
        _ = resolve_cocoqa_data_dir(data_dir, allow_mock=False, force_mock=False)


@pytest.mark.parametrize("data_dir", ["none", "None", None])
def test_mockdir_cocoqa_allowed(data_dir):
    expected_path = expected_paths["coco-qa"]
    res = resolve_cocoqa_data_dir(data_dir, allow_mock=True, force_mock=False)
    assert res == expected_path


@pytest.mark.parametrize("data_dir", ["none", "None", None])
def test_mockdir_cocoqa_forced(data_dir):
    expected_path = expected_paths["coco-qa"]
    res = resolve_cocoqa_data_dir(data_dir, allow_mock=True, force_mock=True)
    assert res == expected_path


@pytest.mark.parametrize("data_dir", ["none", "None", None])
def test_mockdir_ben_none(data_dir):
    with pytest.raises(AssertionError):
        _ = resolve_ben_data_dir(data_dir, allow_mock=False, force_mock=False)


@pytest.mark.parametrize("data_dir", ["none", "None", None])
def test_mockdir_ben_allowed(data_dir):
    expected_path = expected_paths["bigearthnet"]
    res = resolve_ben_data_dir(data_dir, allow_mock=True, force_mock=False)
    assert res == expected_path


@pytest.mark.parametrize("data_dir", ["none", "None", None])
def test_mockdir_ben_forced(data_dir):
    expected_path = expected_paths["bigearthnet"]
    res = resolve_ben_data_dir(data_dir, allow_mock=True, force_mock=True)
    assert res == expected_path


@pytest.mark.parametrize("data_dir", ["none", "None", None])
def test_mockdir_hrvqa_none(data_dir):
    with pytest.raises(AssertionError):
        _ = resolve_hrvqa_data_dir(data_dir, allow_mock=False, force_mock=False)


@pytest.mark.parametrize("data_dir", ["none", "None", None])
def test_mockdir_hrvqa_allowed(data_dir):
    expected_path = expected_paths["hrvqa"]
    res = resolve_hrvqa_data_dir(data_dir, allow_mock=True, force_mock=False)
    assert res == expected_path


@pytest.mark.parametrize("data_dir", ["none", "None", None])
def test_mockdir_hrvqa_forced(data_dir):
    expected_path = expected_paths["hrvqa"]
    res = resolve_hrvqa_data_dir(data_dir, allow_mock=True, force_mock=True)
    assert res == expected_path
