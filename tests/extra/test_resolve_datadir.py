import pytest

import configilm.extra.BEN_lmdb_utils as ben_utils
import configilm.extra.DataSets.COCOQA_DataSet as cocoqa
import configilm.extra.DataSets.HRVQA_DataSet as hrvqa
from configilm.extra.data_dir import mock_data_path as expected_paths


@pytest.mark.parametrize("data_dir", [None])
def test_mockdir_cocoqa_none(data_dir):
    with pytest.raises(AssertionError):
        _ = cocoqa.resolve_data_dir(data_dir, allow_mock=False, force_mock=False)


@pytest.mark.parametrize("data_dir", [None])
def test_mockdir_cocoqa_allowed(data_dir):
    expected_path = expected_paths["cocoqa"]
    res = cocoqa.resolve_data_dir(data_dir, allow_mock=True, force_mock=False)
    assert res == expected_path


@pytest.mark.parametrize("data_dir", [None])
def test_mockdir_cocoqa_forced(data_dir):
    expected_path = expected_paths["cocoqa"]
    res = cocoqa.resolve_data_dir(data_dir, allow_mock=True, force_mock=True)
    assert res == expected_path


@pytest.mark.parametrize("data_dir", [None])
def test_mockdir_ben_none(data_dir):
    with pytest.raises(AssertionError):
        _ = ben_utils.resolve_data_dir(data_dir, allow_mock=False, force_mock=False)


@pytest.mark.parametrize("data_dir", [None])
def test_mockdir_ben_allowed(data_dir):
    expected_path = expected_paths["benv1"]
    res = ben_utils.resolve_data_dir(data_dir, allow_mock=True, force_mock=False)
    assert res == expected_path


@pytest.mark.parametrize("data_dir", [None])
def test_mockdir_ben_forced(data_dir):
    expected_path = expected_paths["benv1"]
    res = ben_utils.resolve_data_dir(data_dir, allow_mock=True, force_mock=True)
    assert res == expected_path


@pytest.mark.parametrize("data_dir", [None])
def test_mockdir_hrvqa_none(data_dir):
    with pytest.raises(AssertionError):
        _ = hrvqa.resolve_data_dir(data_dir, allow_mock=False, force_mock=False)


@pytest.mark.parametrize("data_dir", [None])
def test_mockdir_hrvqa_allowed(data_dir):
    expected_path = expected_paths["hrvqa"]
    res = hrvqa.resolve_data_dir(data_dir, allow_mock=True, force_mock=False)
    assert res == expected_path


@pytest.mark.parametrize("data_dir", [None])
def test_mockdir_hrvqa_forced(data_dir):
    expected_path = expected_paths["hrvqa"]
    res = hrvqa.resolve_data_dir(data_dir, allow_mock=True, force_mock=True)
    assert res == expected_path
