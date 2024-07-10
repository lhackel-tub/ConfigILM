import warnings
from functools import partial

import pytest

with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", category=DeprecationWarning, message=".*distutils.*")
    from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet
    from configilm.extra.DataModules.BENv2_DataModule import BENv2DataModule
    from configilm.extra.BENv2_utils import BENv2LDMBReader, STANDARD_BANDS, stack_and_interpolate

from . import test_data_common
from configilm.extra.BENv2_utils import resolve_data_dir
from typing import Sequence, Union
import torch


@pytest.fixture
def data_dirs():
    return resolve_data_dir(None, force_mock=True)


dataset_params = ["train", "validation", "test", None]

img_sizes = [60, 120, 128, 144, 256]
channels_pass = [2, 3, 4, 10, 12]  # accepted channel configs
channels_fail = [5, 1, 0, -1, 13]  # not accepted configs
img_shapes_pass = [(c, hw, hw) for c in channels_pass for hw in img_sizes]
img_shapes_fail = [(c, hw, hw) for c in channels_fail for hw in img_sizes]
max_lens = [0, 1, 10, 20, None, -1]
max_lens_too_large = [600_000, 1_000_000]


def dataset_ok(
    dataset: Union[BENv2DataSet, None],
    expected_image_shape: Sequence,
    expected_length: Union[int, None],
):
    # In principal dataset may be not set in data modules, but mypy requires this
    # notation to be happy.
    # Check that this is not the case here
    assert dataset is not None
    if expected_length is not None:
        assert len(dataset) == expected_length

    if len(dataset) > 0:
        for i in [0, 100, 2000, 5000, 10000]:
            i = i % len(dataset)
            sample = dataset[i]
            assert len(sample) == 2
            v, lbl = sample
            assert v.shape == expected_image_shape
            assert list(lbl.size()) == [19]


def dataloaders_ok(dm: BENv2DataModule, expected_image_shape: Sequence):
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
            v, lbl = batch
            assert v.shape == expected_image_shape
            assert lbl.shape == (expected_image_shape[0], 19)


def test_ben_4c_dataset_patchname_getter(data_dirs):
    ds = BENv2DataSet(data_dirs=data_dirs, split="validation", return_extras=True)
    assert (
        ds.get_patchname_from_index(0) == "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_33_69"
    ), "Patch name does not match"
    assert ds.get_patchname_from_index(1_000_000) is None, "Patch index OOB should not work"
    assert (
        ds.get_index_from_patchname("S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_33_69") == 0
    ), "Index name does not match"
    assert ds.get_index_from_patchname("abc") is None, "None existing name does not work"


def test_ben_dataset_with_cloud(data_dirs):
    ds = BENv2DataSet(data_dirs=data_dirs, split="validation", include_cloudy=True, include_snowy=False)
    assert len(ds) == 6 + 1, "Only 7 patches should have been returned (6 regular and 1 cloudy)"


def test_ben_dataset_with_snow(data_dirs):
    ds = BENv2DataSet(data_dirs=data_dirs, split="validation", include_cloudy=False, include_snowy=True)
    assert len(ds) == 6 + 1, "Only 7 patches should have been returned (6 regular and 1 snowy)"


def test_ben_dataset_with_cloud_snow(data_dirs):
    ds = BENv2DataSet(data_dirs=data_dirs, split="validation", include_cloudy=True, include_snowy=True)
    assert len(ds) == 6 + 2, "Only 8 patches should have been returned (6 regular and 1 snowy and 1 cloudy)"


def test_ben_4c_dataset_patchname(data_dirs):
    ds = BENv2DataSet(data_dirs=data_dirs, split="validation")
    assert len(ds[0]) == 2, "Only two items should have been returned"
    ds = BENv2DataSet(data_dirs=data_dirs, split="validation", return_extras=True)
    assert len(ds[0]) == 3, "Three items should have been returned"
    assert type(ds[0][2]) == str, "Third item should be a string"
    assert ds[0][2] == "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_33_69", "Patch name does not match"


@pytest.mark.parametrize("split", dataset_params)
def test_ben_4c_dataset_splits(data_dirs, split: str):
    img_size = (4, 120, 120)
    ds = BENv2DataSet(data_dirs=data_dirs, split=split, img_size=img_size)

    dataset_ok(dataset=ds, expected_image_shape=img_size, expected_length=None)


@pytest.mark.parametrize("img_size", img_shapes_pass)
def test_ben_val_dataset_sizes(data_dirs, img_size: tuple):
    ds = BENv2DataSet(data_dirs=data_dirs, split="validation", img_size=img_size)

    dataset_ok(dataset=ds, expected_image_shape=img_size, expected_length=None)


@pytest.mark.parametrize("img_size", img_shapes_pass)
def test_ben_val_dataset_sizes_rescale(data_dirs, img_size: tuple):
    new_size = (img_size[0], 100, 100)
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize(new_size[1:], antialias=True),
        ]
    )
    ds = BENv2DataSet(data_dirs=data_dirs, split="validation", img_size=img_size, transform=transform)

    dataset_ok(dataset=ds, expected_length=None, expected_image_shape=new_size)


@pytest.mark.parametrize("img_size", img_shapes_fail)
def test_ben_val_dataset_sizes_fail(data_dirs, img_size: tuple):
    with pytest.raises(AssertionError):
        _ = BENv2DataSet(data_dirs=data_dirs, split="validation", img_size=img_size)


def test_ben_fail_image_retrieve(data_dirs):
    ds = BENv2DataSet(data_dirs=data_dirs, split="validation", img_size=(3, 120, 120))
    assert ds[0][0].shape == (3, 120, 120)
    # overwrite this lookup
    # would have been a subscribable lookup
    ds.BENv2Loader = {ds.patches[0]: (None, None)}
    with pytest.raises(ValueError):
        _ = ds[0]


@pytest.mark.parametrize("max_len", max_lens)
def test_ben_max_index(data_dirs, max_len: int):
    is_mocked = "mock" in str(data_dirs["metadata_parquet"])
    expected_len = 6 if is_mocked else 123_723
    if max_len is not None and expected_len > max_len and max_len != -1:
        expected_len = max_len

    ds = BENv2DataSet(
        data_dirs=data_dirs,
        split="validation",
        img_size=(3, 120, 120),
        max_len=max_len,
    )

    dataset_ok(
        dataset=ds,
        expected_image_shape=(3, 120, 120),
        expected_length=expected_len,
    )


@pytest.mark.parametrize("max_len", max_lens_too_large)
def test_ben_max_index_too_large(data_dirs, max_len: int):
    ds = BENv2DataSet(
        data_dirs=data_dirs,
        split="validation",
        img_size=(3, 120, 120),
        max_len=max_len,
    )
    assert len(ds) < max_len


def test_ben_prefilter(data_dirs):
    ds = BENv2DataSet(
        data_dirs=data_dirs,
        split="validation",
        img_size=(3, 120, 120),
        patch_prefilter=lambda x: "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_33_" in x,
    )
    assert len(ds) == 2, "There should be exactly 2 patches with this prefix in the mock data"


def test_ben_dm_lightning(data_dirs):
    dm = BENv2DataModule(data_dirs=data_dirs)
    test_data_common._assert_dm_correct_lightning_version(dm)


@pytest.mark.parametrize("split", dataset_params)
def test_ben_dm_default(data_dirs, split: str):
    dm = BENv2DataModule(data_dirs=data_dirs)
    split2stage = {"train": "fit", "validation": "fit", "test": "test", None: None}
    dm.setup(stage=split2stage[split])
    dm.prepare_data()
    if split in ["train", "validation"]:
        dataset_ok(
            dm.train_ds,
            expected_image_shape=(3, 120, 120),
            expected_length=None,
        )
        dataset_ok(dm.val_ds, expected_image_shape=(3, 120, 120), expected_length=None)
        assert dm.test_ds is None
    elif split == "test":
        dataset_ok(
            dm.test_ds,
            expected_image_shape=(3, 120, 120),
            expected_length=None,
        )
        assert dm.train_ds is None
        assert dm.val_ds is None
    elif split is None:
        for ds in [dm.train_ds, dm.val_ds, dm.test_ds]:
            dataset_ok(ds, expected_image_shape=(3, 120, 120), expected_length=None)


@pytest.mark.parametrize("img_size", [[1], [1, 2], [1, 2, 3, 4]])
def test_ben_dm_wrong_imagesize(data_dirs, img_size):
    with pytest.raises(AssertionError):
        _ = BENv2DataModule(data_dirs, img_size=img_size, num_workers_dataloader=0, pin_memory=False)


@pytest.mark.parametrize("bs", [1, 2, 4, 8, 16, 32, 13, 27])
def test_ben_dm_dataloaders(data_dirs, bs):
    dm = BENv2DataModule(data_dirs=data_dirs, batch_size=bs, num_workers_dataloader=0, pin_memory=False)
    dataloaders_ok(dm, expected_image_shape=(bs, 3, 120, 120))


@pytest.mark.filterwarnings("ignore:Shuffle was set to False.")
def test_ben_shuffle_false(data_dirs):
    dm = BENv2DataModule(data_dirs=data_dirs, shuffle=False, num_workers_dataloader=0, pin_memory=False)
    dm.setup(None)
    # should not be equal due to transforms being random!
    assert not torch.equal(
        next(iter(dm.train_dataloader()))[0],
        next(iter(dm.train_dataloader()))[0],
    )
    assert torch.equal(next(iter(dm.val_dataloader()))[0], next(iter(dm.val_dataloader()))[0])
    assert torch.equal(next(iter(dm.test_dataloader()))[0], next(iter(dm.test_dataloader()))[0])


def test_ben_shuffle_none(data_dirs):
    dm = BENv2DataModule(data_dirs=data_dirs, shuffle=None, num_workers_dataloader=0, pin_memory=False)
    dm.setup(None)
    assert not torch.equal(
        next(iter(dm.train_dataloader()))[0],
        next(iter(dm.train_dataloader()))[0],
    )
    assert torch.equal(next(iter(dm.val_dataloader()))[0], next(iter(dm.val_dataloader()))[0])
    assert torch.equal(next(iter(dm.test_dataloader()))[0], next(iter(dm.test_dataloader()))[0])


@pytest.mark.filterwarnings("ignore:Shuffle was set to True.")
def test_ben_shuffle_true(data_dirs):
    dm = BENv2DataModule(data_dirs=data_dirs, shuffle=True, num_workers_dataloader=0, pin_memory=False)
    dm.setup(None)
    assert not torch.equal(
        next(iter(dm.train_dataloader()))[0],
        next(iter(dm.train_dataloader()))[0],
    )
    assert not torch.equal(next(iter(dm.val_dataloader()))[0], next(iter(dm.val_dataloader()))[0])
    assert not torch.equal(next(iter(dm.test_dataloader()))[0], next(iter(dm.test_dataloader()))[0])


def test_dm_unexposed_kwargs(data_dirs):
    dm = BENv2DataModule(
        data_dirs=data_dirs,
        num_workers_dataloader=0,
        pin_memory=False,
    )
    dm.setup(None)
    # changing the param here
    dm.train_ds.return_extras = True
    assert len(dm.train_ds[0]) == 3, f"This change should have returned 3 items but does {len(dm.train_ds[0])}"


def test_utils_keys(data_dirs):
    reader = BENv2LDMBReader(
        image_lmdb_file=data_dirs["images_lmdb"],
        metadata_file=data_dirs["metadata_parquet"],
        metadata_snow_cloud_file=data_dirs["metadata_snow_cloud_parquet"],
    )

    assert len(reader.keys()) == 8 * 3 * 2, "Keys should be 9*3*2 (9 patches, 3 splits, 2 satellites)"
    assert len(reader.S1_keys()) == 8 * 3, "S1 keys should be 9*3 (9 patches, 3 splits)"
    assert len(reader.S2_keys()) == 8 * 3, "S2 keys should be 9*3 (9 patches, 3 splits)"


@pytest.mark.parametrize("bands", list(STANDARD_BANDS.keys()) + STANDARD_BANDS[12] + list(STANDARD_BANDS.values()))
def test_band_selection(data_dirs, bands):
    reader = BENv2LDMBReader(
        image_lmdb_file=data_dirs["images_lmdb"],
        metadata_file=data_dirs["metadata_parquet"],
        metadata_snow_cloud_file=data_dirs["metadata_snow_cloud_parquet"],
        bands=bands,
        process_bands_fn=partial(stack_and_interpolate, img_size=120, upsample_mode="nearest"),
    )
    x, _ = reader[list(reader.S2_keys())[0]]
    expected_len = (
        len(bands)
        if isinstance(bands, list)
        else bands
        if isinstance(bands, int)
        else len(STANDARD_BANDS[bands])
        if bands in STANDARD_BANDS
        else 1
    )
    assert x.shape[0] == expected_len, "Number of bands should match"


def test_invalid_band_type(data_dirs):
    with pytest.raises(ValueError):
        _ = BENv2LDMBReader(
            image_lmdb_file=data_dirs["images_lmdb"],
            metadata_file=data_dirs["metadata_parquet"],
            metadata_snow_cloud_file=data_dirs["metadata_snow_cloud_parquet"],
            bands=3.2,  # noqa
            process_bands_fn=partial(stack_and_interpolate, img_size=120, upsample_mode="nearest"),
        )


def test_train_transform_settable(data_dirs):
    from configilm.extra._defaults import default_train_transform
    from configilm.extra._defaults import default_train_transform_with_noise

    dm = BENv2DataModule(
        data_dirs=data_dirs,
        img_size=(3, 120, 120),
        train_transforms=default_train_transform(120, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    )
    dataloaders_ok(dm, expected_image_shape=(3, 120, 120))

    dm = BENv2DataModule(
        data_dirs=data_dirs,
        img_size=(3, 120, 120),
        train_transforms=default_train_transform_with_noise(120, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        shuffle=False,
    )
    dataloaders_ok(dm, expected_image_shape=(3, 120, 120))


def test_eval_transform_settable(data_dirs):
    from configilm.extra._defaults import default_transform
    from configilm.extra._defaults import default_train_transform_with_noise

    dm = BENv2DataModule(
        data_dirs=data_dirs,
        img_size=(3, 120, 120),
        eval_transforms=default_transform(120, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    )
    dataloaders_ok(dm, expected_image_shape=(3, 120, 120))

    dm = BENv2DataModule(
        data_dirs=data_dirs,
        img_size=(3, 120, 120),
        eval_transforms=default_train_transform_with_noise(120, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        shuffle=False,
    )
    dataloaders_ok(dm, expected_image_shape=(3, 120, 120))
