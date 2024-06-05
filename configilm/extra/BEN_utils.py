"""
Helpful functions when working with lmdb file which contains BigEarthNet as encoded
binary patches.
Functions include reading data and extracting specific band combinations and their
properties.
"""

__author__ = "Leonard Hackel"
__email__ = "l.hackel@tu-berlin.de"

from pathlib import Path
from typing import Iterable, Union, Optional, Sequence, Mapping

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from bigearthnet_common.base import ben_19_labels_to_multi_hot
from bigearthnet_common.constants import BAND_STATS_S1, BAND_STATS_S2
from bigearthnet_patch_interface.merged_interface import BigEarthNet_S1_S2_Patch

from configilm.extra.data_dir import resolve_data_dir_for_ds

BAND_COMBINATION_PREDEFINTIONS = {
    "S1": ["VH", "VV"],
    "S2": ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"],
    "10m20m": [
        "B02",
        "B03",
        "B04",
        "B08",
        "B05",
        "B06",
        "B07",
        "B11",
        "B12",
        "B8A",
        "VH",
        "VV",
    ],
    "RGB": ["B04", "B03", "B02"],
    "RGB-IR": ["B04", "B03", "B02", "B08"],
    2: ["VH", "VV"],
    10: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"],
    12: [
        "B02",
        "B03",
        "B04",
        "B08",
        "B05",
        "B06",
        "B07",
        "B11",
        "B12",
        "B8A",
        "VH",
        "VV",
    ],
    3: ["B04", "B03", "B02"],
    4: ["B04", "B03", "B02", "B08"],
}

_BANDNAME_TO_BEN_INTERFACE_PROPERTY = {
    "B01": ["s2_patch", "band01"],
    "B02": ["s2_patch", "band02"],
    "B03": ["s2_patch", "band03"],
    "B04": ["s2_patch", "band04"],
    "B05": ["s2_patch", "band05"],
    "B06": ["s2_patch", "band06"],
    "B07": ["s2_patch", "band07"],
    "B08": ["s2_patch", "band08"],
    "B09": ["s2_patch", "band09"],
    "B11": ["s2_patch", "band11"],
    "B12": ["s2_patch", "band12"],
    "B8A": ["s2_patch", "band8A"],
    "VH": ["s1_patch", "bandVH"],
    "VV": ["s1_patch", "bandVV"],
}

valid_labels_classification = [
    "Agro-forestry areas",  # 0
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",  # 5
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas " "of natural vegetation",  # 10
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",  # 15
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
]
valid_labels_classification = [x.lower() for x in valid_labels_classification]


def ben19_list_to_onehot(lst):
    """
    Converts a list of [BEN19 Labels] to a one hot vector.
    Elements in the vector are sorted alphabetically. See
    `valid_labels_classification()` for details.
    """
    # all the valid ones and one additional if none are valid
    res = torch.tensor(ben_19_labels_to_multi_hot(lst, lex_sorted=True))
    assert sum(res) >= 1, "Result Tensor is all Zeros - this is not allowed"
    return res


def _resolve_band_combi(bands: Union[Iterable, str, int]) -> list:
    """
    Resolves a predefined combination of bands or a list of bands into a list of
    individual bands and checks if all bands contained are actual S1/S2 band names.

    :param bands: a combination of bands as defined in BAND_COMBINATION_PREDEFINTIONS
        or a list of bands
    :return: a list of bands contained in the predefinition
    """
    if isinstance(bands, str) or isinstance(bands, int):
        assert bands in BAND_COMBINATION_PREDEFINTIONS.keys(), (
            "Band combination unknown, "
            f"please use a list of strings or one of "
            f"{BAND_COMBINATION_PREDEFINTIONS.keys()}"
        )
        bands = BAND_COMBINATION_PREDEFINTIONS[bands]
    for band in bands:
        assert band in _BANDNAME_TO_BEN_INTERFACE_PROPERTY.keys(), f"Band '{band}' unknown"
    return list(bands)


def band_combi_to_mean_std(bands: Union[Iterable, str, int]):
    """
    Retrievs the mean and standard deviation for a given BigEarthNet
    BAND_COMBINATION_PREDEFINTIONS or list of bands.

    :param bands: combination of bands as defined in BAND_COMBINATION_PREDEFINTIONS
        or a list of bandsmb
    :return: mean and standard deviation for the given combination in same order
    """
    bands = _resolve_band_combi(bands)
    S1_bands = ["VH", "VV", "VV/VH"]
    ben_channel_mean = [BAND_STATS_S1["mean"][x] if x in S1_bands else BAND_STATS_S2["mean"][x] for x in bands]
    ben_channel_std = [BAND_STATS_S1["std"][x] if x in S1_bands else BAND_STATS_S2["std"][x] for x in bands]
    return ben_channel_mean, ben_channel_std


def resolve_data_dir(
    data_dir: Optional[Mapping[str, Path]], allow_mock: bool = False, force_mock: bool = False
) -> Mapping[str, Path]:
    """
    Helper function that tries to resolve the correct directory.

    :param data_dir: current path that is suggested
    :param allow_mock: allows mock data path to be returned
    :param force_mock: only mock data path will be returned. Useful for debugging with
        small data
    :return: a valid dir to the dataset if data_dir was none, otherwise data_dir
    """
    return resolve_data_dir_for_ds("benv1", data_dir, allow_mock=allow_mock, force_mock=force_mock)


def read_ben_from_lmdb(env, key):
    """
    Reads a BigEarthNet_S1_S2_Patch from a lmdb environment and decodes it.

    :param env: lmdb environment containing encoded BigEarthNet_S1_S2_Patch
    :param key: patch name as defined in BigEarthNet as string
    :return: Decoded BigEarthNet_S1_S2_Patch
    :raises: AssertionError if key not in database
    """
    bin_key = str(key).encode()
    with env.begin(write=False) as txn:
        binary_patch_data = txn.get(bin_key)
    assert binary_patch_data is not None, f"Patch {key} unknown"
    ben_patch = BigEarthNet_S1_S2_Patch.loads(binary_patch_data)
    return ben_patch


class BENv1LMDBReader:
    def __init__(
        self,
        lmdb_dir: Union[str, Path],
        image_size: Sequence[int],
        bands: Union[Iterable, str, int],
        label_type: str,
    ):
        """
        Initialize a BigEarthNet v1.0 Reader object that reads from a lmdb encoded file.

        :param lmdb_dir: base path that contains a data.mdb and lock.mdb
        :param image_size: final size of the image that it is interpolated to
        :param bands: bands to use for stacking e.g. ["B08", "B03", "VV"] or a string of
            BAND_COMBINATION_PREDEFINTIONS
        :param label_type: "new" or "old" depending on if the old labels (43) or new
            ones (19) should be returned
        """
        assert len(image_size) == 3, f"image_size has to have 3 dimensions (CxHxW) but is {image_size}"
        self.image_size = image_size
        self.bands = _resolve_band_combi(bands)
        assert len(self.bands) == self.image_size[0], (
            f"Number of channels in image_size ({self.image_size[0]}) does not match "
            f"number of bands selected ({len(self.bands)})"
        )

        self.lmdb_dir = lmdb_dir
        self.env = None
        self.label_type = label_type
        self.mean, self.std = band_combi_to_mean_std(self.bands)

    def __getitem__(self, item: str):
        """
        Reads from lmdb file and returns image interpolated to specified size as well as
        the labels
        :param item: name of the S2 patch. If only S1 bands are used, still S2 patch
            name has to be provided
        :return: tuple (interpolated image, labels)
        """
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_dir),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
            )
        # get pure patch data
        ben_patch = read_ben_from_lmdb(self.env, item)

        # get only selected bands
        img_data = [
            ben_patch.__getattribute__(_BANDNAME_TO_BEN_INTERFACE_PROPERTY[x][0]).__getattribute__(
                _BANDNAME_TO_BEN_INTERFACE_PROPERTY[x][1]
            )
            for x in self.bands
        ]
        # Interpolate each band by itself to correct size, as we cannot stack different
        # sizes. !e also have to unsqueeze twice, once for channel and once for batch
        img_data = [
            F.interpolate(
                torch.Tensor(np.float32(x.data)).unsqueeze(dim=0).unsqueeze(dim=0),
                self.image_size[-2:],
                mode="bicubic",
                align_corners=True,
            )
            for x in img_data
        ]
        img_data = torch.cat(img_data, dim=1).squeeze(dim=0)

        return (
            img_data,
            ben_patch.labels if self.label_type == "old" else ben_patch.new_labels,
        )
