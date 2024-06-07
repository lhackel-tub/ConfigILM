from pathlib import Path
from time import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Union

import lmdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.numpy import load as safetensor_load

from configilm.extra.data_dir import resolve_data_dir_for_ds

_s1_bandnames = ["VH", "VV"]
_s2_bandnames = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"]
_all_bandnames = _s2_bandnames + _s1_bandnames

STANDARD_BANDS = {
    "S1": _s1_bandnames,
    "S2": _s2_bandnames,
    "ALL": _all_bandnames,
    "RGB": ["B04", "B03", "B02"],
    "10m": ["B02", "B03", "B04", "B08"],
    "20m": ["B05", "B06", "B07", "B11", "B12", "B8A"],
    "60m": ["B01", "B09"],
    2: _s1_bandnames,
    10: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"],
    12: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A", "VH", "VV"],
    3: ["B04", "B03", "B02"],
    4: ["B04", "B03", "B02", "B08"],
}

"""
This file contains band statistics for BigEarthNet v2 (including S1 stats from v1) for various interpolation types.
The statistics are calculated based on the images of the official train split.
"""

means = {
    "60_nearest": {
        "B01": 361.0767822265625,
        "B02": 438.3785400390625,
        "B03": 614.06201171875,
        "B04": 588.4111328125,
        "B05": 942.8433227539062,
        "B06": 1769.931640625,
        "B07": 2049.551513671875,
        "B08": 2193.331787109375,
        "B09": 2241.455322265625,
        "B11": 1568.226806640625,
        "B12": 997.7324829101562,
        "B8A": 2235.556640625,
        "VH": -19.35258674621582,
        "VV": -12.643914222717285,
    },
    "60_bilinear": {
        "B01": 360.7353820800781,
        "B02": 438.4760437011719,
        "B03": 614.1534423828125,
        "B04": 588.5079956054688,
        "B05": 942.8433227539062,
        "B06": 1769.931640625,
        "B07": 2049.551513671875,
        "B08": 2193.369384765625,
        "B09": 2241.177978515625,
        "B11": 1568.226806640625,
        "B12": 997.7324829101562,
        "B8A": 2235.556640625,
        "VH": -19.352697372436523,
        "VV": -12.643946647644043,
    },
    "60_bicubic": {
        "B01": 360.7255554199219,
        "B02": 438.4646301269531,
        "B03": 614.142578125,
        "B04": 588.4970703125,
        "B05": 942.8433227539062,
        "B06": 1769.931640625,
        "B07": 2049.551513671875,
        "B08": 2193.359619140625,
        "B09": 2241.163330078125,
        "B11": 1568.226806640625,
        "B12": 997.7324829101562,
        "B8A": 2235.556640625,
        "VH": -19.352685928344727,
        "VV": -12.643938064575195,
    },
    "120_nearest": {
        "B01": 361.0767822265625,
        "B02": 438.3720703125,
        "B03": 614.0556640625,
        "B04": 588.4096069335938,
        "B05": 942.8433227539062,
        "B06": 1769.931640625,
        "B07": 2049.551513671875,
        "B08": 2193.2919921875,
        "B09": 2241.455322265625,
        "B11": 1568.226806640625,
        "B12": 997.7324829101562,
        "B8A": 2235.556640625,
        "VH": -19.352558135986328,
        "VV": -12.643863677978516,
    },
    "120_bilinear": {
        "B01": 360.64678955078125,
        "B02": 438.3720703125,
        "B03": 614.0556640625,
        "B04": 588.4096069335938,
        "B05": 942.7476806640625,
        "B06": 1769.8486328125,
        "B07": 2049.475830078125,
        "B08": 2193.2919921875,
        "B09": 2241.10595703125,
        "B11": 1568.2115478515625,
        "B12": 997.715087890625,
        "B8A": 2235.48681640625,
        "VH": -19.352558135986328,
        "VV": -12.643863677978516,
    },
    "120_bicubic": {
        "B01": 360.637451171875,
        "B02": 438.3720703125,
        "B03": 614.0556640625,
        "B04": 588.4096069335938,
        "B05": 942.7472534179688,
        "B06": 1769.8485107421875,
        "B07": 2049.475830078125,
        "B08": 2193.2919921875,
        "B09": 2241.091064453125,
        "B11": 1568.2117919921875,
        "B12": 997.715087890625,
        "B8A": 2235.48681640625,
        "VH": -19.352558135986328,
        "VV": -12.643863677978516,
    },
    "128_nearest": {
        "B01": 361.0660705566406,
        "B02": 438.385009765625,
        "B03": 614.0690307617188,
        "B04": 588.4198608398438,
        "B05": 942.854248046875,
        "B06": 1769.951904296875,
        "B07": 2049.574951171875,
        "B08": 2193.320556640625,
        "B09": 2241.467529296875,
        "B11": 1568.229736328125,
        "B12": 997.7351684570312,
        "B8A": 2235.580810546875,
        "VH": -19.352603912353516,
        "VV": -12.643890380859375,
    },
    "128_bilinear": {
        "B01": 360.64117431640625,
        "B02": 438.366455078125,
        "B03": 614.0504150390625,
        "B04": 588.4042358398438,
        "B05": 942.7412719726562,
        "B06": 1769.84326171875,
        "B07": 2049.470703125,
        "B08": 2193.288330078125,
        "B09": 2241.1015625,
        "B11": 1568.2103271484375,
        "B12": 997.7136840820312,
        "B8A": 2235.482177734375,
        "VH": -19.352550506591797,
        "VV": -12.643857955932617,
    },
    "128_bicubic": {
        "B01": 360.6319885253906,
        "B02": 438.3666687011719,
        "B03": 614.0506591796875,
        "B04": 588.4044799804688,
        "B05": 942.7388916015625,
        "B06": 1769.8411865234375,
        "B07": 2049.468994140625,
        "B08": 2193.288330078125,
        "B09": 2241.086669921875,
        "B11": 1568.210205078125,
        "B12": 997.7135009765625,
        "B8A": 2235.480712890625,
        "VH": -19.35255241394043,
        "VV": -12.64385986328125,
    },
    "no_interpolation": {
        "B01": 361.0767822265625,
        "B02": 438.3720703125,
        "B03": 614.0556640625,
        "B04": 588.4096069335938,
        "B05": 942.8433227539062,
        "B06": 1769.931640625,
        "B07": 2049.551513671875,
        "B08": 2193.2919921875,
        "B09": 2241.455322265625,
        "B11": 1568.226806640625,
        "B12": 997.7324829101562,
        "B8A": 2235.556640625,
        "VH": -19.352558135986328,
        "VV": -12.643863677978516,
    },
}
stds = {
    "60_nearest": {
        "B01": 575.0687255859375,
        "B02": 607.0214233398438,
        "B03": 603.2854614257812,
        "B04": 684.5560302734375,
        "B05": 738.4326782226562,
        "B06": 1100.4560546875,
        "B07": 1275.805419921875,
        "B08": 1369.358154296875,
        "B09": 1316.393310546875,
        "B11": 1070.1612548828125,
        "B12": 813.5276489257812,
        "B8A": 1356.5440673828125,
        "VH": 5.590489864349365,
        "VV": 5.133487224578857,
    },
    "60_bilinear": {
        "B01": 563.8627319335938,
        "B02": 602.119140625,
        "B03": 597.8518676757812,
        "B04": 678.4979858398438,
        "B05": 738.4326782226562,
        "B06": 1100.4560546875,
        "B07": 1275.805419921875,
        "B08": 1359.0667724609375,
        "B09": 1294.6435546875,
        "B11": 1070.1612548828125,
        "B12": 813.5276489257812,
        "B8A": 1356.5440673828125,
        "VH": 5.482285022735596,
        "VV": 5.045091152191162,
    },
    "60_bicubic": {
        "B01": 572.9849853515625,
        "B02": 607.8173217773438,
        "B03": 604.022705078125,
        "B04": 685.4400634765625,
        "B05": 738.4326782226562,
        "B06": 1100.4560546875,
        "B07": 1275.805419921875,
        "B08": 1370.117431640625,
        "B09": 1313.8580322265625,
        "B11": 1070.1612548828125,
        "B12": 813.5276489257812,
        "B8A": 1356.5440673828125,
        "VH": 5.585089206695557,
        "VV": 5.1350226402282715,
    },
    "120_nearest": {
        "B01": 575.0687255859375,
        "B02": 607.02685546875,
        "B03": 603.2968139648438,
        "B04": 684.56884765625,
        "B05": 738.4326782226562,
        "B06": 1100.4560546875,
        "B07": 1275.805419921875,
        "B08": 1369.3717041015625,
        "B09": 1316.393310546875,
        "B11": 1070.1612548828125,
        "B12": 813.5276489257812,
        "B8A": 1356.5440673828125,
        "VH": 5.590505599975586,
        "VV": 5.133493900299072,
    },
    "120_bilinear": {
        "B01": 563.1734008789062,
        "B02": 607.02685546875,
        "B03": 603.2968139648438,
        "B04": 684.56884765625,
        "B05": 727.5784301757812,
        "B06": 1087.4288330078125,
        "B07": 1261.4302978515625,
        "B08": 1369.3717041015625,
        "B09": 1294.35546875,
        "B11": 1063.9197998046875,
        "B12": 806.8846435546875,
        "B8A": 1342.490478515625,
        "VH": 5.590505599975586,
        "VV": 5.133493900299072,
    },
    "120_bicubic": {
        "B01": 572.3436889648438,
        "B02": 607.02685546875,
        "B03": 603.2968139648438,
        "B04": 684.56884765625,
        "B05": 738.3037719726562,
        "B06": 1100.46142578125,
        "B07": 1275.843505859375,
        "B08": 1369.3717041015625,
        "B09": 1313.6488037109375,
        "B11": 1070.8011474609375,
        "B12": 814.0936279296875,
        "B8A": 1356.754150390625,
        "VH": 5.590505599975586,
        "VV": 5.133493900299072,
    },
    "128_nearest": {
        "B01": 575.0419311523438,
        "B02": 607.0984497070312,
        "B03": 603.3590087890625,
        "B04": 684.6243896484375,
        "B05": 738.4649047851562,
        "B06": 1100.47900390625,
        "B07": 1275.8310546875,
        "B08": 1369.4029541015625,
        "B09": 1316.40380859375,
        "B11": 1070.170166015625,
        "B12": 813.5411376953125,
        "B8A": 1356.568359375,
        "VH": 5.590465068817139,
        "VV": 5.133439540863037,
    },
    "128_bilinear": {
        "B01": 563.1304931640625,
        "B02": 601.334716796875,
        "B03": 597.1731567382812,
        "B04": 677.864990234375,
        "B05": 727.5227661132812,
        "B06": 1087.3963623046875,
        "B07": 1261.40380859375,
        "B08": 1358.791015625,
        "B09": 1294.337646484375,
        "B11": 1063.9158935546875,
        "B12": 806.8798217773438,
        "B8A": 1342.467529296875,
        "VH": 5.481307506561279,
        "VV": 5.044347763061523,
    },
    "128_bicubic": {
        "B01": 572.3040161132812,
        "B02": 607.2115478515625,
        "B03": 603.5126953125,
        "B04": 684.9766845703125,
        "B05": 738.2666625976562,
        "B06": 1100.4417724609375,
        "B07": 1275.828857421875,
        "B08": 1369.9761962890625,
        "B09": 1313.63623046875,
        "B11": 1070.8001708984375,
        "B12": 814.0921630859375,
        "B8A": 1356.7420654296875,
        "VH": 5.584985256195068,
        "VV": 5.135045528411865,
    },
    "no_interpolation": {
        "B01": 575.0687255859375,
        "B02": 607.02685546875,
        "B03": 603.2968139648438,
        "B04": 684.56884765625,
        "B05": 738.4326782226562,
        "B06": 1100.4560546875,
        "B07": 1275.805419921875,
        "B08": 1369.3717041015625,
        "B09": 1316.393310546875,
        "B11": 1070.1612548828125,
        "B12": 813.5276489257812,
        "B8A": 1356.5440673828125,
        "VH": 5.590505599975586,
        "VV": 5.133493900299072,
    },
}


def _numpy_level_aggreagate(df, key_col, val_col):
    # optimized version of df.groupby(key_col)[val_col].apply(list).reset_index(name=val_col)
    # credits to B. M. @
    # https://stackoverflow.com/questions/22219004/how-to-group-dataframe-rows-into-list-in-pandas-groupby
    keys, values = df.sort_values(key_col).values.T
    ukeys, index = np.unique(keys, True)
    arrays = np.split(values, index[1:])
    df2 = pd.DataFrame({key_col: ukeys, val_col: [list(a) for a in arrays]})
    return df2


NEW_LABELS_ORIGINAL_ORDER = (
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
)
NEW_LABELS = sorted(NEW_LABELS_ORIGINAL_ORDER)
_val2idx_new_labels = {x: NEW_LABELS.index(x) for x in NEW_LABELS}
_val2idx_new_labels_original_order = {x: NEW_LABELS_ORIGINAL_ORDER.index(x) for x in NEW_LABELS_ORIGINAL_ORDER}


def stack_and_interpolate(
    bands: Dict[str, np.ndarray],
    order: Optional[Iterable[str]] = None,
    img_size: int = 120,
    upsample_mode: str = "nearest",
) -> np.array:
    """
    Supports 2D input (as values in the dict) with "nearest", "bilinear" and "bicubic" interpolation
    """

    def _interpolate(img_data):
        if not img_data.shape[-2:] == (img_size, img_size):
            return F.interpolate(
                torch.Tensor(np.float32(img_data)).unsqueeze(0).unsqueeze(0),
                (img_size, img_size),
                mode=upsample_mode,
                align_corners=True if upsample_mode in ["bilinear", "bicubic"] else None,
            ).squeeze()
        else:
            return torch.Tensor(np.float32(img_data))

    # if order is None, order is alphabetical
    if order is None:
        order = sorted(bands.keys())
    return torch.stack([_interpolate(bands[x]) for x in order])


def ben_19_labels_to_multi_hot(labels: Iterable[str], lex_sorted: bool = True) -> torch.Tensor:
    return torch.from_numpy(ben_19_labels_to_multi_hot_np(labels, lex_sorted))


def ben_19_labels_to_multi_hot_np(labels: Iterable[str], lex_sorted: bool = True) -> np.array:
    """
    Convenience function that converts an input iterable of labels into
    a multi-hot encoded vector.
    If `lex_sorted` is True (default) the classes are lexigraphically ordered, as they are
    in `constants.NEW_LABELS`.
    If `lex_sorted` is False, the original order from the BigEarthNet paper is used, as
    they are given in `constants.NEW_LABELS_ORIGINAL_ORDER`.

    If an unknown label is given, a `KeyError` is raised.

    Be aware that this approach assumes that **all** labels are actually used in the dataset!
    This is not necessarily the case if you are using a subset!
    """
    lbls_to_idx = _val2idx_new_labels if lex_sorted else _val2idx_new_labels_original_order
    idxs = [lbls_to_idx[label] for label in labels]
    multi_hot = np.zeros(len(NEW_LABELS))
    multi_hot[idxs] = 1.0
    return multi_hot


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
    return resolve_data_dir_for_ds("benv2", data_dir, allow_mock=allow_mock, force_mock=force_mock)


def resolve_band_combi(bands: Union[Iterable, str, int]) -> list:
    """
    Resolves a predefined combination of bands or a list of bands into a list of
    individual bands and checks if all bands contained are actual S1/S2 band names.

    :param bands: a combination of bands as defined in BAND_COMBINATION_PREDEFINTIONS
        or a list of bands, or a single band, e.g. "B02", ["B02", "B03"], 2, "S1", "S2"
    :return: a list of bands contained in the predefinition or the single band as list
    """
    if isinstance(bands, str):
        if bands in _s1_bandnames or bands in _s2_bandnames:
            bands = [bands]
        else:
            assert bands in STANDARD_BANDS.keys(), (
                "Band combination unknown, please use a list of strings or one of " f"{STANDARD_BANDS.keys()}"
            )
            bands = STANDARD_BANDS[bands]
    elif isinstance(bands, int):
        assert bands in STANDARD_BANDS.keys(), (
            "Band combination unknown, please use a list of strings or one of " f"{STANDARD_BANDS.keys()}"
        )
        bands = STANDARD_BANDS[bands]
    elif isinstance(bands, Iterable):
        for band in bands:
            assert band in _all_bandnames, f"Band '{band}' unknown"
    else:
        raise ValueError(f"Unknown type of bands: {type(bands)}")
    assert isinstance(bands, list), "Bands should be a list"
    return bands


def band_combi_to_mean_std(bands: Iterable[str], interpolation: str = "120_nearest"):
    """
    Retrievs the mean and standard deviation for a given BigEarthNet
    BAND_COMBINATION_PREDEFINTIONS or list of bands.

    :param bands: a list of bandnames
    :return: mean and standard deviation for the given combination in same order
    """
    bands = resolve_band_combi(bands)
    ben_channel_mean = np.array([means[interpolation][band] for band in bands])
    ben_channel_std = np.array([stds[interpolation][band] for band in bands])
    return ben_channel_mean, ben_channel_std


class BENv2LDMBReader:
    def __init__(
        self,
        image_lmdb_file: Union[str, Path],
        label_file: Union[str, Path],
        s1_mapping_file: Optional[Union[str, Path]] = None,
        bands: Optional[Union[Iterable, str, int]] = None,
        process_bands_fn: Optional[Callable[[Dict[str, np.ndarray], List[str]], Any]] = None,
        process_labels_fn: Optional[Callable[[List[str]], Any]] = None,
        print_info: bool = False,
    ):
        self.image_lmdb_file = image_lmdb_file
        self.env = None
        self.print_info_toggle = print_info

        self.bands = bands if bands is not None else _all_bandnames
        self.bands = resolve_band_combi(self.bands)
        self.uses_s1 = any([x in _s1_bandnames for x in self.bands])
        self.uses_s2 = any([x in _s2_bandnames for x in self.bands])

        if s1_mapping_file is None:
            assert not self.uses_s1, "If you want to use S1 bands, please provide a s2s1_mapping_file"
            self.mapping = None
        else:
            # read and create mapping S2v2 name -> S1 name
            self._print_info("Reading mapping ...")
            mapping = pd.read_csv(str(s1_mapping_file))
            self._print_info("Creating mapping dict ...")
            self.mapping = dict(zip(mapping.patch_id, mapping.s1_name))  # naming of the columns is hardcoded
            del mapping

        # read labels and create mapping S2v2 name -> List[label]
        self._print_info("Reading labels ...")
        lbls = pd.read_csv(str(label_file))

        self._print_info("Aggregating label list ...")
        lbls = _numpy_level_aggreagate(lbls, "patch_id", "label")
        # lbls = lbls.groupby('patch')['lbl_19'].apply(list).reset_index(name='lbl_19')

        self._print_info("Creating label dict ...")
        self.lbls = dict(zip(lbls.patch_id, lbls.label))  # naming of the columns is hardcoded
        self.lbl_key_set = set(self.lbls.keys())
        del lbls

        # set mean and std based on bands selected
        self.mean = None
        self.std = None

        self.process_bands_fn = process_bands_fn if process_bands_fn is not None else lambda x, y: x
        self.process_labels_fn = process_labels_fn if process_labels_fn is not None else lambda x: x

        self._keys: Optional[set] = None
        self._S2_keys: Optional[set] = None
        self._S1_keys: Optional[set] = None

    def _print_info(self, info: str):
        if self.print_info_toggle:
            timestamp = time()
            GREEN = "\033[92m"
            RESET = "\033[0m"
            print(f"{GREEN}[DEBUG] [{timestamp:.5f}]{RESET} ", info)

    def open_env(self):
        if self.env is None:
            self._print_info("Opening LMDB environment ...")
            self.env = lmdb.open(
                str(self.image_lmdb_file),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=False,
                map_size=8 * 1024**3,  # 8GB blocked for caching
                max_spare_txns=16,  # expected number of concurrent transactions (e.g. threads/workers)
            )

    def keys(self, update: bool = False):
        self.open_env()
        if self._keys is None or update:
            self._print_info("(Re-)Reading keys ...")
            assert self.env is not None, "Environment not opened yet"
            with self.env.begin() as txn:
                self._keys = set(txn.cursor().iternext(values=False))
            self._keys = {x.decode() for x in self._keys}
        return self._keys

    def S2_keys(self, update: bool = False):
        if self._S2_keys is None or update:
            self._print_info("(Re-)Reading S2 keys ..")
            self._S2_keys = {key for key in self.keys(update) if key.startswith("S2")}
        return self._S2_keys

    def S1_keys(self, update: bool = False):
        if self._S1_keys is None or update:
            self._print_info("(Re-)Reading S1 keys")
            self._S1_keys = {key for key in self.keys(update) if key.startswith("S1")}
        return self._S1_keys

    def __getitem__(self, key: str):
        # the key is the name of the S2v2 patch

        # open lmdb file if not opened yet
        self.open_env()
        img_data_dict: dict = {}
        if self.uses_s2:
            assert self.env is not None, "Environment not opened yet"
            # read image data for S2v2
            with self.env.begin(write=False, buffers=True) as txn:
                byte_data = txn.get(key.encode())
            img_data_dict.update(safetensor_load(bytes(byte_data)))

        if self.uses_s1:
            # read image data for S1
            assert self.mapping is not None, "S1 bands are used, but no mapping is provided"
            s1_key = self.mapping[key]
            assert self.env is not None, "Environment not opened yet"
            with self.env.begin(write=False, buffers=True) as txn:
                byte_data = txn.get(s1_key.encode())
            img_data_dict.update(safetensor_load(bytes(byte_data)))

        assert isinstance(self.bands, list), "Bands should be a list"
        img_data_dict = {k: v for k, v in img_data_dict.items() if k in self.bands}

        img_data = self.process_bands_fn(img_data_dict, self.bands)
        labels = self.lbls[key] if key in self.lbl_key_set else []
        labels = self.process_labels_fn(labels)

        return img_data, labels
