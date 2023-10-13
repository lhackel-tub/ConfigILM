"""
Dataset for updated BigEarthNet dataset. Files can be requested by contacting
the author.
Original Paper of Image Data:
https://arxiv.org/abs/2105.07921

https://bigearth.net/
"""
from pathlib import Path
from typing import Iterable
from typing import Optional
from typing import Union

import pandas as pd

from configilm.extra.BEN_lmdb_utils import ben19_list_to_onehot
from configilm.extra.DataSets.BEN_DataSet import BENDataSet


class BEN2DataSet(BENDataSet):
    def __init__(
        self,
        root_dir: Union[str, Path] = Path("../"),
        csv_files: Optional[Union[Path, Iterable[Path]]] = None,
        split: Optional[str] = None,
        transform=None,
        max_img_idx=None,
        img_size=(12, 120, 120),
        return_patchname: bool = False,
        new_label_file: Union[str, Path, None] = None,
    ):
        """
        Inherits from BigEarthNet DataSet. See BENDataSet for additional details.

        Creates a torch Dataset for the updated BigEarthNet dataset.
        Assumes that the cvs files named after the requested split are located next to
        the lmdb file (folder), which was created using BigEarthNetEncoder.

        Image lmdb file is expected to be named "BigEarthNetEncoded.lmdb"

        :param root_dir: root directory to lmdb file and optional train/val/test.csv

            :Default: ../

        :param csv_files: None (uses split-specific csv files) or csv file specifying
            patch names

            :Default: None

        :param split: "train", "val" or "test" or None for all

            :Default: None (loads all splits)

        :param transform: transformations to be applied to loaded images aside from
            scaling all bands to img_size.

            :Default: None

        :param max_img_idx: maximum number of images to load. If this number is higher
            than the images found in the csv, None or -1, all images will be loaded.

            :Default: None

        :param img_size: Size to which all channels will be scaled. Interpolation is
            applied bicubic before any transformation.

            Also specifies which channels to load.
            See `BENDataSet.get_available_channel_configurations()` for details.

            :Default: (12, 120, 120)

        :param return_patchname: If set to True, __getitem__ will return
            (img, lbl, patch_name) instead of (img, lbl)

            :Default: False

        :param new_label_file: parquet file of the new labels. If not set, expected to
            be called "labels.parquet" and located next to the lmdb file.

            :Default: None
        """
        # read label_file and make it a dict for fast access
        if new_label_file is None:
            self.new_label_file = Path(root_dir) / "labels.parquet"
        else:
            self.new_label_file = Path(new_label_file)

        label_df = pd.read_parquet(self.new_label_file, engine="pyarrow")
        self.label_dict = dict(zip(label_df.name, label_df.labels))

        # define prefilter function that filter patches before applying max index
        lblset = set(self.label_dict.keys())

        super().__init__(
            root_dir=root_dir,
            csv_files=csv_files,
            split=split,
            transform=transform,
            max_img_idx=max_img_idx,
            img_size=img_size,
            return_patchname=True,
            patch_prefilter=lambda x: x in lblset,
        )
        # we have to use a different variable here because otherwise super will not
        # return the patchname which we need for the new labels
        self.return_patchname_self = return_patchname

    def __getitem__(self, idx):
        ret_val = super().__getitem__(idx=idx)
        assert len(ret_val) == 3, (
            f"Can't handle {len(ret_val)}-element returnvalues. "
            f"There should be 3 values (img, label, key)."
        )
        img, old_labels, key = ret_val
        labels = ben19_list_to_onehot(self.label_dict[key])
        if self.return_patchname_self:
            return img, labels, key
        return img, labels
