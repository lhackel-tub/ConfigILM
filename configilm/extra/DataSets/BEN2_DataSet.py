from pathlib import Path
from typing import Iterable
from typing import Optional
from typing import Union

import pandas as pd

from configilm.extra.BEN_lmdb_utils import ben19_list_to_onehot
from configilm.extra.DataSets.BEN_DataSet import BENDataSet


class BEN2DataSet(BENDataSet):
    avail_chan_configs = {
        2: "Sentinel-1",
        3: "RGB",
        4: "10m Sentinel-2",
        10: "10m + 20m Sentinel-2",
        12: "10m + 20m Sentinel-2 + 10m Sentinel-1",
    }

    @classmethod
    def get_available_channel_configurations(cls):
        print("Available channel configurations are:")
        for c, m in cls.avail_chan_configs.items():
            print(f"    {c:>3} -> {m}")

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
