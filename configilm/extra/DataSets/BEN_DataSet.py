"""
Dataset for BigEarthNet dataset. Files can be requested by contacting
the author.
Original Paper of Image Data:
https://arxiv.org/abs/2105.07921

https://bigearth.net/
"""
import csv
import pathlib
from pathlib import Path
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Union

from torch.utils.data import Dataset

from configilm.extra.BEN_lmdb_utils import ben19_list_to_onehot
from configilm.extra.BEN_lmdb_utils import BENLMDBReader
from configilm.util import Messages


def _csv_files_to_patch_list(csv_files: Union[Path, Iterable[Path]]):
    if isinstance(csv_files, pathlib.Path):
        csv_files = [csv_files]
    patches = []
    for file in csv_files:
        with open(file) as f:
            reader = csv.reader(f)
            patches += list(reader)

    # lines get read as arrays -> flatten to one dimension
    return [x[0] for x in patches]


class BENDataSet(Dataset):
    avail_chan_configs = {
        2: "Sentinel-1",
        3: "RGB",
        4: "10m Sentinel-2",
        10: "10m + 20m Sentinel-2",
        12: "10m + 20m Sentinel-2 + 10m Sentinel-1",
    }

    @classmethod
    def get_available_channel_configurations(cls):
        """
        Prints all available preconfigured channel combinations.
        """
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
        patch_prefilter: Optional[Callable[[str], bool]] = None,
    ):
        """
        Creates a torch Dataset for the BigEarthNet dataset.
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

        :param patch_prefilter: Callable to filter patches after reading csv files

            :Default: None
        """
        super().__init__()
        self.return_patchname = return_patchname
        self.root_dir = Path(root_dir)
        self.lmdb_dir = self.root_dir / "BigEarthNetEncoded.lmdb"
        self.transform = transform
        self.image_size = img_size
        if img_size[0] not in self.avail_chan_configs.keys():
            BENDataSet.get_available_channel_configurations()
            raise AssertionError(f"{img_size[0]} is not a valid channel configuration.")

        self.read_channels = img_size[0]

        print(f"Loading BEN data for {split}...")
        # if csv files are not specified, assume they are located in the root dir and
        # collect from there
        if csv_files is None:
            # get csv files from root dir
            if split is None:
                # collect all splits and get data from there
                splits = ["train", "val", "test"]
                csv_files = [self.root_dir / f"{s}.csv" for s in splits]
            else:
                # get data for this split
                csv_files = self.root_dir / f"{split}.csv"
        else:
            if split is not None:
                Messages.warn(
                    "You specified a split and a csv file - this may be a "
                    "potential conflict and cannot be resolved."
                )

        # get data from this csv file(s)
        self.patches = _csv_files_to_patch_list(csv_files)
        print(f"    {len(self.patches)} patches indexed")

        # if a prefilter is provided, filter patches based on function
        if patch_prefilter:
            self.patches = list(filter(patch_prefilter, self.patches))
        print(f"    {len(self.patches)} pre-filtered patches indexed")

        # sort list for reproducibility
        self.patches.sort()
        if (
            max_img_idx is not None
            and max_img_idx < len(self.patches)
            and max_img_idx != -1
        ):
            self.patches = self.patches[:max_img_idx]

        print(f"    {len(self.patches)} filtered patches indexed")
        self.BENLoader = BENLMDBReader(
            lmdb_dir=self.lmdb_dir,
            label_type="new",
            image_size=self.image_size,
            bands=self.image_size[0],
        )

    def get_patchname_from_index(self, idx: int) -> Optional[None]:
        """
        Gives the patch name of the image at the specified index. May return invalid
        names (names that are not actually loadable because they are not part of the
        lmdb file) if the name is included in the csv file.

        :param idx: index of an image
        :return: patch name of the image or None, if the index is invalid
        """
        if idx > len(self):
            return None
        return self.patches[idx]

    def get_index_from_patchname(self, patchname: str) -> Optional[int]:
        """
        Gives the index of the image of a specific name. Does not distinguish between
        invalid names (not in original BigEarthNet) and names not in loaded list.

        :param patchname: name of an image
        :return: index of the image or None, if the name is not loaded
        """
        if patchname not in set(self.patches):
            return None
        return self.patches.index(patchname)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        key = self.patches[idx]

        # get (& write) image from (& to) LMDB
        # get image from database
        # we have to copy, as the image in imdb is not writeable,
        # which is a problem in .to_tensor()
        img, labels = self.BENLoader[key]

        if img is None:
            print(f"Cannot load {key} from database")
            raise ValueError
        if self.transform:
            img = self.transform(img)

        label_list = labels
        labels = ben19_list_to_onehot(labels)

        assert sum(labels) == len(set(label_list)), f"Label creation failed for {key}"
        if self.return_patchname:
            return img, labels, key
        return img, labels
