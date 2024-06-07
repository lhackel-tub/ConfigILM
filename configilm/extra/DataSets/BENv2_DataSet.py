"""
Dataset for BigEarthNet dataset. Files can be requested by contacting
the author.
Original Paper of Image Data:
https://arxiv.org/abs/2105.07921

https://bigearth.net/
"""
import csv
from functools import partial
from pathlib import Path
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Union

from torch.utils.data import Dataset

from configilm.extra.BENv2_utils import ben_19_labels_to_multi_hot
from configilm.extra.BENv2_utils import BENv2LDMBReader
from configilm.extra.BENv2_utils import stack_and_interpolate
from configilm.extra.BENv2_utils import STANDARD_BANDS


class BENv2DataSet(Dataset):
    """
    Dataset for BigEarthNet dataset. LMDB-Files can be requested by contacting
    the author or by downloading the dataset from the official website and encoding
    it using the BigEarthNet Encoder.

    The dataset can be loaded with different channel configurations. The channel configuration
    is defined by the first element of the img_size tuple (c, h, w).
    The available configurations are:

        - 2 -> Sentinel-1
        - 3 -> RGB
        - 4 -> 10m Sentinel-2
        - 10 -> 10m + 20m Sentinel-2
        - 12 -> 10m + 20m Sentinel-2 + 10m Sentinel-1
    """

    avail_chan_configs = {
        2: "Sentinel-1",
        3: "RGB",
        4: "10m Sentinel-2",
        10: "10m + 20m Sentinel-2",
        12: "10m + 20m Sentinel-2 + 10m Sentinel-1",
        14: "10m + 20m Sentinel-2 + 60m Sentinel-2 + 10m Sentinel-1 ",
    }

    channel_configurations = {
        2: STANDARD_BANDS["S1"],
        3: STANDARD_BANDS["RGB"],  # RGB order
        4: STANDARD_BANDS["10m"],  # BRGIr order
        10: STANDARD_BANDS["10m"] + STANDARD_BANDS["20m"],
        12: STANDARD_BANDS["10m"] + STANDARD_BANDS["20m"] + STANDARD_BANDS["S1"],
        14: STANDARD_BANDS["10m"] + STANDARD_BANDS["20m"] + STANDARD_BANDS["60m"] + STANDARD_BANDS["S1"],
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
        data_dirs: Mapping[str, Union[str, Path]],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_len: Optional[int] = None,
        img_size: tuple = (3, 120, 120),
        return_extras: bool = False,
        patch_prefilter: Optional[Callable[[str], bool]] = None,
    ):
        """
        Dataset for BigEarthNet v2 dataset. Files can be requested by contacting
        the author or visiting the official website.

        Original Paper of Image Data:
        TO BE PUBLISHED

        :param data_dirs: A mapping from file key to file path. The file key is
            used to identify the function of the file. The required keys are:
            "images_lmdb", "labels_csv", "s1_mapping_csv", "split_csv".

        :param split: The name of the split to use. Can be either "train", "val" or
            "test". If None is provided, all splits are used.

            :default: None

        :param transform: A callable that is used to transform the images after
            loading them. If None is provided, no transformation is applied.

            :default: None

        :param max_len: The maximum number of images to use. If None or -1 is
            provided, all images are used.

            :default: None

        :param img_size: The size of the images. Note that this includes the number of
            channels. For example, if the images are RGB images, the size should be
            (3, h, w).

            :default: (3, 120, 120)

        :param return_extras: If True, the dataset will return the patch name
            as a third return value.

            :default: False

        :param patch_prefilter: A callable that is used to filter the patches
            before they are loaded. If None is provided, no filtering is
            applied. The callable must take a patch name as input and return
            True if the patch should be included and False if it should be
            excluded from the dataset.

            :default: None
        """
        super().__init__()
        self.return_extras = return_extras
        self.lmdb_dir = data_dirs["images_lmdb"]
        self.transform = transform
        self.image_size = img_size
        assert len(img_size) == 3, "Image size must be a tuple of length 3"
        c, h, w = img_size
        assert h == w, "Image size must be square"
        if c not in self.avail_chan_configs.keys():
            BENv2DataSet.get_available_channel_configurations()
            raise AssertionError(f"{img_size[0]} is not a valid channel configuration.")

        print(f"Loading BEN data for {split}...")
        # read split csv file
        split_csv = Path(data_dirs["split_csv"])
        with open(split_csv) as f:
            reader = csv.reader(f)
            split_data = list(reader)
            split_data = split_data[1:]  # remove header
        self.patches = [x[0] for x in split_data if split is None or x[1] == split]
        print(f"    {len(self.patches)} patches indexed")

        # if a prefilter is provided, filter patches based on function
        if patch_prefilter:
            self.patches = list(filter(patch_prefilter, self.patches))
        print(f"    {len(self.patches)} pre-filtered patches indexed")

        # sort list for reproducibility
        self.patches.sort()
        if max_len is not None and max_len < len(self.patches) and max_len != -1:
            self.patches = self.patches[:max_len]
        print(f"    {len(self.patches)} filtered patches indexed")

        self.channel_order = self.channel_configurations[c]
        self.BENv2Loader = BENv2LDMBReader(
            image_lmdb_file=self.lmdb_dir,
            label_file=data_dirs["labels_csv"],
            s1_mapping_file=data_dirs["s1_mapping_csv"],
            bands=self.channel_order,
            process_bands_fn=partial(stack_and_interpolate, img_size=h, upsample_mode="nearest"),
            process_labels_fn=ben_19_labels_to_multi_hot,
        )

    def get_patchname_from_index(self, idx: int) -> Optional[str]:
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
        img, labels = self.BENv2Loader[key]

        if img is None:
            print(f"Cannot load {key} from database")
            raise ValueError
        if self.transform:
            img = self.transform(img)

        if self.return_extras:
            return img, labels, key
        return img, labels
