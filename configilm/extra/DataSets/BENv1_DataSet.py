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
from typing import Mapping
from typing import Optional
from typing import Union

from torch.utils.data import Dataset

from configilm.extra.BENv1_utils import ben19_list_to_onehot
from configilm.extra.BENv1_utils import BENv1LMDBReader


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


class BENv1DataSet(Dataset):
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
        Dataset for BigEarthNet dataset. Files can be requested by contacting
        the author.

        Original Paper of Image Data:
        https://arxiv.org/abs/2105.07921

        :param data_dirs: A mapping from file key to file path. The file key is
            used to identify the function of the file. For example, the key
            "train.csv" contains the names of all images in the training set.
            The file path can be either a string or a Path object. Required
            keys are "images_lmdb", "train_data", "train_data" and "train_data".
            The "_data" keys are used to identify the csv file that contains the
            names of the images that are part of the split.

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
        if img_size[0] not in self.avail_chan_configs.keys():
            BENv1DataSet.get_available_channel_configurations()
            raise AssertionError(f"{img_size[0]} is not a valid channel configuration.")

        self.channels = img_size[0]

        print(f"Loading BEN data for {split}...")
        if split is None:
            csv_files = [data_dirs["train_data"], data_dirs["val_data"], data_dirs["test_data"]]
        else:
            csv_files = [data_dirs[f"{split}_data"]]

        # get data from this csv file(s)
        self.patches = _csv_files_to_patch_list([Path(x) for x in csv_files])
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
        self.BENLoader = BENv1LMDBReader(
            lmdb_dir=self.lmdb_dir,
            label_type="new",
            image_size=self.image_size,
            bands=self.image_size[0],
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
        img, labels = self.BENLoader[key]

        if img is None:
            print(f"Cannot load {key} from database")
            raise ValueError
        if self.transform:
            img = self.transform(img)

        label_list = labels
        labels = ben19_list_to_onehot(labels)

        assert sum(labels) == len(set(label_list)), f"Label creation failed for {key}"
        if self.return_extras:
            return img, labels, key
        return img, labels
