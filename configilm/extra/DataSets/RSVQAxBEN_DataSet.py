"""
Dataloader and Datamodule for RSVQAxBEN dataset. Files can be requested by contacting
the author.
Original Paper of Dataset:
https://rsvqa.sylvainlobry.com/IGARSS21.pdf
Based on Image Data from:
https://arxiv.org/abs/2105.07921
https://bigearth.net/
"""
import json
from pathlib import Path
from typing import Callable
from typing import Mapping
from typing import Optional

import torch

from configilm.extra.BENv1_utils import BENv1LMDBReader
from configilm.extra.data_dir import resolve_data_dir_for_ds
from configilm.extra.DataSets.ClassificationVQADataset import ClassificationVQADataset


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
    return resolve_data_dir_for_ds("rsvqaxben", data_dir, allow_mock=allow_mock, force_mock=force_mock)


class RSVQAxBENDataSet(ClassificationVQADataset):
    def __init__(
        self,
        data_dirs: Mapping[str, Path],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_len: Optional[int] = None,
        img_size: tuple = (12, 120, 120),
        selected_answers: Optional[list] = None,
        num_classes: Optional[int] = 1000,
        tokenizer: Optional[Callable] = None,
        seq_length: int = 64,
        return_extras: bool = False,
    ):
        """
        This class implements the RSVQAxBEN dataset. It is a subclass of
        ClassificationVQADataset and provides some dataset specific functionality.

        :param data_dirs: A dictionary containing the paths to the different data directories.
            Required keys are "images_lmdb", "train_data", "val_data" and "test_data".
            The "_data" keys are used to identify the directory that contains the
            data files which are named "RSVQAxBEN_QA_{split}.json" that contains
            the qa-pairs for the split.

        :param split: The name of the split to use. Can be either "train", "val" or
            "test". If None is provided, all splits will be used.

            :default: None

        :param transform: A callable that is used to transform the images after
            loading them. If None is provided, no transformation is applied.

            :default: None

        :param max_len: The maximum number of qa-pairs to use. If None or -1 is
            provided, all qa-pairs are used.

            :default: None

        :param img_size: The size of the images. Note that this includes the number of
            channels. For example, if the images are RGB images, the size should be
            (3, h, w). See BEN_DataSet for available channel configurations.

            :default: (12, 120, 120)

        :param selected_answers: A list of answers that should be used. If None
            is provided, the num_classes most common answers are used. If
            selected_answers is not None, num_classes is ignored.

            :default: None

        :param num_classes: The number of classes to use. Only used if
            selected_answers is None. If set to None, all answers are used.

            :default: 1000

        :param tokenizer: A callable that is used to tokenize the questions. If
            set to None, the default tokenizer (from configilm.util) is used.

            :default: None

        :param seq_length: The maximum length of the tokenized questions.

            :default: 64

        :param return_extras: If True, the dataset will return the type of the
            question in addition to the image, question and answer.

            :default: False
        """
        assert img_size[0] in [2, 3, 4, 10, 12], (
            "Image Channels have to be "
            "2 (Sentinel-1), "
            "3 (RGB), "
            "4 (10m Sentinel-2), "
            "10 (10m + 20m Sentinel-2) or "
            "12 (10m + 20m Sentinel-2 + 10m Sentinel-1) "
            "but was " + f"{img_size[0]}"
        )
        print(f"Loading split RSVQAxBEN data for {split}...")

        super().__init__(
            data_dirs=data_dirs,
            split=split,
            transform=transform,
            max_len=max_len,
            img_size=img_size,
            selected_answers=selected_answers,
            num_classes=num_classes,
            tokenizer=tokenizer,
            seq_length=seq_length,
            return_extras=return_extras,
        )

        self.BENLoader = BENv1LMDBReader(
            lmdb_dir=self.data_dirs["images_lmdb"],
            label_type="new",
            image_size=self.img_size,
            bands=self.img_size[0],
        )

    def split_names(self) -> set[str]:
        return {"train", "val", "test"}

    def prepare_split(self, split: str) -> list:
        with open((self.data_dirs[f"{split}_data"] / f"RSVQAxBEN_QA_{split}.json").resolve()) as read_file:
            data = json.load(read_file)
        return [(v["S2_name"], v["question"], v["answer"], v["type"]) for _, v in data.items()]

    def load_image(self, key: str) -> torch.Tensor:
        img, _ = self.BENLoader[key]
        return img
