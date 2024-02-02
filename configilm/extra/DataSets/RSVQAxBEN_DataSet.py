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

from configilm.extra.BEN_lmdb_utils import BENLMDBReader
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
    max_cache_size = 0

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
        """ """
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

        self.BENLoader = BENLMDBReader(
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
