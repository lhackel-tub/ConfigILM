"""
Datamodule for BigEarthNet dataset. Files can be requested by contacting
the author.
Original Paper of Image Data:
https://arxiv.org/abs/2105.07921
https://bigearth.net/
"""
import os
from datetime import datetime
from pathlib import Path
from time import time
from typing import Mapping
from typing import Optional
from typing import Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from configilm.extra.BEN_lmdb_utils import band_combi_to_mean_std
from configilm.extra.DataModules.dm_defaults import default_train_transform
from configilm.extra.DataModules.dm_defaults import default_transform
from configilm.extra.DataSets.BEN_DataSet import BENDataSet
from configilm.util import Messages


class BENDataModule(pl.LightningDataModule):
    num_classes = 19
    train_ds: Union[None, BENDataSet] = None
    val_ds: Union[None, BENDataSet] = None
    test_ds: Union[None, BENDataSet] = None

    def __init__(
        self,
        batch_size=16,
        data_dir: Union[str, Path] = "./",
        img_size=None,
        num_workers_dataloader=None,
        max_img_idx=None,
        shuffle=None,
        print_infos: bool = False,
        dataset_kwargs: Optional[Mapping] = None,
    ):
        """
        Initializes a pytorch lightning data module.

        :param batch_size: batch size for data loaders

            :Default: 16

        :param data_dir: base data directory for lmdb and csv files

            :Default: ./

        :param img_size: image size `(c, h, w)` in accordance with `BENDataSet`

            :Default: None uses default of dataset

        :param num_workers_dataloader: number of workers used for data loading

            :Default: #CPU_cores/2

        :param max_img_idx: maximum number of images to load per split. If this number
            is higher than the images found in the csv, None or -1, all images will be
            loaded.

            :Default: None

        :param shuffle: Flag if dataset should be shuffled. If set to None, only train
            will be shuffled and validation and test won't.

            :Default: None

        :param print_infos: Flag, if additional information during setup() and reading
            should be printed (e.g. number of workers detected, number of images loaded)

            :Default: False

        :param dataset_kwargs: Other keyword arguments to pass to the dataset during
            creation.
        """
        if img_size is not None and len(img_size) != 3:
            raise ValueError(
                f"Expected image_size with 3 dimensions (HxWxC) or None but got "
                f"{len(img_size)} dimensions instead"
            )
        super().__init__()
        self.print_infos = print_infos
        if num_workers_dataloader is None:
            cpu_count = os.cpu_count()
            if type(cpu_count) is int:
                self.num_workers_dataloader = cpu_count // 2
            else:
                self.num_workers_dataloader = 0
        else:
            self.num_workers_dataloader = num_workers_dataloader
        if self.print_infos:
            print(f"Dataloader using {self.num_workers_dataloader} workers")

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_img_idx = max_img_idx
        self.img_size = (12, 120, 120) if img_size is None else img_size
        self.shuffle = shuffle
        self.ds_kwargs = dataset_kwargs if dataset_kwargs is not None else dict()
        if self.shuffle is not None:
            Messages.hint(
                f"Shuffle was set to {self.shuffle}. This is not recommended for most "
                f"configuration. Use shuffle=None (default) for recommended "
                f"configuration."
            )

        # get mean and std
        ben_mean, ben_std = band_combi_to_mean_std(self.img_size[0])

        self.train_transform = default_train_transform(
            img_size=(self.img_size[1], self.img_size[2]), mean=ben_mean, std=ben_std
        )
        self.transform = default_transform(
            img_size=(self.img_size[1], self.img_size[2]), mean=ben_mean, std=ben_std
        )
        self.pin_memory = torch.cuda.device_count() > 0

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Prepares the data sets for the specific stage.

        - "fit": train and validation data set
        - "test": test data set
        - None: all data sets

        Prints the time it needed for this operation and other statistics if print_infos
        is set.

        :param stage: None, "fit" or "test"
        """
        if self.print_infos:
            print(f"({datetime.now().strftime('%H:%M:%S')}) Datamodule setup called")
        sample_info_msg = ""
        t0 = time()

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_ds = BENDataSet(
                self.data_dir,
                split="train",
                transform=self.train_transform,
                max_img_idx=self.max_img_idx,
                img_size=self.img_size,
                **self.ds_kwargs,
            )

            self.val_ds = BENDataSet(
                self.data_dir,
                split="val",
                transform=self.transform,
                max_img_idx=self.max_img_idx,
                img_size=self.img_size,
                **self.ds_kwargs,
            )
            sample_info_msg += f"  Total training samples: {len(self.train_ds):8,d}"
            sample_info_msg += f"  Total validation samples: {len(self.val_ds):8,d}"

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = BENDataSet(
                self.data_dir,
                split="test",
                transform=self.transform,
                max_img_idx=self.max_img_idx,
                img_size=self.img_size,
                **self.ds_kwargs,
            )
            sample_info_msg += f"  Total test samples: {len(self.test_ds):8,d}"

        if stage == "predict":
            raise NotImplementedError("Predict stage not implemented")

        if self.print_infos:
            print(f"setup took {time() - t0:.2f} seconds")
            print(sample_info_msg)

    def train_dataloader(self):
        """
        Create a Dataloader according to the specification in the `__init__` call.
        Uses the train set and expects it to be set (e.g. via `setup()` call)

        :return: torch DataLoader for the train set
        """
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """
        Create a Dataloader according to the specification in the `__init__` call.
        Uses the validation set and expects it to be set (e.g. via `setup()` call)

        :return: torch DataLoader for the validation set
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        """
        Create a Dataloader according to the specification in the `__init__` call.
        Uses the test set and expects it to be set (e.g. via `setup()` call)

        :return: torch DataLoader for the test set
        """
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )
