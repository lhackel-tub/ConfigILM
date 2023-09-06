import os
from datetime import datetime
from time import time
from typing import List
from typing import Optional
from typing import Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from configilm.extra.DataModules.dm_defaults import default_train_transform
from configilm.extra.DataModules.dm_defaults import default_transform
from configilm.extra.DataSets.HRVQA_DataSet import _means
from configilm.extra.DataSets.HRVQA_DataSet import _stds
from configilm.extra.DataSets.HRVQA_DataSet import HRVQADataSet
from configilm.util import Messages


class HRVQADataModule(pl.LightningDataModule):
    train_ds: Union[None, HRVQADataSet] = None
    val_ds: Union[None, HRVQADataSet] = None
    test_ds: Union[None, HRVQADataSet] = None
    selected_answers: Union[None, List[str]] = None

    def __init__(
        self,
        batch_size=16,
        data_dir: str = "./",
        img_size=None,
        num_workers_dataloader=None,
        max_img_idx=None,
        shuffle=None,
        tokenizer=None,
        seq_length=32,
        selected_answers=None,
        pin_memory=None,
        test_splitting_seed=None,
        test_splitting_division=None,
    ):
        if img_size is not None and len(img_size) != 3:
            raise ValueError(
                f"Expected image_size with 3 dimensions (HxWxC) or None but got "
                f"{len(img_size)} dimensions instead"
            )
        super().__init__()
        if num_workers_dataloader is None:
            cpu_count = os.cpu_count()
            if type(cpu_count) is int:
                self.num_workers_dataloader = cpu_count // 2
            else:
                self.num_workers_dataloader = 0
        else:
            self.num_workers_dataloader = num_workers_dataloader
        print(f"Dataloader using {self.num_workers_dataloader} workers")

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_img_idx = (
            max_img_idx if max_img_idx is None or max_img_idx > 0 else None
        )
        self.img_size = (3, 1024, 1024) if img_size is None else img_size
        self.shuffle = shuffle
        if self.shuffle is not None:
            Messages.hint(
                f"Shuffle was set to {self.shuffle}. This is not recommended for most "
                f"configuration. Use shuffle=None (default) for recommended "
                f"configuration."
            )
        self.selected_answers = selected_answers

        mean = (
            [_means["red"], _means["green"], _means["blue"]]
            if self.img_size[0] == 3
            else [_means["mono"]]
        )
        std = (
            [_stds["red"], _stds["green"], _stds["blue"]]
            if self.img_size[0] == 3
            else [_stds["mono"]]
        )

        self.train_transform = default_train_transform(
            img_size=(self.img_size[1], self.img_size[2]), mean=mean, std=std
        )
        self.transform = default_transform(
            img_size=(self.img_size[1], self.img_size[2]), mean=mean, std=std
        )
        # self.transform = None
        self.pin_memory = torch.cuda.device_count() > 0
        self.pin_memory = self.pin_memory if pin_memory is None else pin_memory
        Messages.hint(
            f"pin_memory set to {pin_memory}"
            f"{' ' if pin_memory is None else ' via overwrite'}"
        )

        self.tokenizer = tokenizer
        self.seq_length = seq_length

        assert isinstance(test_splitting_seed, int) or test_splitting_seed in [
            "repeat",
            None,
        ], (
            "test_splitting parameter has to be 'repeat' to use the val split for "
            "testing again, None for no test split or an integer for random splitting "
            "of the val split"
        )
        self.test_splitting_seed = test_splitting_seed
        self.test_splitting_div = test_splitting_division

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        print(f"({datetime.now().strftime('%H:%M:%S')}) Datamodule setup called")
        sample_info_msg = ""
        t0 = time()

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.train_ds is None:
                self.train_ds = HRVQADataSet(
                    self.data_dir,
                    split="train",
                    transform=self.train_transform,
                    max_img_idx=self.max_img_idx,
                    img_size=self.img_size,
                    tokenizer=self.tokenizer,
                    seq_length=self.seq_length,
                )
            if self.selected_answers is None:
                self.selected_answers = self.train_ds.selected_answers

            if self.val_ds is None:
                if self.test_splitting_seed is None:
                    val_split = "val"
                    division_seed = "Should Not Matter"
                else:
                    val_split = "val-div"
                    division_seed = self.test_splitting_seed
                self.val_ds = HRVQADataSet(
                    self.data_dir,
                    split=val_split,
                    div_seed=division_seed,
                    split_size=self.test_splitting_div,
                    transform=self.transform,
                    max_img_idx=self.max_img_idx,
                    img_size=self.img_size,
                    tokenizer=self.tokenizer,
                    seq_length=self.seq_length,
                    selected_answers=self.selected_answers,
                )
            sample_info_msg += f"  Total training samples: {len(self.train_ds):8,d}"
            sample_info_msg += f"  Total validation samples: {len(self.val_ds):8,d}"

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            if self.test_splitting_seed is None:
                raise NotImplementedError("Test stage None not implemented")
            else:
                self.test_ds = HRVQADataSet(
                    self.data_dir,
                    split="test-div",
                    div_seed=self.test_splitting_seed,
                    split_size=self.test_splitting_div,
                    transform=self.transform,
                    max_img_idx=self.max_img_idx,
                    img_size=self.img_size,
                    tokenizer=self.tokenizer,
                    seq_length=self.seq_length,
                    selected_answers=self.selected_answers,
                )

        if stage == "predict":
            raise NotImplementedError("Predict stage not implemented")

        print(f"setup took {time() - t0:.2f} seconds")
        print(sample_info_msg)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.test_splitting_seed is None:
            return None
        else:
            return DataLoader(
                self.test_ds,
                batch_size=self.batch_size,
                shuffle=False if self.shuffle is None else self.shuffle,
                num_workers=self.num_workers_dataloader,
                pin_memory=self.pin_memory,
            )
