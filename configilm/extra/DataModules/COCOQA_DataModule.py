import os
import warnings
from datetime import datetime
from time import time
from typing import List
from typing import Optional
from typing import Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from configilm.extra.CustomTorchClasses import MyGaussianNoise
from configilm.extra.DataSets.COCOQA_DataSet import COCOQADataSet
from configilm.util import Messages


class COCOQADataModule(pl.LightningDataModule):
    train_ds: Union[None, COCOQADataSet] = None
    val_ds: Union[None, COCOQADataSet] = None
    test_ds: Union[None, COCOQADataSet] = None
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
        seq_length=64,
    ):
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
        self.max_img_idx = max_img_idx
        self.img_size = (3, 120, 120) if img_size is None else img_size
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        if self.shuffle is not None:
            Messages.hint(
                f"Shuffle was set to {self.shuffle}. This is not recommended for most "
                f"configuration. Use shuffle=None (default) for recommended "
                f"configuration."
            )

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size[1], self.img_size[2]), antialias=True),
                MyGaussianNoise(0.1),
                # transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=10),
                # MyRotateTransform([0, 90, 180, 270]),
                # normalize?
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size[1], self.img_size[2]), antialias=True),
                # normalize?
            ]
        )
        # self.transform = None
        self.pin_memory = torch.cuda.device_count() > 0

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        print(f"({datetime.now().strftime('%H:%M:%S')}) Datamodule setup called")
        sample_info_msg = ""
        t0 = time()

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_ds = COCOQADataSet(
                self.data_dir,
                split="train",
                transform=self.train_transform,
                max_img_idx=self.max_img_idx,
                img_size=self.img_size,
                tokenizer=self.tokenizer,
                seq_length=self.seq_length,
            )
            if self.selected_answers is None:
                self.selected_answers = self.train_ds.answers
                self.selected_answers.extend(
                    ["INVALID"]
                    * (self.train_ds.num_classes - len(self.train_ds.answers))
                )

            self.val_ds = COCOQADataSet(
                self.data_dir,
                split="test",
                transform=self.transform,
                max_img_idx=self.max_img_idx,
                img_size=self.img_size,
                tokenizer=self.tokenizer,
                seq_length=self.seq_length,
            )
            sample_info_msg += f"  Total training samples: {len(self.train_ds):8,d}"
            sample_info_msg += f"  Total validation samples: {len(self.val_ds):8,d}"
            warnings.warn("Validation and Test set are equal in this Dataset.")

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = COCOQADataSet(
                self.data_dir,
                split="test",
                transform=self.transform,
                max_img_idx=self.max_img_idx,
                img_size=self.img_size,
                tokenizer=self.tokenizer,
                seq_length=self.seq_length,
            )
            sample_info_msg += f"  Total test samples: {len(self.test_ds):8,d}"
            warnings.warn("Validation and Test set are equal in this Dataset.")

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
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )
