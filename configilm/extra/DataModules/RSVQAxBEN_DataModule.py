"""
Dataloader and Datamodule for RSVQAxBEN dataset. Files can be requested by contacting
the author.
Original Paper of Dataset:
https://rsvqa.sylvainlobry.com/IGARSS21.pdf
Based on Image Data from:
https://arxiv.org/abs/2105.07921
https://bigearth.net/
"""
from datetime import datetime
from pathlib import Path
from time import time
from typing import Callable
from typing import Mapping
from typing import Optional

from torch.utils.data import DataLoader

from configilm.extra.BEN_lmdb_utils import band_combi_to_mean_std
from configilm.extra.DataModules.ClassificationVQADataModule import ClassificationVQADataModule
from configilm.extra.DataSets.RSVQAxBEN_DataSet import RSVQAxBENDataSet
from configilm.extra._defaults import default_train_transform
from configilm.extra._defaults import default_transform


class RSVQAxBENDataModule(ClassificationVQADataModule):
    def __init__(
            self,
            data_dirs: Mapping[str, Path],
            batch_size: int = 16,
            img_size: tuple = (12, 120, 120),
            num_workers_dataloader: int = 4,
            shuffle: Optional[bool] = None,
            max_len: Optional[int] = None,
            tokenizer: Optional[Callable] = None,
            seq_length: int = 64,
            pin_memory: Optional[bool] = None,
    ):
        super().__init__(
            data_dirs=data_dirs,
            batch_size=batch_size,
            img_size=img_size,
            num_workers_dataloader=num_workers_dataloader,
            shuffle=shuffle,
            max_len=max_len,
            tokenizer=tokenizer,
            seq_length=seq_length,
            pin_memory=pin_memory,
        )

        ben_mean, ben_std = band_combi_to_mean_std(self.img_size[0])

        self.train_transforms = default_train_transform(
            img_size=(self.img_size[1], self.img_size[2]), mean=ben_mean, std=ben_std
        )
        self.eval_transforms = default_transform(img_size=(self.img_size[1], self.img_size[2]), mean=ben_mean,
                                                 std=ben_std)

    def setup(self, stage: Optional[str] = None):
        sample_info_msg = ""

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.train_ds is None:
                self.train_ds = RSVQAxBENDataSet(
                    self.data_dirs,
                    split="train",
                    transform=self.train_transforms,
                    max_len=self.max_len,
                    img_size=self.img_size,
                    tokenizer=self.tokenizer,
                    seq_length=self.seq_length,
                )
            if self.selected_answers is None:
                self.selected_answers = self.train_ds.answers

            if self.val_ds is None:
                self.val_ds = RSVQAxBENDataSet(
                    self.data_dirs,
                    split="val",
                    transform=self.eval_transforms,
                    max_len=self.max_len,
                    img_size=self.img_size,
                    tokenizer=self.tokenizer,
                    seq_length=self.seq_length,
                )
            sample_info_msg += f"  Total training samples: {len(self.train_ds):8,d}"
            sample_info_msg += f"  Total validation samples: {len(self.val_ds):8,d}"

        # Assign test dataset for use in dataloader(s)
        if (stage == "test" or stage is None) and self.test_ds is None:
            self.test_ds = RSVQAxBENDataSet(
                self.data_dirs,
                split="test",
                transform=self.eval_transforms,
                max_len=self.max_len,
                img_size=self.img_size,
                tokenizer=self.tokenizer,
                seq_length=self.seq_length,
            )
            sample_info_msg += f"  Total test samples: {len(self.test_ds):8,d}"

        if stage == "predict":
            raise NotImplementedError("Predict stage not implemented")

        print(sample_info_msg)

    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()

    def test_dataloader(self):
        return super().test_dataloader()
