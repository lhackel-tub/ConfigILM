from pathlib import Path
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Union
from warnings import warn

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class ClassificationVQADataModule(LightningDataModule):
    train_ds: Union[None, torch.utils.data.Dataset] = None
    val_ds: Union[None, torch.utils.data.Dataset] = None
    test_ds: Union[None, torch.utils.data.Dataset] = None

    train_transforms: Optional[Callable] = None
    eval_transforms: Optional[Callable] = None

    selected_answers: Union[None, list[str]] = None

    def __init__(
            self,
            data_dirs: Mapping[str, Union[str, Path]],
            batch_size: int = 16,
            img_size: tuple = (3, 120, 120),
            num_workers_dataloader: int = 4,
            shuffle: Optional[bool] = None,
            max_len: Optional[int] = None,
            tokenizer: Optional[Callable] = None,
            seq_length: int = 64,
            pin_memory: Optional[bool] = None,
    ):
        super().__init__()
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers_dataloader = num_workers_dataloader
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        self.pin_memory = pin_memory
        if self.pin_memory is None:
            self.pin_memory = True if torch.cuda.is_available() else False

        self.shuffle = shuffle
        if self.shuffle is not None:
            warn(
                f"Shuffle was set to {self.shuffle}. This is not recommended for most "
                f"configuration. Use shuffle=None (default) for recommended "
                f"configuration."
            )

    def setup(self, stage=None):
        raise NotImplementedError("This method must be implemented by subclasses.")

    def train_dataloader(self):
        assert self.train_ds is not None, "setup() for training must be called before train_dataloader()"
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        assert self.val_ds is not None, "setup() for validation must be called before val_dataloader()"
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        assert self.test_ds is not None, "setup() for testing must be called before test_dataloader()"
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )
