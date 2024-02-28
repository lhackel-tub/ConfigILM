from pathlib import Path
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Union
from warnings import warn

import torch

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl
from torch.utils.data import DataLoader


class ClassificationVQADataModule(pl.LightningDataModule):
    train_ds: Union[None, torch.utils.data.Dataset] = None
    val_ds: Union[None, torch.utils.data.Dataset] = None
    test_ds: Union[None, torch.utils.data.Dataset] = None
    predict_ds: Union[None, torch.utils.data.Dataset] = None

    train_transforms: Optional[Callable] = None
    eval_transforms: Optional[Callable] = None

    selected_answers: Union[None, list[str]] = None

    def __init__(
        self,
        data_dirs: Mapping[str, Path],
        batch_size: int = 16,
        img_size: tuple = (3, 120, 120),
        num_workers_dataloader: int = 4,
        shuffle: Optional[bool] = None,
        max_len: Optional[int] = None,
        tokenizer: Optional[Callable] = None,
        seq_length: int = 64,
        pin_memory: Optional[bool] = None,
    ):
        """
        This class is a base class for datamodules that are used for classification
        of visual question answering. It provides some basic functionality that
        is shared between different datamodules. It is not intended to be used
        directly, but rather must be subclassed.

        :param data_dirs: A mapping from file key to file path. The file key is
            used to identify the function of the file. For example, the key
            "questions.txt" is used to identify the file that contains the
            questions. The file path can be either a string or a Path object.
            Required keys are "images", "train_data" and "test_data".

        :param batch_size: The batch size to use for the dataloaders.

            :default: 16

        :param img_size: The size of the images. Note that this includes the
            number of channels. For example, if the images are RGB images, the
            size should be (3, 120, 120) but for grayscale images, the size should
            be (1, 120, 120). This is dataset specific.

            :default: (3, 120, 120)

        :param num_workers_dataloader: The number of workers to use for the
            dataloaders.

            :default: 4

        :param shuffle: Whether to shuffle the data or not. If None is provided,
            the data is shuffled for training and not shuffled for validation and
            test. This is recommended for most training scenarios.

            :default: None

        :param max_len: The maximum number of qa-pairs to use. If None or -1 is
            provided, all qa-pairs are used.

            :default: None

        :param tokenizer: A callable that is used to tokenize the questions. If
            set to None, the default tokenizer (from configilm.util) is used.

            :default: None

        :param seq_length: The maximum length of the tokenized questions.

            :default: 64

        :param pin_memory: Whether to pin the memory or not. If None is provided,
            the memory is pinned if a GPU is available. Cannot be set to True if
            no GPU is available.

            :default: None
        """
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

    def setup(self, stage: Optional[str] = None):
        """
        Prepares the data sets for the specific stage.

        - "fit": train and validation data set
        - "test": test data set
        - None: all data sets

        :param stage: None, "fit", or "test"

            :default: None
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def train_dataloader(self):
        """
        Returns the dataloader for the training data.

        :raises: AssertionError if the training dataset is not set up. This can happen if the setup()
            method is not called before this method or the dataset has no training data.
        """
        assert self.train_ds is not None, "setup() for training must be called before train_dataloader()"
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """
        Returns the dataloader for the validation data.

        :raises: AssertionError if the validation dataset is not set up. This can happen if the setup()
            method is not called before this method or the dataset has no validation data.
        """
        assert self.val_ds is not None, "setup() for validation must be called before val_dataloader()"
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        """
        Returns the dataloader for the test data.

        :raises: AssertionError if the test dataset is not set up. This can happen if the setup()
            method is not called before this method or the dataset has no test data.
        """
        assert self.test_ds is not None, "setup() for testing must be called before test_dataloader()"
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        assert self.predict_ds is not None, "setup() for prediction must be called before predict_dataloader()"
        return DataLoader(
            self.predict_ds,
            batch_size=self.batch_size,
            shuffle=False if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )
