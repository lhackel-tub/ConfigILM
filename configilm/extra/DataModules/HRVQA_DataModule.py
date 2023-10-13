import os
from datetime import datetime
from time import time
from typing import List
from typing import Mapping
from typing import Optional
from typing import Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from configilm.extra.DataModules.dm_defaults import default_train_transform
from configilm.extra.DataModules.dm_defaults import default_transform
from configilm.extra.DataSets.HRVQA_DataSet import _means_1024
from configilm.extra.DataSets.HRVQA_DataSet import _stds_1024
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
        print_infos: bool = False,
        dataset_kwargs: Optional[Mapping] = None,
    ):
        """
        Initializes a pytorch lightning data module.

        :param batch_size: batch size for data loaders

            :Default: 16

        :param data_dir: root directory to images and jsons folder

            :Default: ./

        :param img_size: Size to which all channels will be scaled. Interpolation is
            applied bicubic before any transformation. Also selects if the returned
            images are RGB or grayscale based on the number of channels.

            :Default: (3, 1024, 1024)

        :param num_workers_dataloader: number of workers used for data loading

            :Default: #CPU_cores/2

        :param max_img_idx: maximum number of images to load per split. If this number
            is higher than the images found in the csv, None or -1, all images will be
            loaded.

            :Default: None

        :param shuffle: Flag if dataset should be shuffled. If set to None, only train
            will be shuffled and validation and test won't.

            :Default: None

        :param tokenizer: Tokenizer to use for tokenization of input questions. Expects
            standard huggingface tokenizer. If not set, a default tokenizer will be
            used and a warning shown.

            :Default: None

        :param seq_length: Length of tokenized question. Will be caped to this as
            maximum and expanded to this if the question is too short. Includes start
            and end token.

            :Default: 32

        :param selected_answers: List of selected answers or None. If set to None,
            answers will be selected based on `classes` for the data set in order of
            frequency of the training set.

            :Default: None

        :param pin_memory: Flag if memory should be pinned for data loading. If not
            specified set to True if cuda devices are used, else false.

            :Default: None

        :param test_splitting_seed: Seed for random division of the "val-div" and
            "test-div" splits. If not set, an AssertionError is raised. If set to
            "repeat", val and test will be the same data.

            For division, the set defined for validation is split into "val-div" and
            "test-div" depending on the `div_seed` and `split_size`. All random states
            are rebuild after a call to the division.

            To get the disjoint sets for validation and test, call the dataset with the
            same parameters once for "val-div" and once for "test-div".

            :Default: None

        :param test_splitting_division: relative size of the validation div if
            subdivision of the validation split is applicable.

            :Default: 0.5

        :param print_infos: Flag, if additional information during setup() and reading
            should be printed (e.g. number of workers detected, number of images loaded)

            :Default: False

        :param dataset_kwargs: Other keyword arguments to pass to the dataset during
            creation.

        Split example:
            >>> ds_v = HRVQADataSet(..., div_seed=0, split_size=0.3, split="val-div")
            >>> ds_t = HRVQADataSet(..., div_seed=0, split_size=0.3, split="test-div")
            ds_v and ds_t are disjoint with dv containing 30% of all validation samples
            and dt 70%
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
            [_means_1024["red"], _means_1024["green"], _means_1024["blue"]]
            if self.img_size[0] == 3
            else [_means_1024["mono"]]
        )
        std = (
            [_stds_1024["red"], _stds_1024["green"], _stds_1024["blue"]]
            if self.img_size[0] == 3
            else [_stds_1024["mono"]]
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
        self.ds_kwargs = dataset_kwargs if dataset_kwargs is not None else dict()

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

    def _print_info(self, info):
        """
        Helper method that only prints if `print_info` is set. Used to reduce
        complexity in functions.
        """
        if self.print_infos:
            print(info)
        else:
            pass

    def setup(self, stage: Optional[str] = None):
        """
        Prepares the data sets for the specific stage.

        - "fit": train and validation data set
        - "test": test data set
        - None: all data sets

        Prints the time it needed for this operation and other statistics if print_infos
        is set.

        :param stage: None, "fit" or "test"
        """
        self._print_info(
            f"({datetime.now().strftime('%H:%M:%S')}) Datamodule setup called"
        )
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
                    **self.ds_kwargs,
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
                    **self.ds_kwargs,
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
                    **self.ds_kwargs,
                )

        if stage == "predict":
            raise NotImplementedError("Predict stage not implemented")

        self._print_info(f"setup took {time() - t0:.2f} seconds")
        self._print_info(sample_info_msg)

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
