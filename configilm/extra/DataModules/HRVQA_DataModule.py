"""
Dataloader and Datamodule for RSVQA LR dataset.
"""
from pathlib import Path
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Union

from configilm.extra._defaults import default_train_transform
from configilm.extra._defaults import default_transform
from configilm.extra.DataModules.ClassificationVQADataModule import ClassificationVQADataModule
from configilm.extra.DataSets.HRVQA_DataSet import _means_1024
from configilm.extra.DataSets.HRVQA_DataSet import _stds_1024
from configilm.extra.DataSets.HRVQA_DataSet import HRVQADataSet


class HRVQADataModule(ClassificationVQADataModule):
    def __init__(
        self,
        data_dirs: Mapping[str, Path],
        batch_size: int = 16,
        img_size: tuple = (3, 1024, 1024),
        num_workers_dataloader: int = 4,
        shuffle: Optional[bool] = None,
        max_len: Optional[int] = None,
        tokenizer: Optional[Callable] = None,
        seq_length: int = 64,
        pin_memory: Optional[bool] = None,
        test_splitting_seed: Optional[Union[str, int]] = None,
        test_splitting_size: Union[float, int] = 0.5,
    ):
        """
        This class implements the DataModule for the HR-VQA dataset.

        :param data_dirs: A mapping of strings to Path objects that contains the
            paths to the data directories. It should contain the following keys:
            "images", "train_data", "val_data", "test_data". The "_data" keys
            should point to the directory that contains the question and answer
            json files. Each directory should contain the following files:
            "{split}_question.json" and "{split}_answer.json".

        :param batch_size: The batch size to use for the dataloaders.

            :default: 16

        :param img_size: The size of the images.

            :default: (3, 1024, 1024)

        :param num_workers_dataloader: The number of workers to use for the dataloaders.

            :default: 4

        :param shuffle: Whether to shuffle the data in the dataloaders. If None is provided, the data is shuffled
            for training and not shuffled for validation and test.

            :default: None

        :param max_len: The maximum number of qa-pairs to use. If None or -1 is
            provided, all qa-pairs are used.

            :default: None

        :param tokenizer: A callable that is used to tokenize the questions. If set to None, the default tokenizer
            (from configilm.util) is used.

            :default: None

        :param seq_length: The maximum length of the tokenized questions. If the tokenized question is longer than
            this, it will be truncated. If it is shorter, it will be padded.

            :default: 64

        :param pin_memory: Whether to use pinned memory for the dataloaders. If None is
            provided, it is set to True if a GPU is available and False otherwise.

            :default: None

        :param test_splitting_seed: The seed to use for the split of the val-div and test-div
            splits. If set to "repeat", the split will be the same full val split for
            both val-div and test-div. If set to an integer, the split will be different
            every time the dataset is loaded and the seed will be used to initialize
            the random number generator. The state of the random number generator
            will be saved before the split and restored after the split to ensure
            reproducibility independent of the global random state and also that the
            global random state is not affected by the split.

            :default: 42

        :param test_splitting_size: The size of the val-div and test-div splits. If set to a
            float, it should be a value between 0 and 1 and will be interpreted as the
            fraction of the val split to use for the val-div. The rest of the val split
            will be used for the test-div. If set to an integer, it will be interpreted
            as the number of samples to use for the val-div. The rest of the val split
            will be used for the test-div. If div_seed is set to "repeat", the split
            will be the same (full val split) for both val-div and test-div.

            :default: 0.5
        """
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
        assert img_size[0] == 3 and len(img_size) == 3, f"Invalid img_size: {img_size}, expected (3, height, width)"

        mean = [_means_1024["red"], _means_1024["green"], _means_1024["blue"]]
        std = [_stds_1024["red"], _stds_1024["green"], _stds_1024["blue"]]

        self.train_transform = default_train_transform(
            img_size=(self.img_size[1], self.img_size[2]), mean=mean, std=std
        )
        self.eval_transforms = default_transform(img_size=(self.img_size[1], self.img_size[2]), mean=mean, std=std)

        assert isinstance(test_splitting_seed, int) or test_splitting_seed in ["repeat", None,], (
            "test_splitting parameter has to be 'repeat' to use the val split for "
            "testing again, None for no test split or an integer for random splitting "
            f"of the val split. Got: {test_splitting_seed}"
        )
        self.test_splitting_seed = test_splitting_seed
        self.test_splitting_size = test_splitting_size

    def setup(self, stage: Optional[str] = None):
        sample_info_msg = ""

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.train_ds is None:
                self.train_ds = HRVQADataSet(
                    self.data_dirs,
                    split="train",
                    transform=self.train_transform,
                    max_len=self.max_len,
                    img_size=self.img_size,
                    tokenizer=self.tokenizer,
                    seq_length=self.seq_length,
                )
            if self.selected_answers is None:
                self.selected_answers = self.train_ds.answers

            if self.val_ds is None:
                if self.test_splitting_seed is None:
                    # don't split the val set if test_splitting_seed is None
                    val_split = "val"
                    division_seed: Union[str, int] = "Should Not Matter"
                else:
                    # split the val set into val and test
                    val_split = "val-div"
                    division_seed = self.test_splitting_seed

                self.val_ds = HRVQADataSet(
                    self.data_dirs,
                    split=val_split,
                    div_seed=division_seed,
                    split_size=self.test_splitting_size,
                    transform=self.eval_transforms,
                    max_len=self.max_len,
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
                    self.data_dirs,
                    split="test-div",
                    div_seed=self.test_splitting_seed,
                    split_size=self.test_splitting_size,
                    transform=self.eval_transforms,
                    max_len=self.max_len,
                    img_size=self.img_size,
                    tokenizer=self.tokenizer,
                    seq_length=self.seq_length,
                    selected_answers=self.selected_answers,
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
