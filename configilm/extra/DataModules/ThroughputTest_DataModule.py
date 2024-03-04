"""
Dataloader and Datamodule for ThroughputTest dataset.
"""
from pathlib import Path
from typing import Callable
from typing import Mapping
from typing import Optional

from configilm.extra.DataModules.ClassificationVQADataModule import ClassificationVQADataModule
from configilm.extra.DataSets.ThroughputTest_DataSet import VQAThroughputTestDataset


class VQAThroughputTestDataModule(ClassificationVQADataModule):
    def __init__(
        self,
        data_dirs: Mapping[str, Path],
        batch_size: int = 16,
        img_size: tuple = (3, 256, 256),
        num_workers_dataloader: int = 4,
        shuffle: Optional[bool] = None,
        max_len: Optional[int] = None,
        tokenizer: Optional[Callable] = None,
        seq_length: int = 64,
        pin_memory: Optional[bool] = None,
        num_samples: int = 1000,
        num_classes: int = 1000,
    ):
        """
        This class implements the DataModule for the ThroughputTest dataset.

        :param data_dirs: A mapping from file key to file path. Is ignored for this dataset.

        :param batch_size: The batch size to use for the dataloaders.

            :default: 16

        :param img_size: The size of the images.

            :default: (3, 256, 256)

        :param num_workers_dataloader: The number of workers to use for the dataloaders.

            :default: 4

        :param shuffle: Whether to shuffle the data in the dataloaders. If None is provided, the data is shuffled
            for training and not shuffled for validation and test. However, for this dataset, all the data is
            the same, so this parameter has no effect.

            :default: None

        :param max_len: The maximum number of qa-pairs to use. If None or -1 is
            provided, all qa-pairs are used. "all" is defines in num_samples.

            :default: None

        :param tokenizer: A callable that is used to tokenize the questions. Is
            ignored, as the dataset does not use any real questions but included for
            compatibility with other datasets.

            :default: None

        :param seq_length: The maximum length of the tokenized questions. If the tokenized question is longer than
            this, it will be truncated. If it is shorter, it will be padded.

            :default: 64

        :param pin_memory: Whether to use pinned memory for the dataloaders. If None is
            provided, it is set to True if a GPU is available and False otherwise.

            :default: None

        :param num_samples: The number of samples to simulate per split.

            :default: 1000

        :param num_classes: The number of classes in the dataset.

            :default: 1000
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
        self.num_samples = num_samples
        self.num_classes = num_classes

    def setup(self, stage: Optional[str] = None):
        sample_info_msg = ""

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.train_ds is None:
                self.train_ds = VQAThroughputTestDataset(
                    self.data_dirs,
                    split="train",
                    transform=None,
                    max_len=self.max_len,
                    img_size=self.img_size,
                    tokenizer=None,
                    seq_length=self.seq_length,
                    num_samples=self.num_samples,
                    num_classes=self.num_classes,
                )
            if self.selected_answers is None:
                self.selected_answers = self.train_ds.answers

            if self.val_ds is None:
                self.val_ds = VQAThroughputTestDataset(
                    self.data_dirs,
                    split="val",
                    transform=None,
                    max_len=self.max_len,
                    img_size=self.img_size,
                    tokenizer=self.tokenizer,
                    seq_length=self.seq_length,
                    selected_answers=self.selected_answers,
                    num_samples=self.num_samples,
                    num_classes=self.num_classes,
                )
            sample_info_msg += f"  Total training samples: {len(self.train_ds):8,d}"
            sample_info_msg += f"  Total validation samples: {len(self.val_ds):8,d}"

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = VQAThroughputTestDataset(
                self.data_dirs,
                split="test",
                transform=self.eval_transforms,
                max_len=self.max_len,
                img_size=self.img_size,
                tokenizer=self.tokenizer,
                seq_length=self.seq_length,
                selected_answers=self.selected_answers,
                num_samples=self.num_samples,
                num_classes=self.num_classes,
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
