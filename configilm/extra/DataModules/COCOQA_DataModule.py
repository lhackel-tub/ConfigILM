import warnings
from pathlib import Path
from typing import Callable
from typing import Mapping
from typing import Optional

from torchvision import transforms

from configilm.extra.CustomTorchClasses import MyGaussianNoise
from configilm.extra.DataModules.ClassificationVQADataModule import (
    ClassificationVQADataModule,
)
from configilm.extra.DataSets.COCOQA_DataSet import COCOQADataSet


class COCOQADataModule(ClassificationVQADataModule):
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
        This class implements the DataModule for the COCO-QA dataset.

        :param data_dirs: A dictionary containing the paths to the data directories.
            Should contain the keys "images", "train_data", and "test_data". The "images" directory
            should contain two subdirectories "train2014" and "val2014" with the images for the training
            and validation set, respectively. The "train_data" and "test_data" directories should contain
            for txt files "questions.txt", "answers.txt", "img_ids.txt", and "types.txt" each, following the
            COCO-QA dataset format.

        :param batch_size: The batch size to use for the dataloaders.

            :default: 16

        :param img_size: The size of the images.

            :default: (3, 120, 120)

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
        self.eval_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size[1], self.img_size[2]), antialias=True),
                # normalize?
            ]
        )

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        sample_info_msg = ""

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_ds = COCOQADataSet(
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

            self.val_ds = COCOQADataSet(
                self.data_dirs,
                split="test",
                transform=self.eval_transforms,
                max_len=self.max_len,
                img_size=self.img_size,
                tokenizer=self.tokenizer,
                seq_length=self.seq_length,
                selected_answers=self.selected_answers,
            )
            sample_info_msg += f"  Total training samples: {len(self.train_ds):8,d}"
            sample_info_msg += f"  Total validation samples: {len(self.val_ds):8,d}"
            warnings.warn("Validation and Test set are equal in this Dataset.")

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = COCOQADataSet(
                self.data_dirs,
                split="test",
                transform=self.eval_transforms,
                max_len=self.max_len,
                img_size=self.img_size,
                tokenizer=self.tokenizer,
                seq_length=self.seq_length,
                selected_answers=self.selected_answers,
            )
            sample_info_msg += f"  Total test samples: {len(self.test_ds):8,d}"
            warnings.warn("Validation and Test set are equal in this Dataset.")

        if stage == "predict":
            raise NotImplementedError("Predict stage not implemented")

        print(sample_info_msg)

    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()

    def test_dataloader(self):
        return super().test_dataloader()
