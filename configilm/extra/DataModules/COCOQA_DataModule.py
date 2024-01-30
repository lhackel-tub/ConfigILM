import warnings
from pathlib import Path
from time import time
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Union

from torchvision import transforms

from configilm.extra.CustomTorchClasses import MyGaussianNoise
from configilm.extra.DataModules.ClassificationVQADataModule import ClassificationVQADataModule
from configilm.extra.DataSets.COCOQA_DataSet import COCOQADataSet


class COCOQADataModule(ClassificationVQADataModule):
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
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size[1], self.img_size[2]), antialias=True),
                # normalize?
            ]
        )

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        sample_info_msg = ""
        t0 = time()

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
                self.selected_answers.extend(
                    ["INVALID"]
                    * (self.train_ds.num_classes - len(self.train_ds.answers))
                )

            self.val_ds = COCOQADataSet(
                self.data_dirs,
                split="test",
                transform=self.transform,
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
                transform=self.transform,
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
