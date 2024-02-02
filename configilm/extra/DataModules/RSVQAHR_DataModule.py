"""
Dataloader and Datamodule for RSVQA HR dataset.
"""
from pathlib import Path
from typing import Callable
from typing import Mapping
from typing import Optional

from configilm.extra._defaults import default_train_transform
from configilm.extra._defaults import default_transform
from configilm.extra.DataModules.ClassificationVQADataModule import ClassificationVQADataModule
from configilm.extra.DataSets.RSVQAHR_DataSet import _means
from configilm.extra.DataSets.RSVQAHR_DataSet import _stds
from configilm.extra.DataSets.RSVQAHR_DataSet import RSVQAHRDataSet


class RSVQAHRDataModule(ClassificationVQADataModule):
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
        use_phili_test: bool = False,
        use_file_format: str = "tif",
        quantize_answers: bool = True,
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

        mean = [_means["red"], _means["green"], _means["blue"]]
        std = [_stds["red"], _stds["green"], _stds["blue"]]

        self.train_transforms = default_train_transform(
            img_size=(self.img_size[1], self.img_size[2]), mean=mean, std=std
        )
        self.eval_transforms = default_transform(img_size=(self.img_size[1], self.img_size[2]), mean=mean, std=std)

        self.use_phili_test = use_phili_test
        self.use_file_format = use_file_format
        self.quantize_answers = quantize_answers

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
        sample_info_msg = ""

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.train_ds is None:
                self.train_ds = RSVQAHRDataSet(
                    self.data_dirs,
                    split="train",
                    transform=self.train_transforms,
                    max_len=self.max_len,
                    img_size=self.img_size,
                    tokenizer=self.tokenizer,
                    seq_length=self.seq_length,
                    use_file_format=self.use_file_format,
                    quantize_answers=self.quantize_answers,
                )
            if self.selected_answers is None:
                self.selected_answers = self.train_ds.answers

            if self.val_ds is None:
                self.val_ds = RSVQAHRDataSet(
                    self.data_dirs,
                    split="val",
                    transform=self.eval_transforms,
                    max_len=self.max_len,
                    img_size=self.img_size,
                    tokenizer=self.tokenizer,
                    seq_length=self.seq_length,
                    selected_answers=self.selected_answers,
                    use_file_format=self.use_file_format,
                    quantize_answers=self.quantize_answers,
                )
            sample_info_msg += f"  Total training samples: {len(self.train_ds):8,d}"
            sample_info_msg += f"  Total validation samples: {len(self.val_ds):8,d}"

        # Assign test dataset for use in dataloader(s)
        if (stage == "test" or stage is None) and self.test_ds is None:
            self.test_ds = RSVQAHRDataSet(
                self.data_dirs,
                split="test_phili" if self.use_phili_test else "test",
                transform=self.eval_transforms,
                max_len=self.max_len,
                img_size=self.img_size,
                tokenizer=self.tokenizer,
                seq_length=self.seq_length,
                selected_answers=self.selected_answers,
                use_file_format=self.use_file_format,
                quantize_answers=self.quantize_answers,
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
