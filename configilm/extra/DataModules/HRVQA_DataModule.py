"""
Dataloader and Datamodule for RSVQA LR dataset.
"""
from pathlib import Path
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Union

from configilm.extra.DataModules.ClassificationVQADataModule import ClassificationVQADataModule
from configilm.extra.DataSets.HRVQA_DataSet import HRVQADataSet
from configilm.extra.DataSets.HRVQA_DataSet import _means_1024
from configilm.extra.DataSets.HRVQA_DataSet import _stds_1024
from configilm.extra._defaults import default_train_transform
from configilm.extra._defaults import default_transform


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

        assert isinstance(test_splitting_seed, int) or test_splitting_seed in ["repeat", None, ], (
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
                    division_seed = "Should Not Matter"
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
