"""
Dataloader and Datamodule for BigEarthNet dataset. Files can be requested by contacting
the author.
Original Paper of Image Data:
https://arxiv.org/abs/2105.07921
https://bigearth.net/
"""
from datetime import datetime
from pathlib import Path
from time import time
from typing import Optional
from typing import Union

from configilm.extra.DataModules.BEN_DataModule import BENDataModule
from configilm.extra.DataSets.BEN2_DataSet import BEN2DataSet


class BEN2DataModule(BENDataModule):
    def __init__(
        self,
        batch_size=16,
        data_dir: Union[str, Path] = "./",
        img_size=None,
        num_workers_dataloader=None,
        max_img_idx=None,
        shuffle=None,
        new_label_file: Union[str, Path, None] = None,
    ):
        if img_size is not None and len(img_size) != 3:
            raise ValueError(
                f"Expected image_size with 3 dimensions (HxWxC) or None but got "
                f"{len(img_size)} dimensions instead"
            )
        super().__init__(
            batch_size=batch_size,
            data_dir=data_dir,
            img_size=img_size,
            num_workers_dataloader=num_workers_dataloader,
            max_img_idx=max_img_idx,
            shuffle=shuffle,
        )
        self.new_label_file = new_label_file

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        print(f"({datetime.now().strftime('%H:%M:%S')}) Datamodule setup called")
        sample_info_msg = ""
        t0 = time()

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_ds = BEN2DataSet(
                root_dir=self.data_dir,
                split="train",
                transform=self.train_transform,
                max_img_idx=self.max_img_idx,
                img_size=self.img_size,
                new_label_file=self.new_label_file,
            )

            self.val_ds = BEN2DataSet(
                root_dir=self.data_dir,
                split="val",
                transform=self.transform,
                max_img_idx=self.max_img_idx,
                img_size=self.img_size,
                new_label_file=self.new_label_file,
            )
            sample_info_msg += f"  Total training samples: {len(self.train_ds):8,d}"
            sample_info_msg += f"  Total validation samples: {len(self.val_ds):8,d}"

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = BEN2DataSet(
                root_dir=self.data_dir,
                split="test",
                transform=self.transform,
                max_img_idx=self.max_img_idx,
                img_size=self.img_size,
                new_label_file=self.new_label_file,
            )
            sample_info_msg += f"  Total test samples: {len(self.test_ds):8,d}"

        if stage == "predict":
            raise NotImplementedError("Predict stage not implemented")

        print(f"setup took {time() - t0:.2f} seconds")
        print(sample_info_msg)
