"""
Datamodule for updated BigEarthNet dataset. Files can be requested by contacting
the author.
Original Paper of Image Data:
https://arxiv.org/abs/2105.07921
https://bigearth.net/
"""
from datetime import datetime
from pathlib import Path
from time import time
from typing import Mapping
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
        print_infos: bool = False,
        dataset_kwargs: Optional[Mapping] = None,
    ):
        """
        Initializes a pytorch lightning data module.

        :param batch_size: batch size for data loaders

            :Default: 16

        :param data_dir: base data directory for lmdb and csv files

            :Default: ./

        :param img_size: image size `(c, h, w)` in accordance with `BENDataSet`

            :Default: None uses default of dataset

        :param num_workers_dataloader: number of workers used for data loading

            :Default: #CPU_cores/2

        :param max_img_idx: maximum number of images to load per split. If this number
            is higher than the images found in the csv, None or -1, all images will be
            loaded.

            :Default: None

        :param shuffle: Flag if dataset should be shuffled. If set to None, only train
            will be shuffled and validation and test won't.

            :Default: None

        :param new_label_file: Path to parquet file with new label information or None,
            if the file is called "labels.parquet" and located in the data_dir.

            :Default: None

        :param print_infos: Flag, if additional information during setup() and reading
            should be printed (e.g. number of workers detected, number of images loaded)

            :Default: False

        :param dataset_kwargs: Other keyword arguments to pass to the dataset during
            creation.
        """
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
            dataset_kwargs=dataset_kwargs,
            print_infos=print_infos,
        )
        self.new_label_file = new_label_file
        self.print_infos = print_infos

    def setup(self, stage: Optional[str] = None):
        if self.print_infos:
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
                **self.ds_kwargs,
            )

            self.val_ds = BEN2DataSet(
                root_dir=self.data_dir,
                split="val",
                transform=self.transform,
                max_img_idx=self.max_img_idx,
                img_size=self.img_size,
                new_label_file=self.new_label_file,
                **self.ds_kwargs,
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
                **self.ds_kwargs,
            )
            sample_info_msg += f"  Total test samples: {len(self.test_ds):8,d}"

        if stage == "predict":
            raise NotImplementedError("Predict stage not implemented")

        if self.print_infos:
            print(f"setup took {time() - t0:.2f} seconds")
            print(sample_info_msg)
