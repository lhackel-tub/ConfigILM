"""
Dataloader and Datamodule for BigEarthNet dataset. Files can be requested by contacting
the author.
Original Paper of Image Data:
https://arxiv.org/abs/2105.07921
https://bigearth.net/
"""
import csv
import os
from datetime import datetime
from time import time
from typing import Optional
from typing import Union

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from configvlm.extra.BEN_lmdb_utils import band_combi_to_mean_std
from configvlm.extra.BEN_lmdb_utils import ben19_list_to_onehot
from configvlm.extra.BEN_lmdb_utils import BENLMDBReader
from configvlm.extra.CustomTorchClasses import MyGaussianNoise
from configvlm.extra.CustomTorchClasses import MyRotateTransform
from configvlm.util import Messages


class BENDataSet(Dataset):
    avail_chan_configs = {
        2: "Sentinel-1",
        3: "RGB",
        4: "10m Sentinel-2",
        10: "10m + 20m Sentinel-2",
        12: "10m + 20m Sentinel-2 + 10m Sentinel-1",
    }

    @classmethod
    def get_available_channel_configurations(cls):
        print("Available channel configurations are:")
        for c, m in cls.avail_chan_configs.items():
            print(f"    {c:>3} -> {m}")

    def __init__(
        self,
        root_dir="./",
        split: Optional[str] = None,
        transform=None,
        max_img_idx=None,
        img_size=(12, 120, 120),
    ):
        super().__init__()
        self.root_dir = root_dir
        self.lmdb_dir = os.path.join(self.root_dir, "BigEarthNetEncoded.lmdb")
        self.transform = transform
        self.image_size = img_size
        if img_size[0] not in self.avail_chan_configs.keys():
            BENDataSet.get_available_channel_configurations()
            raise AssertionError(f"{img_size[0]} is not a valid channel configuration.")

        self.read_channels = img_size[0]

        print(f"Loading BEN data for {split}...")
        if split is not None:
            with open(os.path.join(self.root_dir, f"{split}.csv")) as f:
                reader = csv.reader(f)
                patches = list(reader)
        else:
            splits = ["train", "val", "test"]
            patches = []
            for s in splits:
                with open(os.path.join(self.root_dir, s + ".csv")) as f:
                    reader = csv.reader(f)
                    patches += list(reader)

        # lines get read as arrays -> flatten to one dimension
        self.patches = [x[0] for x in patches]
        # sort list for reproducibility
        self.patches.sort()
        print(f"    {len(self.patches)} patches indexed")
        if (
            max_img_idx is not None
            and max_img_idx < len(self.patches)
            and max_img_idx != -1
        ):
            self.patches = self.patches[:max_img_idx]

        print(f"    {len(self.patches)} filtered patches indexed")
        self.BENLoader = BENLMDBReader(
            lmdb_dir=self.lmdb_dir,
            label_type="new",
            image_size=self.image_size,
            bands=self.image_size[0],
        )

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        key = self.patches[idx]

        # get (& write) image from (& to) LMDB
        # get image from database
        # we have to copy, as the image in imdb is not writeable,
        # which is a problem in .to_tensor()
        img, labels = self.BENLoader[key]

        if img is None:
            print(f"Cannot load {key} from database")
            raise ValueError
        if self.transform:
            img = self.transform(img)

        label_list = labels
        labels = ben19_list_to_onehot(labels)

        assert sum(labels) == len(set(label_list)), f"Label creation failed for {key}"
        return img, labels


class BENDataModule(pl.LightningDataModule):
    num_classes = 19
    train_ds: Union[None, BENDataSet] = None
    val_ds: Union[None, BENDataSet] = None
    test_ds: Union[None, BENDataSet] = None

    def __init__(
        self,
        batch_size=16,
        data_dir: str = "./",
        img_size=None,
        num_workers_dataloader=None,
        max_img_idx=None,
        shuffle=None,
    ):
        if img_size is not None and len(img_size) != 3:
            raise ValueError(
                f"Expected image_size with 3 dimensions (HxWxC) or None but got "
                f"{len(img_size)} dimensions instead"
            )
        super().__init__()
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
        self.max_img_idx = max_img_idx
        self.img_size = (12, 120, 120) if img_size is None else img_size
        self.shuffle = shuffle
        if self.shuffle is not None:
            Messages.hint(
                f"Shuffle was set to {self.shuffle}. This is not recommended for most "
                f"configuration. Use shuffle=None (default) for recommended "
                f"configuration."
            )

        # get mean and std
        ben_mean, ben_std = band_combi_to_mean_std(self.img_size[0])

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size[1], self.img_size[2])),
                MyGaussianNoise(20),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                MyRotateTransform([0, 90, 180, 270]),
                transforms.Normalize(ben_mean, ben_std),
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size[1], self.img_size[2])),
                transforms.Normalize(ben_mean, ben_std),
            ]
        )
        self.pin_memory = torch.cuda.device_count() > 0

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        print(f"({datetime.now().strftime('%H:%M:%S')}) Datamodule setup called")
        sample_info_msg = ""
        t0 = time()

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_ds = BENDataSet(
                self.data_dir,
                split="train",
                transform=self.train_transform,
                max_img_idx=self.max_img_idx,
                img_size=self.img_size,
            )

            self.val_ds = BENDataSet(
                self.data_dir,
                split="val",
                transform=self.transform,
                max_img_idx=self.max_img_idx,
                img_size=self.img_size,
            )
            sample_info_msg += f"  Total training samples: {len(self.train_ds):8,d}"
            sample_info_msg += f"  Total validation samples: {len(self.val_ds):8,d}"

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = BENDataSet(
                self.data_dir,
                split="test",
                transform=self.transform,
                max_img_idx=self.max_img_idx,
                img_size=self.img_size,
            )
            sample_info_msg += f"  Total test samples: {len(self.test_ds):8,d}"

        if stage == "predict":
            raise NotImplementedError("Predict stage not implemented")

        print(f"setup took {time() - t0:.2f} seconds")
        print(sample_info_msg)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )
