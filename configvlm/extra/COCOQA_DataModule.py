import json
import os
import warnings
from datetime import datetime
from os.path import isdir
from os.path import join
from pathlib import Path
from time import time
from typing import List
from typing import Optional
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer

import configvlm
from configvlm.extra.CustomTorchClasses import MyGaussianNoise
from configvlm.util import huggingface_tokenize_and_pad
from configvlm.util import Messages


def resolve_cocoqa_data_dir(
    data_dir: Union[str, None], allow_mock: bool = False, force_mock: bool = False
) -> str:
    """
    Helper function that tries to resolve the correct directory
    :param data_dir: current path that is suggested
    :return: a valid dir to the dataset if data_dir was none, otherwise data_dir
    """
    if data_dir in [None, "none", "None"]:
        Messages.warn("No data directory provided, trying to resolve")

        paths = [
            "",  # shared memory
            "/home/lhackel/Documents/datasets/COCO-QA/",  # laptop
            "",  # MARS
            "",  # ERDE
            "",  # last resort: storagecube (MARS)
            "",  # (ERDE)
            "",  # eolab legacy
        ]
        for p in paths:
            if isdir(p):
                data_dir = p
                Messages.warn(f"Changing path to {data_dir}")
                break

    # using mock data if allowed and no other found or forced
    if data_dir in [None, "none", "None"] and allow_mock:
        Messages.warn("Mock data being used, no alternative available.")
        data_dir = str(Path(__file__).parent.joinpath("mock_data").resolve(True))
    if force_mock:
        Messages.warn("Forcing Mock data")
        data_dir = str(Path(__file__).parent.joinpath("mock_data").resolve(True))

    if data_dir is None:
        raise AssertionError("Could not resolve data directory")
    elif data_dir in ["none", "None"]:
        raise AssertionError("Could not resolve data directory")
    else:
        return data_dir


class COCOQADataSet(Dataset):
    num_classes = 430

    def __init__(
        self,
        root_dir="./",
        split: Optional[str] = None,
        transform=None,
        max_img_idx=None,
        img_size=(3, 120, 120),
        tokenizer=None,
        seq_length=64,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = img_size

        if tokenizer is None:
            Messages.warn(
                "No tokenizer was provided, using BertTokenizer (uncased). "
                "This may result in very bad performance if the used network "
                "expected other tokens"
            )
            # get path relative to this script, not relative to the calling main script
            default_tokenizer = (
                Path(configvlm.__file__)
                .parent.joinpath("huggingface_tokenizers", "bert-base-uncased.tok")
                .resolve(True)
            )
            self.tokenizer = BertTokenizer.from_pretrained(default_tokenizer)
        else:
            self.tokenizer = tokenizer
        self.seq_length = seq_length

        print(f"Loading COCOQA data for {split}...")
        self.qa_pairs = {}
        if split is not None:
            f_name = f"COCO-QA_QA_{split}.json"
            with open(join(self.root_dir, f_name)) as read_file:
                self.qa_pairs.update(json.load(read_file))
        else:
            splits = ["train", "test"]
            for s in splits:
                f_name = f"COCO-QA_QA_{s}.json"
                with open(join(self.root_dir, f_name)) as read_file:
                    qa_pairs = json.load(read_file)
                    qa_pairs = {f"{k}_{s}": v for k, v in qa_pairs.items()}
                    self.qa_pairs.update(qa_pairs)

        #  sort list for reproducibility
        self.qa_values = [self.qa_pairs[key] for key in sorted(self.qa_pairs)]
        del self.qa_pairs
        print(f"    {len(self.qa_values):6,d} QA-pairs indexed")
        if (
            max_img_idx is not None
            and max_img_idx < len(self.qa_values)
            and max_img_idx != -1
        ):
            self.qa_values = self.qa_values[:max_img_idx]

        image_name_mapping = os.listdir(join(self.root_dir, "images"))
        self.image_name_mapping = {int(x[-14:-4]): x for x in image_name_mapping}

        print(f"    {len(self.qa_values):6,d} QA-pairs in reduced data set")

        self.answers = sorted(list({x["answer"] for x in self.qa_values}))
        assert self.num_classes >= len(
            self.answers
        ), "There are more different answers than classes, this should not happen"

    def _load_image(self, name):
        name = self.image_name_mapping[int(name)]
        convert = transforms.ToTensor()

        img = Image.open(join(self.root_dir, "images", name)).convert("RGB")
        tensor = convert(img)
        # for some reason, HEIGHT and WIDTH are swapped in ToTensor() ... sometimes
        # FIXME: axis are only sometimes swapped
        # tensor = tensor.swapaxes(1, 2)
        tensor = F.interpolate(
            tensor.unsqueeze(dim=0),
            self.image_size[-2:],
            mode="bicubic",
            align_corners=True,
        ).squeeze(dim=0)
        return tensor

    def _answer_to_onehot(self, answer):
        label = torch.zeros(self.num_classes)
        try:
            label_idx = self.answers.index(answer)
            label[label_idx] = 1
        except ValueError:
            # label not in list, return empty vector
            pass
        except AttributeError:
            pass
        return label

    def __len__(self):
        return len(self.qa_values)

    def __getitem__(self, idx):
        data = self.qa_values[idx]

        # get image
        # get answer
        img = self._load_image(data["img_id"])
        question_ids = huggingface_tokenize_and_pad(
            tokenizer=self.tokenizer,
            string=data["question"],
            seq_length=self.seq_length,
        )
        answer = self._answer_to_onehot(data["answer"])
        assert answer.sum() == 1, "Answer not valid, sum not 1"

        if self.transform:
            img = self.transform(img)

        assert img.shape == self.image_size or self.transform is None, (
            f"Shape mismatch for index {idx}\n "
            f"Is {img.shape} but should be {self.image_size}"
        )
        return img, question_ids, answer


class COCOQADataModule(pl.LightningDataModule):
    train_ds: Union[None, COCOQADataSet] = None
    val_ds: Union[None, COCOQADataSet] = None
    test_ds: Union[None, COCOQADataSet] = None
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
        seq_length=64,
    ):
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
        self.img_size = (3, 120, 120) if img_size is None else img_size
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        if self.shuffle is not None:
            Messages.hint(
                f"Shuffle was set to {self.shuffle}. This is not recommended for most "
                f"configuration. Use shuffle=None (default) for recommended "
                f"configuration."
            )

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size[1], self.img_size[2])),
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
                transforms.Resize((self.img_size[1], self.img_size[2])),
                # normalize?
            ]
        )
        # self.transform = None
        self.pin_memory = torch.cuda.device_count() > 0

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        print(f"({datetime.now().strftime('%H:%M:%S')}) Datamodule setup called")
        sample_info_msg = ""
        t0 = time()

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_ds = COCOQADataSet(
                self.data_dir,
                split="train",
                transform=self.train_transform,
                max_img_idx=self.max_img_idx,
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
                self.data_dir,
                split="test",
                transform=self.transform,
                max_img_idx=self.max_img_idx,
                img_size=self.img_size,
                tokenizer=self.tokenizer,
                seq_length=self.seq_length,
            )
            sample_info_msg += f"  Total training samples: {len(self.train_ds):8,d}"
            sample_info_msg += f"  Total validation samples: {len(self.val_ds):8,d}"
            warnings.warn("Validation and Test set are equal in this Dataset.")

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = COCOQADataSet(
                self.data_dir,
                split="test",
                transform=self.transform,
                max_img_idx=self.max_img_idx,
                img_size=self.img_size,
                tokenizer=self.tokenizer,
                seq_length=self.seq_length,
            )
            sample_info_msg += f"  Total test samples: {len(self.test_ds):8,d}"
            warnings.warn("Validation and Test set are equal in this Dataset.")

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
