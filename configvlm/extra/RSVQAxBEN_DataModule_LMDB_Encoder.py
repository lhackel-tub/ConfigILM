"""
Dataloader and Datamodule for RSVQAxBEN dataset. Files can be requested by contacting
the author.
Original Paper of Dataset:
https://rsvqa.sylvainlobry.com/IGARSS21.pdf
Based on Image Data from:
https://arxiv.org/abs/2105.07921
https://bigearth.net/
"""
import json
import os
import warnings
from datetime import datetime
from os.path import join
from pathlib import Path
from time import time
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import psutil
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer

from configvlm.extra.BEN_lmdb_utils import band_combi_to_mean_std
from configvlm.extra.BEN_lmdb_utils import BENLMDBReader
from configvlm.extra.CustomTorchClasses import MyGaussianNoise
from configvlm.extra.CustomTorchClasses import MyRotateTransform
from configvlm.util import huggingface_tokenize_and_pad
from configvlm.util import Messages


def select_answers_from_qa_pairs(qa_pairs, number_of_answers=1000):
    # this dict will store as keys the answers and the values are the frequencies
    # they occur
    freq_dict = {}

    for qa_pair in tqdm(qa_pairs, desc="Counting Answers"):

        answer_str = qa_pair["answer"]

        # update the dictionary
        if answer_str not in freq_dict:
            freq_dict[answer_str] = 1
        else:
            freq_dict[answer_str] += 1

    # sort the dictionary by the most common
    # so that position 0 contains the most frequent word
    answers_by_appearence = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)

    if number_of_answers > len(answers_by_appearence):
        Messages.warn(
            f"There are fewer possible answers then requested ({number_of_answers} "
            f"requested, but {len(answers_by_appearence)} found)."
        )
        answers_by_appearence += [("INVALID", 0)] * (
            number_of_answers - len(answers_by_appearence)
        )

    selected_answers = answers_by_appearence[:number_of_answers]

    # print the percentage of used answers
    perc_answers = sum([x[1] for x in selected_answers]) / len(qa_pairs) * 100
    print(
        f"The {number_of_answers} most frequent answers cover about "
        f"{perc_answers:5.2f} "
        f"% of the total answers."
    )

    # return only the strings, not how often they appear
    return [x[0] for x in selected_answers]


class RSVQAxBENDataSet(Dataset):
    max_cache_size = 0

    def __init__(
        self,
        root_dir="./",
        split: Optional[str] = None,
        transform=None,
        max_img_idx=None,
        img_size=(12, 120, 120),
        selected_answers=None,
        classes=1000,
        tokenizer=None,
        seq_length=32,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.lmdb_dir = os.path.join(self.root_dir, "BigEarthNetEncoded.lmdb")
        self.transform = transform
        self.image_size = img_size
        self.classes = classes
        assert img_size[0] in [2, 3, 4, 10, 12], (
            "Image Channels have to be "
            "2 (Sentinel-1), "
            "3 (RGB), "
            "4 (10m Sentinel-2), "
            "10 (10m + 20m Sentinel-2) or "
            "12 (10m + 20m Sentinel-2 + 10m Sentinel-1) "
            "but was " + f"{img_size[0]}"
        )

        if tokenizer is None:
            Messages.warn(
                "No tokenizer was provided, using BertTokenizer (uncased). This may "
                "result in very bad performance if the used network expected other "
                "tokens."
            )
            # get path relative to this script, not relative to the calling main script
            # TODO: make this more flexible
            default_tokenizer = (
                Path(__file__)
                .parent.parent.joinpath(
                    "huggingface_tokenizers", "bert-base-uncased.tok"
                )
                .resolve(True)
            )
            self.tokenizer = BertTokenizer.from_pretrained(default_tokenizer.resolve())
        else:
            self.tokenizer = tokenizer
        self.seq_length = seq_length

        print(f"Loading split RSVQAxBEN data for {split}...")
        # check that enough ram (in GB) is available, otherwise use subset
        subset = "_subset" if psutil.virtual_memory().total / 1024**3 < 32 else ""
        self.qa_pairs = {}
        if split is not None:
            f_name = f"RSVQAxBEN_QA_{split}{subset}.json"
            with open(join(self.root_dir, "VQA_RSVQAxBEN", f_name)) as read_file:
                self.qa_pairs.update(json.load(read_file))
        else:
            splits = ["train", "val", "test"]
            for s in splits:
                f_name = f"RSVQAxBEN_QA_{s}{subset}.json"
                with open(join(self.root_dir, "VQA_RSVQAxBEN", f_name)) as read_file:
                    self.qa_pairs.update(json.load(read_file))

        # sort list for reproducibility
        self.qa_values = [self.qa_pairs[key] for key in sorted(self.qa_pairs)]
        del self.qa_pairs
        print(f"    {len(self.qa_values):12,d} QA-pairs indexed")
        if (
            max_img_idx is not None
            and max_img_idx < len(self.qa_values)
            and max_img_idx != -1
        ):
            self.qa_values = self.qa_values[:max_img_idx]

        print(f"    {len(self.qa_values):12,d} QA-pairs in reduced data set")

        # select answers
        if split == "train":
            self.selected_answers = select_answers_from_qa_pairs(
                qa_pairs=self.qa_values, number_of_answers=self.classes
            )
        else:
            self.selected_answers = selected_answers

        self._split_qa()

        self.BENLoader = BENLMDBReader(
            lmdb_dir=self.lmdb_dir,
            label_type="new",
            image_size=self.image_size,
            bands=self.image_size[0],
        )

    def _split_qa(self):
        # make a lookup for index -> question
        # the full set contains ~8.6 m questions, but < 250 000 unique ones
        # we can save a lot of memory this way
        self.idx_to_question = np.array(list({x["question"] for x in self.qa_values}))
        # temporary lookup question -> index.
        # Otherwise, conversion to index would be very slow
        q2idx = {q: i for i, q in enumerate(self.idx_to_question)}
        q, a, n, t = [], [], [], []
        for i, d in tqdm(
            enumerate(self.qa_values),
            desc="Converting to NP arrays",
            total=len(self.qa_values),
        ):
            q.append(q2idx[d["question"]])
            a.append(d["answer"])
            n.append(d["S2_name"])
            t.append(d["type"])
            self.qa_values[i] = None

        del self.qa_values
        self.names = np.asarray(n)
        self.types = np.asarray(t)
        self.answers = np.asarray(a)
        self.questions = np.asarray(q)

    def _to_labels(self, labels):
        label = torch.zeros(self.classes)
        try:
            label_idx = self.selected_answers.index(labels)
            label[label_idx] = 1
        except ValueError:
            # label not in list, return empty vector
            pass
        except AttributeError:
            pass
        return label

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        if idx > len(self):
            warnings.warn(f"Index {idx} > {len(self)}, using modulo")
            idx = idx % len(self)

        qa_pair = {
            "type": self.types[idx],
            "question": self.idx_to_question[self.questions[idx]],
            "answer": self.answers[idx],
            "S2_name": self.names[idx],
        }
        key = qa_pair["S2_name"]
        # get (& write) image from (& to) LMDB
        # get image from database
        # we have to copy, as the image in imdb is not writeable,
        # which is a problem in .to_tensor()
        img, labels = self.BENLoader[key]

        if self.transform:
            img = self.transform(img)

        label = self._to_labels(qa_pair["answer"])

        # tokenize question
        question_ids = huggingface_tokenize_and_pad(
            tokenizer=self.tokenizer,
            string=qa_pair["question"],
            seq_length=self.seq_length,
        )

        return img, question_ids, label


class RSVQAxBENDataModule(pl.LightningDataModule):
    train_ds: Union[None, RSVQAxBENDataSet] = None
    val_ds: Union[None, RSVQAxBENDataSet] = None
    test_ds: Union[None, RSVQAxBENDataSet] = None
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
        seq_length=32,
        selected_answers=None,
        pin_memory=None,
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
        self.max_img_idx = (
            max_img_idx if max_img_idx is None or max_img_idx > 0 else None
        )
        self.img_size = (12, 120, 120) if img_size is None else img_size
        self.shuffle = shuffle
        if self.shuffle is not None:
            Messages.hint(
                f"Shuffle was set to {self.shuffle}. This is not recommended for most "
                f"configuration. Use shuffle=None (default) for recommended "
                f"configuration."
            )
        self.selected_answers = selected_answers

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
        # self.transform = None
        self.pin_memory = torch.cuda.device_count() > 0
        self.pin_memory = self.pin_memory if pin_memory is None else pin_memory
        Messages.hint(
            f"pin_memory set to {pin_memory}"
            f"{' ' if pin_memory is None else ' via overwrite'}"
        )

        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        print(f"({datetime.now().strftime('%H:%M:%S')}) Datamodule setup called")
        sample_info_msg = ""
        t0 = time()

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.train_ds is None:
                self.train_ds = RSVQAxBENDataSet(
                    self.data_dir,
                    split="train",
                    transform=self.train_transform,
                    max_img_idx=self.max_img_idx,
                    img_size=self.img_size,
                    tokenizer=self.tokenizer,
                    seq_length=self.seq_length,
                )
            if self.selected_answers is None:
                self.selected_answers = self.train_ds.selected_answers

            if self.val_ds is None:
                self.val_ds = RSVQAxBENDataSet(
                    self.data_dir,
                    split="val",
                    transform=self.transform,
                    max_img_idx=self.max_img_idx,
                    img_size=self.img_size,
                    tokenizer=self.tokenizer,
                    seq_length=self.seq_length,
                    selected_answers=self.selected_answers,
                )
            sample_info_msg += f"  Total training samples: {len(self.train_ds):8,d}"
            sample_info_msg += f"  Total validation samples: {len(self.val_ds):8,d}"

        # Assign test dataset for use in dataloader(s)
        if (stage == "test" or stage is None) and self.test_ds is None:
            self.test_ds = RSVQAxBENDataSet(
                self.data_dir,
                split="test",
                transform=self.transform,
                max_img_idx=self.max_img_idx,
                img_size=self.img_size,
                tokenizer=self.tokenizer,
                seq_length=self.seq_length,
                selected_answers=self.selected_answers,
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
