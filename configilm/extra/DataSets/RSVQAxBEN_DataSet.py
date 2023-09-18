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
from os.path import join
from typing import Optional

import numpy as np
import psutil
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from configilm.extra.BEN_lmdb_utils import BENLMDBReader
from configilm.util import get_default_tokenizer
from configilm.util import huggingface_tokenize_and_pad
from configilm.util import Messages


def select_answers_from_qa_pairs(
    qa_pairs, number_of_answers: int = 1_000, use_tqdm: bool = False
):
    # this dict will store as keys the answers and the values are the frequencies
    # they occur
    freq_dict = {}

    it = tqdm(qa_pairs, desc="Counting Answers") if use_tqdm else qa_pairs
    for qa_pair in it:

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

    if len(qa_pairs) > 0:
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
            self.tokenizer = get_default_tokenizer()
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
