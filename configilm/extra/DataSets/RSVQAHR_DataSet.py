import json
import pathlib
from pathlib import Path
from typing import Optional
from typing import Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from configilm.extra.data_dir import resolve_data_dir_for_ds
from configilm.util import Messages
from configilm.util import get_default_tokenizer
from configilm.util import huggingface_tokenize_and_pad

# values based on train images - of original split
_means = {"red": 0.4640, "green": 0.4682, "blue": 0.4376, "mono": 0.4566}
_stds = {"red": 0.1843, "green": 0.1740, "blue": 0.1656, "mono": 0.1764}


def resolve_data_dir(
        data_dir: Optional[str], allow_mock: bool = False, force_mock: bool = False
) -> str:
    """
    Helper function that tries to resolve the correct directory

    :param data_dir: current path that is suggested
    :param allow_mock: allows mock data path to be returned
    :param force_mock: only mock data path will be returned. Useful for debugging with
    small data
    :return: a valid dir to the dataset if data_dir was none, otherwise data_dir
    """
    return resolve_data_dir_for_ds(
        dataset_name="rsvqa-hr",
        data_dir=data_dir,
        allow_mock=allow_mock,
        force_mock=force_mock,
    )


def select_answers(answers, number_of_answers: int = 1_000, use_tqdm: bool = False):
    """
    Selects the most frequently present answers and returns them in order of frequency.

    :param answers: input list of answers
    :param number_of_answers: how many answers should be selected

        :Default: 1_000

    :param use_tqdm: Flag to use tqdm as progress bar

        :Default: False

    :return: most frequent answers (list of length number_of_answers), ordered by
        frequency
    """
    # this dict will store as keys the answers and the values are the frequencies
    # they occur
    freq_dict = {}

    it = tqdm(answers, desc="Counting Answers") if use_tqdm else answers
    for k in it:
        a = answers[k]
        answer_str = a["answer"]

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

    if len(answers) > 0:
        # print the percentage of used answers
        perc_answers = sum([x[1] for x in selected_answers]) / len(answers) * 100
        print(
            f"The {number_of_answers} most frequent answers cover about "
            f"{perc_answers:5.2f} "
            f"% of the total answers."
        )

    # return only the strings, not how often they appear
    return [x[0] for x in selected_answers]


def _get_question_answers(split: str, root_dir: pathlib.Path):
    def _remove_key_from_dict(d: dict, k: str):
        n = d.copy()
        n.pop(k)
        return n

    f_name = f"USGS_split_{split}_images.json"
    with open(root_dir.joinpath(f_name)) as read_file:
        images = json.load(read_file)["images"]
    qids = [x["questions_ids"] for x in images if x["active"]]
    del images
    qids_set = {x for sublist in qids for x in sublist}
    del qids

    f_name = f"USGS_split_{split}_questions.json"
    with open(root_dir.joinpath(f_name)) as read_file:
        questions = json.load(read_file)["questions"]
    questions = [
        {
            "question": x["question"],
            "type": x["type"],
            "img_id": x["img_id"],
            "id": x["id"],
        }
        for x in questions
        if x["id"] in qids_set
    ]
    q_dict = {x["id"]: _remove_key_from_dict(x, "id") for x in questions}
    del questions

    f_name = f"USGS_split_{split}_answers.json"
    with open(root_dir.joinpath(f_name)) as read_file:
        answers = json.load(read_file)["answers"]
    answers = [
        {"answer": x["answer"], "question_id": x["question_id"]}
        for x in answers
        if x["active"] and x["question_id"] in qids_set
    ]
    del qids_set
    a_dict = {
        x["question_id"]: _remove_key_from_dict(x, "question_id") for x in answers
    }
    del answers

    return q_dict, a_dict


def _quantize_answers(a_dict: dict):
    def _to_bucket(x: int):
        if x == 0:
            return "0m2"
        if 1 <= x <= 10:
            return "between 1m2 and 10m2"
        if 11 <= x <= 100:
            return "between 11m2 and 100m2"
        if 101 <= x <= 1_000:
            return "between 101m2 and 1000m2"
        if x > 1_000:
            return "more than 1000m2"
        raise ValueError("Buckets are only defined for non-negative values.")

    for k, v in a_dict.items():
        a_dict[k] = (
            {"answer": _to_bucket(int(v["answer"][:-2]))}
            if v["answer"].endswith("m2")
            else v
        )
    return a_dict


class RSVQAHRDataSet(Dataset):
    def __init__(
            self,
            root_dir: Union[Path, str],
            split: Optional[str] = None,
            transform=None,
            max_img_idx=None,
            img_size=(3, 256, 256),
            selected_answers=None,
            classes: int = 94,
            tokenizer=None,
            seq_length: int = 32,
            use_file_format: str = "tif",
            quantize_answers: bool = True,
    ):
        """
        :param root_dir: root directory to images and jsons folder

        :param split: "train", "val", "test_phili" or "test" or None for all

            :Default: None (loads all splits)

        :param transform: transformations to be applied to loaded images aside from
            scaling all bands to img_size.

            :Default: None

        :param max_img_idx: maximum number of images to load. If this number is higher
            than the images found in the csv, None or -1, all images will be loaded.

            :Default: None

        :param img_size: Size to which all channels will be scaled. Interpolation is
            applied bicubic before any transformation. Also selects if the returned
            images are RGB or grayscale based on the number of channels.

            :Default: (3, 256, 256)

        :param selected_answers: List of selected answers or None. If set to None,
            answers will be selected based on `classes` in order of frequency of the
            set.

            :Default: None

        :param classes: Number of classes (possible answers)

            :Default: 94

        :param tokenizer: Tokenizer to use for tokenization of input questions. Expects
            standard huggingface tokenizer. If not set, a default tokenizer will be
            used and a warning shown.

            :Default: None

        :param seq_length: Length of tokenized question. Will be caped to this as
            maximum and expanded to this if the question is too short. Includes start
            and end token.

            :Default: 32

        :param use_file_format: file type to use ("png" or "tif")

            :Default: "tif"

        :param quantize_answers: Flag, if answers should be quantized by magnitude as
            done in the original paper. This means that classes 1, 2, ... 10 are
            collected into one class, 11, 12, ... 100 into another one etc. Done for
            both count and area.

            :Default: True
        """
        super().__init__()
        assert split in [
            None,
            "train",
            "val",
            "test",
            "test_phili",
        ], f"Split '{split}' not supported for RSVQA-HR DataSet"

        assert img_size[0] in [1, 3], (
            f"RSVQA-HR only supports 3 channel (RGB) or 1 "
            f"channel (grayscale). {img_size[0]} channels "
            f"unsupported."
        )
        self.is_rgb = img_size[0] == 3

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
        if self.is_rgb:
            self.pre_transforms = transforms.Compose(
                [transforms.Resize(img_size[1:]), transforms.ToTensor()]
            )
        else:
            self.pre_transforms = transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.Resize(img_size[1:]),
                    transforms.ToTensor(),
                ]
            )
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.max_img_idx = max_img_idx if max_img_idx is not None else -1
        self.classes = classes
        self.use_file_format = use_file_format

        if self.split is not None:
            self.questions, self.answers = _get_question_answers(
                self.split, self.root_dir
            )
        else:
            self.questions, self.answers = dict(), dict()
            for s in ["train", "val", "test", "test_phili"]:
                q, a = _get_question_answers(s, self.root_dir)
                self.questions.update(q)
                self.answers.update(a)

        assert self.answers.keys() == self.questions.keys(), (
            "The IDs of questions and answers are not the same. Some questions/answers"
            "do not have corresponding answers/questions"
        )

        # restrict qs and as
        if 0 < self.max_img_idx < len(self.questions):
            allowed_keys = set(list(self.questions.keys())[:max_img_idx])
            self.questions = {
                k: v for k, v in self.questions.items() if k in allowed_keys
            }
            self.answers = {k: v for k, v in self.answers.items() if k in allowed_keys}
        self.qids = sorted(self.questions.keys())

        if quantize_answers:
            self.answers = _quantize_answers(self.answers)

        if selected_answers is None:
            self.selected_answers = select_answers(
                answers=self.answers, number_of_answers=self.classes
            )
        else:
            self.selected_answers = selected_answers

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
        question = self.questions[self.qids[idx]]
        answer = self.answers[self.qids[idx]]

        # tokenize question
        question_ids = huggingface_tokenize_and_pad(
            tokenizer=self.tokenizer,
            string=question["question"],
            seq_length=self.seq_length,
        )
        label = self._to_labels(answer["answer"])

        img_path = (
                self.root_dir
                / "Images"
                / "Data"
                / f'{question["img_id"]}.{self.use_file_format}'
        )
        img = Image.open(img_path.resolve()).convert("RGB")
        img = self.pre_transforms(img)
        if self.transform:
            img = self.transform(img)
        if not self.is_rgb:
            img = img.mean(0).unsqueeze(0)

        return img, question_ids, label
