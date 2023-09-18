import json
import pathlib
import random
from os.path import isdir
from os.path import isfile
from pathlib import Path
from typing import Optional
from typing import Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from configilm.util import get_default_tokenizer
from configilm.util import huggingface_tokenize_and_pad
from configilm.util import Messages

# values based on train images - of original split
# at 256 x 256
# _means = {
#     "red": 0.4257,
#     "green": 0.4435,
#     "blue": 0.4239,
#     "mono": 0.4310
# }
#
# _stds = {
#     "red": 0.1335,
#     "green": 0.1202,
#     "blue": 0.1117,
#     "mono": 0.1218
# }

# at 1024 x 1024
_means = {"red": 0.4255, "green": 0.4433, "blue": 0.4237, "mono": 0.4309}

_stds = {"red": 0.1398, "green": 0.1279, "blue": 0.1203, "mono": 0.1308}


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
    if data_dir in [None, "none", "None"]:
        Messages.warn("No data directory provided, trying to resolve")
        paths = [
            "/mnt/storagecube/data/datasets/HRVQA-1.0 release",  # MARS Storagecube
            "/media/storagecube/data/datasets/HRVQA-1.0 release",  # ERDE Storagecube
        ]
        for p in paths:
            if isdir(p):
                data_dir = p
                Messages.warn(f"Changing path to {data_dir}")
                break

    # using mock data if allowed and no other found or forced
    if data_dir in [None, "none", "None"] and allow_mock:
        Messages.warn("Mock data being used, no alternative available.")
        data_dir_p = pathlib.Path(__file__).parent.parent / "mock_data" / "HRVQA"
        data_dir = str(data_dir_p.resolve(True))
    if force_mock:
        Messages.warn("Forcing Mock data")
        data_dir_p = pathlib.Path(__file__).parent.parent / "mock_data" / "HRVQA"
        data_dir = str(data_dir_p.resolve(True))

    if data_dir is None:
        raise AssertionError("Could not resolve data directory")
    elif data_dir in ["none", "None"]:
        raise AssertionError("Could not resolve data directory")
    else:
        return data_dir


def select_answers(answers, number_of_answers: int = 1_000, use_tqdm: bool = False):
    # this dict will store as keys the answers and the values are the frequencies
    # they occur
    freq_dict = {}

    it = tqdm(answers, desc="Counting Answers") if use_tqdm else answers
    for a in it:

        answer_str = a["multiple_choice_answer"]

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


def _subsplit_qa(questions, answers, qa_in_split, sub_split, seed):
    no_samples_val = (
        qa_in_split
        if isinstance(qa_in_split, int)
        else (int(qa_in_split * len(questions)))
    )
    assert 0 <= no_samples_val <= len(questions), (
        f"Samples should be between 0 and {len(questions)}, but is {no_samples_val} "
        f"(specified as {qa_in_split})."
    )
    if no_samples_val == 0:
        Messages.warn(
            "There are zero (0) samples in your selected split configuration."
        )

    # get a random subset based on a seed
    # to make it reproducible in all contexts,
    # rebuild the random state afterward
    random_state = random.getstate()
    random.seed(seed)
    sample_set = set(random.sample(range(len(questions)), no_samples_val))
    random.setstate(random_state)
    if sub_split == "val":
        # only use samples from the sample set
        questions = [questions[i] for i in sample_set]
        answers = [answers[i] for i in sample_set]
    else:
        # only use samples that are not in the sample set
        inv_sample_set = set(range(len(questions))).difference(sample_set)
        questions = [questions[i] for i in inv_sample_set]
        answers = [answers[i] for i in inv_sample_set]

    return questions, answers


def _get_question_answers(
    split: Optional[str],
    root_dir: pathlib.Path,
    split_size: Union[int, float],
    split_seed,
):
    subsplit = None  # should never be relevant unless overwritten
    if split in ["train", "val", "test", None]:
        subsplit_qa = False
        if split == "test":
            q_path = root_dir / "jsons" / f"{split}_question.json"
            a_path = root_dir / "jsons" / f"{split}_answer.json"
            if isfile(q_path) and isfile(a_path):
                pass
            else:
                Messages.warn(
                    "Test json files don't exist. Changing split to 'val' and"
                    " assuming you want to repeat the val split for testing"
                )
                split = "val"
    else:
        # we want to subdivide the questions
        subsplit_qa = True
        # unless the seed is "repeat"
        if split_seed == "repeat":
            subsplit_qa = False
        # also we have to save what our actual real split was before overwriting
        # which is necessary to load the correct files
        subsplit = "val" if split == "val-div" else "test"
        split = "val"

    # read questions
    questions = []
    if split is not None:
        f_path = root_dir / "jsons" / f"{split}_question.json"
        with open(f_path.resolve()) as read_file:
            questions = json.load(read_file)["questions"]
    else:
        splits = ["train", "val"]
        for s in splits:
            f_path = root_dir / "jsons" / f"{s}_question.json"
            with open(f_path.resolve()) as read_file:
                questions += json.load(read_file)["questions"]
    questions = sorted(questions, key=lambda x: x["question_id"])

    # read answers
    answers = []
    if split is not None:
        f_path = root_dir / "jsons" / f"{split}_answer.json"
        with open(f_path.resolve()) as read_file:
            answers = json.load(read_file)["annotations"]
    else:
        splits = ["train", "val"]
        for s in splits:
            with open(root_dir / "jsons" / f"{s}_answer.json") as read_file:
                answers += json.load(read_file)["annotations"]
    answers = sorted(answers, key=lambda x: x["question_id"])

    # if val set has to be divided into val and test, do it now
    if subsplit_qa:
        questions, answers = _subsplit_qa(
            questions=questions,
            answers=answers,
            qa_in_split=split_size,
            sub_split=subsplit,
            seed=split_seed,
        )

    return questions, answers


class HRVQADataSet(Dataset):
    def __init__(
        self,
        root_dir: Union[Path, str],
        split: Optional[str] = None,
        transform=None,
        max_img_idx=None,
        img_size=(3, 1024, 1024),
        selected_answers=None,
        classes: int = 1_000,
        tokenizer=None,
        seq_length: int = 32,
        div_seed=None,
        split_size: Union[float, int] = 0.5,
    ):
        super().__init__()
        assert split in [
            None,
            "train",
            "val",
            "val-div",
            "test-div",
            "test",
        ], f"Split '{split}' not supported for HRVQA DataSet"

        assert img_size[0] in [1, 3], (
            f"HRVQA only supports 3 channel (RGB) or 1 "
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
        self.pre_transforms = transforms.Compose(
            [transforms.Resize(img_size[1:]), transforms.ToTensor()]
        )
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.max_img_idx = max_img_idx if max_img_idx is not None else -1
        self.classes = classes

        # changing the split for some cases
        # this is a convince feature e.g. for splitting
        self.questions, self.answers = _get_question_answers(
            split=split,
            root_dir=self.root_dir,
            split_size=split_size,
            split_seed=div_seed,
        )

        assert len(self.answers) == len(self.questions), (
            f"Number of questions ({len(self.questions)}) is not the same as number of"
            f" answers ({len(self.answers)})"
        )

        assert {x["question_id"] for x in self.answers} == {
            x["question_id"] for x in self.questions
        }, (
            "Sets of question and answers do not fit (not same question_ids in both "
            "sets)"
        )

        # restrict qs and as
        if 0 < self.max_img_idx < len(self.questions):
            self.questions = self.questions[:max_img_idx]
            self.answers = self.answers[:max_img_idx]

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
        question = self.questions[idx]
        answer = self.answers[idx]
        assert (
            question["question_id"] == answer["question_id"]
        ), f"ID mismatch for question and answer for index {idx}"
        img_path = self.root_dir / "images" / f'{question["image_id"]}.png'
        img = Image.open(img_path.resolve()).convert("RGB")
        img = self.pre_transforms(img)
        if self.transform:
            img = self.transform(img)
        if not self.is_rgb:
            img = img.mean(0).unsqueeze(0)

        # tokenize question
        question_ids = huggingface_tokenize_and_pad(
            tokenizer=self.tokenizer,
            string=question["question"],
            seq_length=self.seq_length,
        )
        label = self._to_labels(answer["multiple_choice_answer"])

        return img, question_ids, label
