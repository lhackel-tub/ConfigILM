import json
from pathlib import Path
from typing import Callable
from typing import Mapping
from typing import Optional

import numpy as np
import torch
from PIL import Image

from configilm.extra.data_dir import resolve_data_dir_for_ds
from configilm.extra.DataSets.ClassificationVQADataset import ClassificationVQADataset

# values based on train images - of original split
_means = {"red": 0.4640, "green": 0.4682, "blue": 0.4376, "mono": 0.4566}
_stds = {"red": 0.1843, "green": 0.1740, "blue": 0.1656, "mono": 0.1764}


def resolve_data_dir(
    data_dir: Optional[Mapping[str, Path]], allow_mock: bool = False, force_mock: bool = False
) -> Mapping[str, Path]:
    """
    Helper function that tries to resolve the correct directory
    for the RSVQA-HR dataset.

    :param data_dir: Optional path to the data directory. If None, the default data
        directory will be used.

    :param allow_mock: allows mock data path to be returned

        :Default: False

    :param force_mock: only mock data path will be returned. Useful for debugging with
        small data or if the data is not downloaded yet.

        :Default: False
    """
    return resolve_data_dir_for_ds(
        dataset_name="rsvqa-hr",
        data_dir_mapping=data_dir,
        allow_mock=allow_mock,
        force_mock=force_mock,
    )


def _get_question_answers(
    data_dirs: Mapping[str, Path], split: str, quantize_answers: bool = True
) -> list[tuple[str, str, str, str]]:
    split_data_dir = data_dirs[f"{split}_data"]

    # get all question ids
    f_name = f"USGS_split_{split}_images.json"
    with open((split_data_dir / f_name).resolve()) as read_file:
        images = json.load(read_file)["images"]
    # keep only active images
    qids = [x["questions_ids"] for x in images if x["active"]]
    del images
    # make a set of all question ids for fast lookup
    qids_set = {x for sublist in qids for x in sublist}
    del qids

    f_name = f"USGS_split_{split}_questions.json"
    # load all questions
    with open((split_data_dir / f_name).resolve()) as read_file:
        questions = json.load(read_file)["questions"]
    # keep only active questions
    questions = {
        x["id"]: {
            "question": x["question"],
            "type": x["type"],
            "img_id": x["img_id"],
        }
        for x in questions
        if x["id"] in qids_set
    }

    f_name = f"USGS_split_{split}_answers.json"
    # load all questions
    with open((split_data_dir / f_name).resolve()) as read_file:
        answers = json.load(read_file)["answers"]
    # keep only active questions
    answers = {x["question_id"]: x["answer"] for x in answers if x["active"] and x["question_id"] in qids_set}

    if quantize_answers:
        answers = _quantize_answers(answers)

    # merge questions and answers
    qa_data = []
    for qid, q in questions.items():
        qa_data.append((q["img_id"], q["question"], answers[qid], q["type"]))

    return qa_data


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
        # remove the m2 from the answer and convert to int, then put in respective bucket
        if v.endswith("m2"):
            input_value = int(v[:-2])
            a_dict[k] = _to_bucket(input_value)
    return a_dict


class RSVQAHRDataSet(ClassificationVQADataset):
    def __init__(
        self,
        data_dirs: Mapping[str, Path],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_len: Optional[int] = None,
        img_size: tuple = (3, 256, 256),
        selected_answers: Optional[list] = None,
        num_classes: Optional[int] = 94,
        tokenizer: Optional[Callable] = None,
        seq_length: int = 64,
        return_extras: bool = False,
        use_file_format: str = "tif",
        quantize_answers: bool = True,
    ):
        """
        This class implements the RSVQA-HR dataset. It is a subclass of
        ClassificationVQADataset and provides some dataset specific functionality.

        :param data_dirs: A dictionary containing the paths to the different data directories.
            It should contain the following keys:
            - images: Path to the directory containing the images.
            - train_data: Path to the directory containing the training data.
            - val_data: Path to the directory containing the validation data.
            - test_data: Path to the directory containing the test data.
            - test_phili_data: Path to the directory containing the test data for the Philadelphia test split.

        :param split: The name of the split to use. Can be either "train", "val", "test" or "test_phili".
            If None is provided, all splits will be used.

            :default: None

        :param transform: A callable that is used to transform the images after loading them.
            If None is provided, no transformation is applied.

            :default: None

        :param max_len: The maximum number of qa-pairs to use. If None or -1 is provided, all qa-pairs will be used.

            :default: None

        :param img_size: The size of the images.

            :default: (3, 256, 256)

        :param selected_answers: A list of answers that should be used. If None is provided, the num_classes most common
             answers are used.

            :default: None

        :param num_classes: The number of classes to use. Only used if selected_answers is None. If set to None, all
            answers are used.

            :default: 94

        :param tokenizer: A callable that is used to tokenize the questions. If
            set to None, the default tokenizer (from configilm.util) is used.

            :default: None

        :param seq_length: The maximum length of the tokenized questions.

            :default: 64

        :param return_extras: If True, the dataset will return the type of the
            question in addition to the image, question and answer.

            :default: False

        :param use_file_format: The file format of the images. Can be either "tif" or "png".

            :default: "tif"

        :param quantize_answers: If True, the answers for area questions will be quantized into 5 buckets:
            0m2, between 1m2 and 10m2, between 11m2 and 100m2, between 101m2 and 1000m2 and more than 1000m2.

            :default: True
        """
        self.use_file_format = use_file_format
        self.quantize_answers = quantize_answers
        assert img_size[0] == 3, "RSVQA-HR only supports RGB images."
        super().__init__(
            data_dirs=data_dirs,
            split=split,
            transform=transform,
            max_len=max_len,
            img_size=img_size,
            selected_answers=selected_answers,
            num_classes=num_classes,
            tokenizer=tokenizer,
            seq_length=seq_length,
            return_extras=return_extras,
        )

    def prepare_split(self, split: str) -> list[tuple[str, str, str, str]]:
        return _get_question_answers(data_dirs=self.data_dirs, split=split, quantize_answers=self.quantize_answers)

    def split_names(self) -> set[str]:
        return {"train", "val", "test", "test_phili"}

    def load_image(self, key: str) -> torch.Tensor:
        img_path = self.data_dirs["images"] / f"{key}.{self.use_file_format}"
        img = Image.open(img_path).convert("RGB")
        tensor = torch.tensor(np.array(img)).permute(2, 0, 1)
        # resize image
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=self.img_size[1:], mode="bilinear", align_corners=False
        ).squeeze(0)
        return tensor
