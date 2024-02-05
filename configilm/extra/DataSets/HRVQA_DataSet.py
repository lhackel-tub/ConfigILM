import json
import random
from pathlib import Path
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Union

import numpy as np
import torch
from PIL import Image

from configilm.extra.DataSets.ClassificationVQADataset import ClassificationVQADataset
from configilm.extra.data_dir import resolve_data_dir_for_ds

# values based on train images of original split at 256 x 256
_means_256 = {"red": 0.4257, "green": 0.4435, "blue": 0.4239}
_stds_256 = {"red": 0.1335, "green": 0.1202, "blue": 0.1117}

# values based on train images of original split at 1024 x 1024
_means_1024 = {"red": 0.4255, "green": 0.4433, "blue": 0.4237}
_stds_1024 = {"red": 0.1398, "green": 0.1279, "blue": 0.1203}


def resolve_data_dir(
        data_dir: Optional[Mapping[str, Path]], allow_mock: bool = False, force_mock: bool = False
) -> Mapping[str, Path]:
    """
    Helper function that tries to resolve the correct directory
    for the HRVQA dataset.

    :param data_dir: Optional path to the data directory. If None, the default data
        directory will be used.

    :param allow_mock: allows mock data path to be returned

        :Default: False

    :param force_mock: only mock data path will be returned. Useful for debugging with
        small data or if the data is not downloaded yet.

        :Default: False
    """
    return resolve_data_dir_for_ds(
        dataset_name="hrvqa",
        data_dir_mapping=data_dir,
        allow_mock=allow_mock,
        force_mock=force_mock,
    )


def _get_question_answers(data_dirs: Mapping[str, Path], split: str) -> list[tuple[str, str, str, str]]:
    split_data_dir = data_dirs[f"{split}_data"]

    # load the question data
    q_file = split_data_dir / f"{split}_question.json"

    with open(q_file) as read_file:
        questions = json.load(read_file)["questions"]

    question_dict = {q["question_id"]: q for q in questions}

    data = []
    if split == "test":
        # no answers for test set so just return questions
        for q_id, q in question_dict.items():
            data.append((q["image_id"], q["question"], "", q["question_type"]))
        return data

    # load the answer data if not in test set
    a_file = split_data_dir / f"{split}_answer.json"
    with open(a_file) as read_file:
        answers = json.load(read_file)["annotations"]
    for a in answers:
        q_id = a["question_id"]
        assert q_id in question_dict, f"Question {q_id} not found in question file, but found in answer file."
        if q_id in question_dict:
            data.append(
                (
                    question_dict[q_id]["image_id"],
                    question_dict[q_id]["question"],
                    a["multiple_choice_answer"],
                    question_dict[q_id]["question_type"],
                )
            )
    return data


class HRVQADataSet(ClassificationVQADataset):
    def __init__(
            self,
            data_dirs: Mapping[str, Path],
            split: Optional[str] = None,
            transform: Optional[Callable] = None,
            max_len: Optional[int] = None,
            img_size: tuple = (3, 1024, 1024),
            selected_answers: Optional[list] = None,
            num_classes: Optional[int] = 1_000,
            tokenizer: Optional[Callable] = None,
            seq_length: int = 64,
            return_extras: bool = False,
            div_seed: Union[int, str] = 42,
            split_size: Union[float, int] = 0.5,
    ):
        assert split in {
            None,
            "train",
            "val",
            "val-div",
            "test-div",
            "test",
        }, f"Invalid split: {split}, expected one of: train, val, val-div, test-div, test"
        if isinstance(div_seed, str):
            div_seed = div_seed.lower()
        if split in {"val-div", "test-div"}:
            assert isinstance(div_seed, int) or div_seed == "repeat", \
                f"Invalid div_seed: {div_seed}, expected int or 'repeat'"
            if isinstance(split_size, float):
                assert 0 <= split_size <= 1, \
                    f"Invalid split_size: {split_size}, expected 0 <= split_size <= 1 for type float"
        self.div_seed = div_seed
        self.split_size = split_size
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
        assert img_size[0] == 3 and len(img_size) == 3, f"Invalid img_size: {img_size}, expected (3, height, width)"

    def split_names(self) -> set[str]:
        """
        Returns the names of the splits that are available for this dataset.

        :Note: This dataset has actually 5 splits: train, val, val-div, test-div, test. However, the val-div and
            test-div splits are just the val-split split into two parts. This is done to allow for a standard
            train/val/test splitting even though the original dataset does not have a test set with public answers.
            This is also the reason why test is not included in the return value of this method, as the answers for
            the test set are not public and therefore set to an empty string.
        """
        return {"train", "val"}

    def prepare_split(self, split: str) -> list:
        if split in {"train", "val", "test"}:
            return _get_question_answers(self.data_dirs, split)
        elif split in {"val-div", "test-div"}:
            # load val split now and then split later
            val_data = _get_question_answers(self.data_dirs, "val")
            # sort val_data by question
            val_data.sort(key=lambda x: x[1])
            samples_in_val_split = (
                self.split_size if isinstance(self.split_size, int) else int(len(val_data) * self.split_size)
            )
            if self.div_seed == "repeat":
                return val_data
            # save the current random state
            state = random.getstate()
            random.seed(self.div_seed)
            # shuffle the data
            random.shuffle(val_data)
            # recover the random state
            random.setstate(state)
            # return the data depending on the split parameter
            if split == "val-div":
                return val_data[:samples_in_val_split]
            else:
                return val_data[samples_in_val_split:]

    def load_image(self, key: str) -> torch.Tensor:
        img_path = self.data_dirs["images"] / f"{key}.png"
        img = Image.open(img_path).convert("RGB")
        tensor = torch.tensor(np.array(img)).permute(2, 0, 1)
        # resize image
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=self.img_size[1:], mode="bilinear", align_corners=False
        ).squeeze(0)
        return tensor
