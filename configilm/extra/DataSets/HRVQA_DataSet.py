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

from configilm.extra.data_dir import resolve_data_dir_for_ds
from configilm.extra.DataSets.ClassificationVQADataset import ClassificationVQADataset

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
        """
        This class implements the HRVQA dataset. It is a subclass of
        ClassificationVQADataset and is used to load the HRVQA dataset.

        :param data_dirs: A mapping of strings to Path objects that contains the
            paths to the data directories. It should contain the following keys:
            "images", "train_data", "val_data", "test_data". The "_data" keys
            should point to the directory that contains the question and answer
            json files. Each directory should contain the following files:
            "{split}_question.json" and "{split}_answer.json".

        :param split: The split of the dataset to load. It can be one of the following:
            "train", "val", "val-div", "test-div", "test". If None, the train and val
            splits will be loaded. The "val-div" and "test-div" splits are just the
            val-split split into two parts. This is done to allow for a standard
            train/val/test splitting even though the original dataset does not have a
            test set with public answers. This is also the reason why test is not included
            in the return value of the split_names method.

            :default: None

        :param transform: A callable that is used to transform the images after
            loading them. If None, no transformation will be applied.

            :default: None

        :param max_len: The maximum number of qa-pairs to use. If None or -1 is
            provided, all qa-pairs are used.

            :default: None

        :param img_size: The size of the images.

                :default: (3, 1024, 1024)

        :param selected_answers: A list of answers that should be used. If None
            is provided, the num_classes most common answers are used. If
            selected_answers is not None, num_classes is ignored.

            :default: None

        :param num_classes: The number of classes to use. Only used if
            selected_answers is None. If set to None, all answers are used.

            :default: 1_000

        :param tokenizer: A callable that is used to tokenize the questions. If
            set to None, the default tokenizer (from configilm.util) is used.

            :default: None

        :param seq_length: The maximum length of the tokenized questions. If the
            tokenized question is longer than seq_length, it will be truncated.

            :default: 64

        :param return_extras: If True, the dataset will return the type of the
            question in addition to the image, question and answer.

            :default: False

        :param div_seed: The seed to use for the split of the val-div and test-div
            splits. If set to "repeat", the split will be the same full val split for
            both val-div and test-div. If set to an integer, the split will be different
            every time the dataset is loaded and the seed will be used to initialize
            the random number generator. The state of the random number generator
            will be saved before the split and restored after the split to ensure
            reproducibility independent of the global random state and also that the
            global random state is not affected by the split.

            :default: 42

        :param split_size: The size of the val-div and test-div splits. If set to a
            float, it should be a value between 0 and 1 and will be interpreted as the
            fraction of the val split to use for the val-div. The rest of the val split
            will be used for the test-div. If set to an integer, it will be interpreted
            as the number of samples to use for the val-div. The rest of the val split
            will be used for the test-div. If div_seed is set to "repeat", the split
            will be the same (full val split) for both val-div and test-div.

            :default: 0.5
        """
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
            assert (
                isinstance(div_seed, int) or div_seed == "repeat"
            ), f"Invalid div_seed: {div_seed}, expected int or 'repeat'"
            if isinstance(split_size, float):
                assert (
                    0 <= split_size <= 1
                ), f"Invalid split_size: {split_size}, expected 0 <= split_size <= 1 for type float"
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
        else:
            raise ValueError(f"Split {split} unknown.")

    def load_image(self, key: str) -> torch.Tensor:
        img_path = self.data_dirs["images"] / f"{key}.png"
        img = Image.open(img_path).convert("RGB")
        tensor = torch.tensor(np.array(img)).permute(2, 0, 1)
        # resize image
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=self.img_size[1:], mode="bilinear", align_corners=False
        ).squeeze(0)
        return tensor
