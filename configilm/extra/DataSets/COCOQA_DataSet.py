import os
from os.path import join
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Union

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from configilm.extra.data_dir import resolve_data_dir_for_ds
from configilm.extra.DataSets.ClassificationVQADataset import ClassificationVQADataset


def resolve_data_dir(
    data_dir: Optional[Mapping[str, Path]], allow_mock: bool = False, force_mock: bool = False
) -> Mapping[str, Union[str, Path]]:
    """
    Helper function that tries to resolve the correct directory

    :param data_dir: current path that is suggested

    :param allow_mock: if True, mock data will be used if no real data is found

        :Default: False
    :param force_mock: if True, only mock data will be used

        :Default: False
    :return: a dict with all paths to the data
    """
    return resolve_data_dir_for_ds("cocoqa", data_dir, allow_mock, force_mock)


def _txts_to_dict(base_dir: str):
    # collect all .txt files in dir
    txt_files = [join(base_dir, x) for x in os.listdir(base_dir) if x.endswith(".txt")]
    data: Dict = {
        "answers": [],
        "img_ids": [],
        "questions": [],
        "types": [],
    }
    # read all files
    for f_name in sorted(txt_files):
        key = Path(f_name).stem
        assert key in data, f"Unknown file {f_name}"
        with open(join(base_dir, f_name)) as f:
            data[key] = f.readlines()
    # remove \n from all strings
    data = {k: [x.strip() for x in v] for k, v in data.items()}
    # correct img_ids to match image names
    split = "train" if "train" in Path(base_dir).stem else "val"
    data["img_ids"] = [f"COCO_{split}2014_{x:>012}.jpg" for x in data["img_ids"]]
    # zip all lists together to one list of 4 item tuples
    return list(zip(data["img_ids"], data["questions"], data["answers"], data["types"]))


class COCOQADataSet(ClassificationVQADataset):
    def __init__(
        self,
        data_dirs: Mapping[str, Path],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_len: Optional[int] = None,
        img_size: tuple = (3, 120, 120),
        selected_answers: Optional[list] = None,
        num_classes: Optional[int] = 430,
        tokenizer: Optional[Callable] = None,
        seq_length: int = 64,
        return_extras: bool = False,
    ):
        """
        This class implements the COCO-QA dataset. It is a subclass of
        ClassificationVQADataset and provides some dataset specific functionality.

        :param data_dirs: A mapping from file key to file path. The file key is
            used to identify the function of the file. For example, the key
            "questions.txt" is used to identify the file that contains the
            questions. The file path can be either a string or a Path object.
            Required keys are "images", "train_data" and "test_data". The "_data"
            keys each point to a directory that contains the data files which are
            named "questions.txt", "answers.txt", "img_ids.txt" and "types.txt".

        :param split: The name of the split to use. Can be either "train" or
            "test". If None is provided, all splits will be used.

            :default: None

        :param transform: A callable that is used to transform the images after
            loading them. If None is provided, no transformation is applied.

            :default: None

        :param max_len: The maximum number of qa-pairs to use. If None or -1 is
            provided, all qa-pairs are used.

            :default: None

        :param img_size: The size of the images.

            :default: (3, 120, 120)

        :param selected_answers: A list of answers that should be used. If None
            is provided, the num_classes most common answers are used. If
            selected_answers is not None, num_classes is ignored.

            :default: None

        :param num_classes: The number of classes to use. Only used if
            selected_answers is None. If set to None, all answers are used.

            :default: 430

        :param tokenizer: A callable that is used to tokenize the questions. If
            set to None, the default tokenizer (from configilm.util) is used.

            :default: None

        :param seq_length: The maximum length of the tokenized questions.

                :default: 64

        :param return_extras: If True, the dataset will return the type of the
            question in addition to the image, question and answer.

            :default: False
        """
        print(f"Loading COCOQA data for {split}...")
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
        self.convert = transforms.ToTensor()

    def prepare_split(self, split: str):
        if split == "train":
            path = Path(self.data_dirs["train_data"]).resolve()
            return _txts_to_dict(str(path))
        else:
            path = Path(self.data_dirs["test_data"]).resolve()
            return _txts_to_dict(str(path))

    def load_image(self, key: str) -> torch.Tensor:
        img_split = "train2014" if "train" in key else "val2014"
        img_dir = join(self.data_dirs["images"], img_split, key)
        img = Image.open(img_dir).convert("RGB")
        tensor = self.convert(img)
        # for some reason, HEIGHT and WIDTH are swapped in ToTensor() ... sometimes
        # FIXME: axis are only sometimes swapped
        # tensor = tensor.swapaxes(1, 2)
        tensor = F.interpolate(
            tensor.unsqueeze(dim=0),
            self.img_size[-2:],
            mode="bilinear",
            align_corners=True,
        ).squeeze(dim=0)
        return tensor

    def split_names(self) -> set[str]:
        return {"train", "test"}
