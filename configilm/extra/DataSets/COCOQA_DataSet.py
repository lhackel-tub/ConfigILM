import json
import os
from os.path import isdir
from os.path import join
from pathlib import Path
from typing import Optional
from typing import Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from configilm.util import get_default_tokenizer
from configilm.util import huggingface_tokenize_and_pad
from configilm.util import Messages


def resolve_data_dir(
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
        data_dir = str(
            Path(__file__)
            .parent.parent.joinpath("mock_data")
            .joinpath("COCO-QA")
            .resolve(True)
        )
    if force_mock:
        Messages.warn("Forcing Mock data")
        data_dir = str(
            Path(__file__)
            .parent.parent.joinpath("mock_data")
            .joinpath("COCO-QA")
            .resolve(True)
        )

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
            self.tokenizer = get_default_tokenizer()
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
