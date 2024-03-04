from pathlib import Path
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Union

import torch

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
    return resolve_data_dir_for_ds("throughput_test", data_dir, allow_mock, force_mock)


class FakeTokenizer:
    def __init__(self):
        pass

    def convert_tokens_to_ids(self, tokens):
        if tokens == "[CLS]":
            return [100]
        elif tokens == "[SEP]":
            return [101]
        elif tokens == "[PAD]":
            return [0]
        else:
            return [42]

    def tokenize(self, string):
        return [42] * (len(string) // 4)

    def __call__(self, string):
        return self.tokenize(string)


class VQAThroughputTestDataset(ClassificationVQADataset):
    def __init__(
        self,
        data_dirs: Mapping[str, Path],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_len: Optional[int] = None,
        img_size: tuple = (3, 256, 256),
        selected_answers: Optional[list] = None,
        num_classes: Optional[int] = 1000,
        tokenizer: Optional[Callable] = None,
        seq_length: int = 64,
        return_extras: bool = False,
        num_samples: int = 1000,
    ):
        """
        This class implements the ThroughputTest dataset. It is a subclass of
        ClassificationVQADataset and provides some dataset specific functionality.

        :param data_dirs: A mapping from file key to file path. Is ignored, as
            the dataset does not use any real files but included for compatibility.

        :param split: The name of the split to use. Is ignored, as the dataset
            does not use any real splits but included for compatibility with other

            :default: None

        :param transform: A callable that is used to transform the images after
            loading them. Is ignored, as the dataset does not use any real images but
            included for compatibility with other datasets.

            :default: None

        :param max_len: The maximum number of qa-pairs to use. If None or -1 is
            provided, all qa-pairs are used. "all" is defined in num_samples.

            :default: None

        :param img_size: The size of the images.

            :default: (3, 120, 120)

        :param selected_answers: A list of answers that should be used. If None
            is provided, the num_classes most common answers are used. If
            selected_answers is not None, num_classes is ignored. Only used to infer
            the number of classes.

            :default: None

        :param num_classes: The number of classes to use. Only used if
            selected_answers is None. If set to None, all answers are used.

            :default: 430

        :param tokenizer: A callable that is used to tokenize the questions. Is
            ignored, as the dataset does not use any real questions but included for
            compatibility with other datasets.

            :default: None

        :param seq_length: The exact length of the tokenized questions.

                :default: 64

        :param return_extras: Is ignored, as the dataset does not use any real
            data but included for compatibility with other datasets.

            :default: False

        :param num_samples: The number of samples to simulate.

            :default: 1000
        """
        print(f"Loading ThroughputTest data for {split}...")
        assert (
            num_classes is not None or selected_answers is not None
        ), "Either num_classes or selected_answers must be provided."
        self.num_samples = num_samples
        assert self.num_samples > 0, "num_samples must be greater than 0"
        assert img_size[2] > 0, "Invalid width, must be > 0"
        assert img_size[1] > 0, "Invalid height, must be > 0"
        assert img_size[0] > 0, "Invalid number of channels, must be > 0"
        self.sample_img = torch.rand(img_size)
        self.qa_pair = (
            "sample/key",  # key
            "Sample Question?",  # question
            "Sample answer!",  # answer
        )
        super().__init__(
            data_dirs=data_dirs,
            split=split,
            transform=transform,
            max_len=max_len,
            img_size=img_size,
            selected_answers=selected_answers,
            num_classes=num_classes,
            tokenizer=FakeTokenizer(),  # no tokenizer needed
            seq_length=seq_length,
            return_extras=return_extras,
        )

    def prepare_split(self, split: str):
        return [self.qa_pair for _ in range(self.num_samples)]

    def load_image(self, key: str) -> torch.Tensor:
        return self.sample_img

    def split_names(self) -> set[str]:
        return {"train", "val", "test"}
