from collections import Counter
from pathlib import Path
from typing import Callable
from typing import List
from typing import Mapping
from typing import Optional
from warnings import warn

import torch
from torch.utils.data import Dataset

from configilm.util import get_default_tokenizer
from configilm.util import huggingface_tokenize_and_pad


class ClassificationVQADataset(Dataset):
    def __init__(
        self,
        data_dirs: Mapping[str, Path],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_len: Optional[int] = None,
        img_size: tuple = (3, 120, 120),
        selected_answers: Optional[list] = None,
        num_classes: Optional[int] = 1000,
        tokenizer: Optional[Callable] = None,
        seq_length: int = 64,
        return_extras: bool = False,
    ):
        """
        This class is a base class for datasets that are used for classification
        of visual question answering. It provides some basic functionality that
        is shared between different datasets. It is not intended to be used
        directly, but rather must be subclassed.

        :param data_dirs: A mapping from file key to file path. The file key is
            used to identify the function of the file. For example, the key
            "questions.txt" could be used to identify the file that contains the
            questions. The file path can be either a string or a Path object.
        :param split: The name of the split to use. This is dataset specific. If
            None is provided, the dataset is expected to provide a method called
            split_names that returns a set of strings, each string being the name
            of a split. All splits will be used in this case. If
            split_names is not implemented, the default implementation will
            return {"train", "val", "test"}.

            :default: None

        :param transform: A callable that is used to transform the images after
            loading them. This is dataset specific. If None is provided, no
            transformation is applied.

            :default: None

        :param max_len: The maximum number of qa-pairs to use. If None or -1 is
            provided, all qa-pairs are used.

            :default: None

        :param img_size: The size of the images. This is dataset specific.

            :default: (3, 120, 120)

        :param selected_answers: A list of answers that should be used. If None
            is provided, the num_classes most common answers are used. If
            selected_answers is not None, num_classes is ignored.

            :default: None

        :param num_classes: The number of classes to use. Only used if
            selected_answers is None. If set to None, all answers are used.

            :default: 1000

        :param tokenizer: A callable that is used to tokenize the questions. If
            set to None, the default tokenizer (from configilm.util) is used.

            :default: None

        :param seq_length: The maximum length of the tokenized questions.

                :default: 64

        :param return_extras: If set to True, the __getitem__ method will return
            a tuple of (image, question_ids, answer, *extras). The extras are
            dataset specific and are provided by the dataset implementation. They
            are the additional elements of the tuple that is returned by the
            prepare_split method at index 3 and higher. If set to False, the
            __getitem__ method will return a tuple of (image, question_ids,
            answer).

            :default: False
        """
        super().__init__()
        if selected_answers is not None:
            # ignore num_classes
            num_classes = None
        self.data_dirs = data_dirs
        self.data_dirs = {k: Path(v) for k, v in self.data_dirs.items()}
        self.transform = transform
        assert len(img_size) == 3, f"Invalid img_size: {img_size}, expected (channels, height, width)"
        self.img_size = img_size
        self.seq_length = seq_length
        self.return_extras = return_extras

        if split is not None:
            self.qa_data = self.prepare_split(split)
        else:
            self.qa_data = []
            for s in self.split_names():
                self.qa_data += self.prepare_split(s)

        # sort list for reproducibility, sort by question (alphabetically)
        # this assumes that the qa_data is a list of tuples, where the question
        # is at index 1
        self.qa_data = sorted(self.qa_data, key=lambda x: x[1])
        print(f"    {len(self.qa_data):8,d} QA-pairs indexed")

        # limit the number of qa-pairs if requested
        if max_len is not None and max_len < len(self.qa_data) and max_len != -1:
            self.qa_data = self.qa_data[:max_len]
        print(f"    {len(self.qa_data):8,d} QA-pairs used")

        if selected_answers is None:
            # filter to only use num_classes most common answers
            # this assumes that the qa_data is a list of tuples, where the answer
            # is at index 2
            answer_stats = Counter([x[2] for x in self.qa_data])
            selected_answers = sorted(answer_stats.keys(), key=lambda x: answer_stats[x])[:num_classes]
        else:
            # override num_classes if selected_answers is provided
            num_classes = len(selected_answers)
        self.answers = selected_answers
        # "INVALID" is used to indicate that the answer is not in the self.answers list
        # add this string to the end of the list until the list has the correct length
        assert num_classes is not None, "num_classes should have been set at this point, manually or automatically"
        self.answers.extend(["INVALID"] * (num_classes - len(self.answers)))
        self.num_classes = num_classes
        assert (
            len(self.answers) == self.num_classes
        ), f"Number of answers ({len(self.answers)}) is not equal to num_classes ({self.num_classes})"

        if tokenizer is None:
            warn(
                "No tokenizer was provided, using BertTokenizer (uncased). "
                "This may result in very bad performance if the used network "
                "expected other tokens"
            )
            self.tokenizer = get_default_tokenizer()
        else:
            self.tokenizer = tokenizer

    def split_names(self) -> set[str]:
        """
        Returns the names of the splits that are available for this dataset. The
        default implementation returns {"train", "val", "test"}. If you want to
        use different names, you should override this method.

        :return: A set of strings, each string being the name of a split
        """
        return {"train", "val", "test"}

    def prepare_split(self, split: str) -> list:
        """
        This method should return a list of tuples, where each tuple contains
        the following elements:

        - The key of the image at index 0
        - The question at index 1
        - The answer at index 2
        - additional information at index 3 and higher

        :param split: The name of the split to prepare

        :return: A list of tuples, each tuple containing the elements described
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    def load_image(self, key: str) -> torch.Tensor:
        """
        This method should load the image with the given name and return it as
        a tensor.

        :param key: The name of the image to load

        :return: The image as a tensor
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    def _answers_to_tensor(self, answers: List[str]) -> torch.Tensor:
        """
        This method converts the given list of answers to a tensor from the self.answers
        list. If the answer is not in the self.answers list, it is ignored.

        :param answers: The answers to convert to a tensor

        :return: The tensor containing one-hot encoded answers

        :hint: If only answers that are not in the self.answers list are provided,
            the tensor will be empty (all zeros).
        """
        answer_tensor = torch.zeros(self.num_classes)
        for a in answers:
            if a in self.answers:
                answer_tensor[self.answers.index(a)] = 1
        return answer_tensor

    def __getitem__(self, idx):
        qa_pair = self.qa_data[idx]

        # get image
        img = self.load_image(qa_pair[0]).to(torch.float32)
        if self.transform is not None:
            img = self.transform(img)
        assert img.shape == self.img_size, f"Image shape is {img.shape}, expected {self.img_size}"

        # tokenize question
        question_ids = huggingface_tokenize_and_pad(
            tokenizer=self.tokenizer,
            string=qa_pair[1],
            seq_length=self.seq_length,
        )
        assert (
            len(question_ids) == self.seq_length
        ), f"Question length is {len(question_ids)}, expected {self.seq_length}"

        # convert answer to tensor
        # note: this assumes that the qa_pair is a tuple of length 3, where the
        # answer is at index 2
        # the answer is a list of strings, where each string is an answer, therefore we
        # need wrap the single answer in a list
        tmp_answer = qa_pair[2]
        if not isinstance(tmp_answer, list):
            # answers are sometimes provided as a single string, sometimes as a list
            # of strings
            # if it is a single string, we need to wrap it in a list
            tmp_answer = [tmp_answer]
        answer = self._answers_to_tensor(tmp_answer)

        if self.return_extras:
            return img, question_ids, answer, *qa_pair[3:]
        else:
            return img, question_ids, answer

    def __len__(self):
        return len(self.qa_data)
