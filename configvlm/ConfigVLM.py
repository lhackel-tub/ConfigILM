"""
Framework for combining vision and language models
"""

__author__ = "Leonard Hackel"
__credits__ = ["Leonard Hackel"]
__maintainer__ = "Leonard Hackel"
__email__ = "l.hackel@tu-berlin.de"

import timm
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)
from typing import Sequence, Union, Callable
import torch
from torch import nn

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from collections import OrderedDict
from os import listdir
from os.path import isdir, join
import warnings
from requests.exceptions import HTTPError  # type: ignore
from requests.exceptions import ReadTimeout  # type: ignore
from appdirs import user_cache_dir


def _available_hf_models(base_path: Path):
    """
    Searches a local path and returns names of all available huggingface models
    :param base_path: path to search in
    :return: list of all available models
    """
    all_models = []
    Path(base_path).mkdir(parents=True, exist_ok=True)
    # get all huggingface usernames -> folders
    users = listdir(base_path)
    users = [u for u in users if isdir(join(base_path, u))]
    for u in users:
        # get all sub-folders of this folder -> models of this user
        models = listdir(join(base_path, u))
        models = [m for m in models if isdir(join(base_path, u, m))]
        for m in models:
            all_models += [f"{u}/{m}"]
        if len(models) < 1:
            # model name is not user/model but just model
            all_models += [u]
    return all_models


def get_hf_model(
    model_name: str,
    load_pretrained_if_available: bool = False,
):
    """
    Loads a huggingface model including tokenizer. Searches local files first.
    If the model is not available locally, first download it to the local
    directory, cache it and then load it.
    :param model_name: huggingface model name
    :param load_pretrained_if_available: load the model including pretrained
                                         weights, not just the architecture
    :param save_directory: local directory to use for search and caching
    :return: tokenizer and model
    :raises: Connection error if no Internet connection can be established
             (only if model is not found locally)
    :raises: HTTP error if no name matches the one given (locally or on
             huggingface hub)
    """
    save_directory = Path(user_cache_dir(appname="configvlm")).joinpath(
        "pretrained_models", "huggingface_models"
    )

    if model_name not in _available_hf_models(save_directory):
        # warn that it is not available
        warnings.warn(f"Model '{model_name}' not available. Trying to download...\n")

        # try to download
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            model.save_pretrained(join(save_directory, model_name))
            tokenizer.save_pretrained(join(save_directory, model_name))
        except HTTPError:
            raise HTTPError(
                f"Model '{model_name}' could not be fetched. Please check spelling."
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model '{model_name}' could not be fetched. "
                f"Network is down and file not cached."
            )
        except ReadTimeout:
            raise ReadTimeout(
                f"Model '{model_name}' could not be fetched. "
                f"Timeout and file not cached."
            )

    # Model is available or was made available
    model_path = join(save_directory, model_name)

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, config=config)

    if load_pretrained_if_available:
        model = transformers.AutoModel.from_pretrained(
            model_path, config=config, local_files_only=True
        )
        warnings.warn("Tokenizer was initialized pretrained")
    else:
        model = transformers.AutoModel.from_config(config=config)

    return tokenizer, model


def get_timm_model(model_name: str, timm_kwargs: dict):
    while True:
        try:
            encoder = timm.create_model(model_name, **timm_kwargs)
            break
        except TypeError as t:
            # get keyword that failed and drop it
            failed_kw = t.args[0].split("'")[1]
            del timm_kwargs[failed_kw]
            warnings.warn(
                f"Keyword '{failed_kw}' unknown. Trying to ignore and restart creation."
            )
    return encoder


class VLMType(Enum):
    VISION_CLASSIFICATION = 0
    VQA_CLASSIFICATION = 1


@dataclass
class VLMConfiguration:
    timm_model_name: str
    hf_model_name: Union[None, str] = None

    # values with default
    image_size: int = 120
    channels: int = 3
    classes: int = 10
    class_names: Union[None, Sequence[str]] = None
    network_type: VLMType = VLMType.VISION_CLASSIFICATION

    # currently only used for VQA_CLASSIFICATION
    visual_features_out: int = 512
    fusion_in: int = 512
    fusion_hidden: int = 256
    v_dropout_rate: float = 0.25
    t_dropout_rate: float = 0.25
    fusion_dropout_rate: float = 0.25
    fusion_method: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.mul
    fusion_activation: Callable[[torch.Tensor], torch.Tensor] = nn.Tanh()

    # specific for timm lib
    drop_rate: Union[float, None] = 0.2
    # specific for hf
    use_pooler_output: bool = True
    max_sequence_length: int = 32

    # pretrained models
    load_timm_if_available: bool = False
    load_hf_if_available: bool = True


class ConfigVLM(nn.Module):
    def __init__(self, config: VLMConfiguration):
        super().__init__()
        self.config = config
        if self.config.class_names is None:
            self.config.class_names = [str(i) for i in range(self.config.classes)]

        if config.network_type in [
            VLMType.VQA_CLASSIFICATION,
            VLMType.VISION_CLASSIFICATION,
        ]:
            # keyword arguments as expected by Timm lib
            timm_kwargs = {
                "num_classes": self.config.visual_features_out
                if config.network_type in [VLMType.VQA_CLASSIFICATION]
                else self.config.classes,
                "img_size": self.config.image_size,
                "in_chans": self.config.channels,
                "drop_rate": self.config.drop_rate,
                "drop_path_rate": self.config.drop_rate,
                "pretrained": self.config.load_timm_if_available,
            }

            # create timm_model
            self.vision_encoder = get_timm_model(config.timm_model_name, timm_kwargs)

        if config.network_type in [VLMType.VQA_CLASSIFICATION]:
            # create huggingface model
            assert (
                config.hf_model_name is not None
            ), "Requesting huggingface model but not specifying which"
            self.tokenizer, self.text_encoder = get_hf_model(
                config.hf_model_name,
                load_pretrained_if_available=config.load_hf_if_available,
            )

        if config.network_type == VLMType.VQA_CLASSIFICATION:
            # create fusion layer
            # get number of out-features
            text_features_out = self.text_encoder.config.hidden_size
            if not self.config.use_pooler_output:
                text_features_out *= self.config.max_sequence_length
            else:
                # check if text encoder has a pooler
                if not hasattr(self.text_encoder, "pooler"):
                    warnings.warn(
                        f"Text encoder '{config.hf_model_name}' has no pooler, "
                        f"changing use_pooler_output to False"
                    )
                    text_features_out *= self.config.max_sequence_length
                    self.config.use_pooler_output = False

            # map text encoder -> fusion_in
            self.text_linear = nn.Linear(text_features_out, self.config.fusion_in)
            # map vision encoder -> fusion_in
            self.visual_linear = nn.Linear(
                self.config.visual_features_out, self.config.fusion_in
            )

            self.dropout_v = torch.nn.Dropout(self.config.v_dropout_rate)
            self.dropout_q = torch.nn.Dropout(self.config.t_dropout_rate)

            self.fusion = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "classifier_activation_1",
                            self.config.fusion_activation,
                        ),
                        (
                            "classifier_dropout_1",
                            nn.Dropout(self.config.fusion_dropout_rate),
                        ),
                        (
                            "classifier_linear_1",
                            nn.Linear(self.config.fusion_in, self.config.fusion_hidden),
                        ),
                        (
                            "classifier_activation_2",
                            self.config.fusion_activation,
                        ),
                        (
                            "classifier_dropout_2",
                            nn.Dropout(self.config.fusion_dropout_rate),
                        ),
                        (
                            "classifier_linear_2",
                            nn.Linear(self.config.fusion_hidden, self.config.classes),
                        ),
                    ]
                )
            )

    def _check_input(self, batch):
        if self.config.network_type == VLMType.VISION_CLASSIFICATION:
            assert len(batch.shape) == 4, (
                f"For vision classification, input should be (b, c, h, w) (4 dims) "
                f"but has shape {batch.shape} ({len(batch)} dims)."
            )
        elif self.config.network_type == VLMType.VQA_CLASSIFICATION:
            assert len(batch) == 2, (
                f"For VQA classification, input should be (v, q) (2 dims) but is "
                f"{len(batch)} dims."
            )
            shape_img = list(batch[0].shape)
            shape_text = list(batch[1].shape)
            assert len(batch[0].shape) == 4, (
                f"For VQA classification, input[0] should be (b, c, h, w) (4 dims) "
                f"but has shape {shape_img} ({len(batch)} dims)."
            )
            assert len(batch[1].shape) == 2, (
                f"For VQA classification, input[1] should be (b, t) (2 dims) "
                f"but has shape {shape_text} ({len(batch)} dims)."
            )
            assert shape_img[0] == shape_text[0], (
                f"For VQA classification, inputs 0 and 1 should be of "
                f"same batch size but are {shape_img[0]} and {shape_text[0]}"
            )
            if (
                shape_text[1] != self.config.max_sequence_length
                and not self.config.use_pooler_output
            ):
                warnings.warn(
                    f"Text input has a length of {shape_text[1]} but sequence length is"
                    f" {self.config.max_sequence_length} and no pooler is used. This "
                    f"will probably fail. Please use a pooler, set the sequence length "
                    f"when initializing the model or pad accordingly."
                )
        else:
            raise ValueError(f"Configuration type '{self.config.network_type}' unknown")

    def forward(self, batch):
        # check that input is correct before running network
        self._check_input(batch)

        if self.config.network_type == VLMType.VISION_CLASSIFICATION:
            return self.vision_encoder(batch)
        elif self.config.network_type == VLMType.VQA_CLASSIFICATION:
            v, q = batch
            # visual path
            v = self.vision_encoder(v)
            v = self.dropout_v(v)
            v = self.visual_linear(v)
            v = self.config.fusion_activation(v)

            # text path
            if self.config.use_pooler_output:
                # linear input is QFormer.config.hidden_size
                q = self.text_encoder(q).pooler_output.flatten(start_dim=1)
            else:
                # linear input is QFormer.config.hidden_size * seq_length
                q = self.text_encoder(q).last_hidden_state.flatten(start_dim=1)
            q = self.dropout_q(q)
            q = self.text_linear(q)
            q = self.config.fusion_activation(q)

            # late fusion
            x = self.config.fusion_method(v, q)
            x = self.fusion(x)
            return x
