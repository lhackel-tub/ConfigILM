"""
Framework for combining vision and language models
"""

__author__ = "Leonard Hackel"
__credits__ = ["Leonard Hackel"]
__maintainer__ = "Leonard Hackel"
__email__ = "l.hackel@tu-berlin.de"

import warnings
from collections import OrderedDict
from dataclasses import dataclass, asdict
from enum import Enum
from importlib import import_module
from os import listdir
from os.path import isdir, join
from pathlib import Path
from typing import Sequence, Callable, Optional, Union, Tuple

import timm
import torch
import transformers
from appdirs import user_cache_dir
from requests.exceptions import HTTPError  # type: ignore
from requests.exceptions import ReadTimeout  # type: ignore
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)


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


def _get_hf_model(
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
    save_directory = Path(user_cache_dir(appname="configilm")).joinpath("pretrained_models", "huggingface_models")

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
            raise HTTPError(f"Model '{model_name}' could not be fetched. Please check spelling.")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model '{model_name}' could not be fetched. " f"Network is down and file not cached."
            )
        except ReadTimeout:
            raise ReadTimeout(f"Model '{model_name}' could not be fetched. " f"Timeout and file not cached.")

    # Model is available or was made available
    model_path = join(save_directory, model_name)

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, config=config)

    if load_pretrained_if_available:
        model = transformers.AutoModel.from_pretrained(model_path, config=config, local_files_only=True)
        warnings.warn("Tokenizer was initialized pretrained")
    else:
        model = transformers.AutoModel.from_config(config=config)

    return tokenizer, model


def _get_timm_model(model_name: str, kwargs: dict):
    """
    Wrapper around timm.create_model that tries to create a model with specified keyword
    arguments. If an argument does not work, it will be removed from the list and
    creation is retried until successful.

    :param model_name: string name of the requested model - see timm.list_models("*")
        for all available models
    :param kwargs: arguments passed to the create function of timm
    :return: pytorch model of requested timm style
    """
    while True:
        try:
            encoder = timm.create_model(model_name, **kwargs)
            break
        except TypeError as t:
            # get keyword that failed and drop it
            failed_kw = t.args[0].split("'")[1]
            del kwargs[failed_kw]
            warnings.warn(f"Keyword '{failed_kw}' unknown. Trying to ignore and restart creation.")
    return encoder


class ILMType(int, Enum):
    """
    Class for different types of architectures supported by ILMConfigurations
    """

    IMAGE_CLASSIFICATION = 0
    VQA_CLASSIFICATION = 1

    def __str__(self):
        return f"{self.name} (value: {self.value})"


_MODULE_CONVERTER = {
    "nn": "torch.nn",
    "F": "torch.nn.functional",
    "timm": "timm",
    "transformers": "transformers",
}


def _str_to_callable(s: str):
    """
    Converts a string to a callable if possible. If not, returns the string.

    :param s: string to convert
    :return: callable or string
    """
    module = None
    module_split = s.split(".")
    if len(module_split) > 1:
        try:
            module_str = ".".join(module_split[:-1])
            if module_str in _MODULE_CONVERTER.keys():
                module = import_module(_MODULE_CONVERTER[module_str])
            else:
                module = import_module(module_str)
            method = module_split[-1]
            callable_method = eval(".".join(["module", method]))
        except ImportError:
            raise ImportError(f"Module '{module_split[0]}' not found. Are you sure it is installed?")
        except AttributeError:
            assert module is not None
            raise AttributeError(f"Attribute '{module_split[-1]}' not found in module '{module.__name__}'.")
    else:
        try:
            callable_method = eval(s)
        except NameError:
            raise NameError(f"Name '{s}' not found.")
    assert callable(callable_method), "Method is not callable"
    return callable_method


def _callable_to_str(c: Callable):
    """
    Converts a callable to a string if possible. If not, returns the string.

    :param c: callable to convert
    :return: string or callable
    """
    with_brackets = False
    if str(c).endswith("()"):
        with_brackets = True
        c = c.__class__
    try:
        callable_name = c.__name__
        module_name = c.__module__
        return f"{module_name}.{callable_name}{('()' if with_brackets else '')}"
    except AttributeError:
        raise AttributeError(
            f"Callable '{c}' has no name or module, cannot convert to string. Please use a string "
            f"instead, e.g. use 'torch.nn.Tanh()' instead of torch.nn.Tanh()."
        )


def _callable_tuple_to_str(t: Tuple[str, Callable]):
    """
    Converts a tuple of a string and a callable to a string if possible. Otherwise, raises an assertion error.

    :param t: tuple of a string and a callable to convert
    :return: string
    """
    assert len(t) == 2, "Tuple has to have two elements"
    assert isinstance(t[0], str), "First element has to be a string"
    assert callable(t[1]), "Second element has to be a callable"
    # set the name of the callable into globals to enable later evaluation
    try:
        globals()[t[1].__name__] = t[1]
    except AttributeError:
        try:
            if hasattr(t[1], "_get_name"):
                globals()[t[1]._get_name()] = t[1]
            else:
                raise AttributeError(f"{t[1]} has no attribute '_get_name' to set name to globals.")
        except AttributeError:
            raise AttributeError(f"Could not set name of callable '{t[1]}' to globals.")
    return t[0]


@dataclass
class ILMConfiguration:
    """
    Configuration dataclass that defines all properties of ConfigILM models. The datatypes within the dataclass are
    selected to be compatible with json serialization and deserialization.

    :param channels: Number of input channels for the image model.

        :Default: 3

    :param class_names: Names of the classes in the classifier. Usable for
        class-specific performance. If none, classes will be enumerated with
        numbers.

        :Default: None

    :param classes: Number of classes for the output of IMAGE_CLASSIFICATION
        classifier or VQA_CLASSIFICATION classifier.

        :Default: 10

    :param custom_fusion_activation: Activation function inside all classification head
        layers. Only used for initialization. After initialization, the fusion activation is
        accessible via the fusion_activation property. Default options from torch.nn or torch.nn.functional
        can be used as strings (e.g. "nn.Tanh()") or as callables (e.g. nn.Tanh()).
        If a custom
        function is used, it has to be a callable with a single input (tensor) and a single output
        (tensor) where the input tensor is single dimension (plus batch dimension). The activation is then
        passed as a tuple (str, m) where str is the string representation of the function and m is the
        callable, which has to be a torch.nn.Module and initialized with the correct parameters, e.g.
        ("tanh", nn.Tanh()).

        :Default: nn.Tanh()

    :param custom_fusion_method: Fusion method to combine text and image features. Callable
        with two inputs (tensor, tensor) and a single output (tensor) where each
        tensor is single dimension (plus batch dimension). First input is flatten
        output of image model with dimension fusion_in, second input is flatten
        output of the text model with dimension fusion_in. Output should have the
        dimension fusion_out. Only used for initialization. After initialization, the
        fusion method is accessible via the fusion_method property.
        If a custom function is used, it has to be a callable with two inputs (tensor, tensor) and a single output
        (tensor) where each tensor is single dimension (plus batch dimension). The fusion method is then
        passed as a tuple (str, f) where str is the string representation of the function and f is the
        callable, e.g. ("mul", torch.mul).

        :Default: torch.mul

    :param drop_rate: Dropout and drop path rate for timm models.

        :Default: 0.2

    :param fusion_dropout_rate: Drop rate inside all classification head layers.

        :Default: 0.25

    :param fusion_hidden: Number of neurons inside the hidden layer of the
        classification head.

        :Default: 256

    :param fusion_in: Input dimension to the fusion method.

        :Default: 512

    :param fusion_out: Output dimension of the fusion method. If None, output will
        be same as input (e.g. for point-wise operations).

        :Default: None

    :param hf_model_name: Name of the text model from huggingface if applicable. The
        model has to be a model for text sequence classification.

        :Default: None

    :param image_size: Size of input images for image models. Only applicable for
        some specific models.

        :Default: 120

    :param load_pretrained_hf_if_available: Load pretrained weights for huggingface
        model.

        :Default: True

    :param load_pretrained_timm_if_available: Load pretrained weights for timm
        model.

        :Default: False

    :param max_sequence_length: Maximum sequence length of huggingface models.
        Sequences that are shorter will be padded, longer ones are cropped to this
        maximum length.

        :Default: 32

    :param network_type: Type of ILM-network. Available types are listed in ILMType
        enum.

        :Default: ILMType.IMAGE_CLASSIFICATION

    :param t_dropout_rate: Dropout rate of the mapping from the huggingface text
        model to the dimension of the fusion method.

        :Default: 0.25

    :param timm_model_name: (required) Name of the image model as defined in
        timm.list_models()

    :param use_pooler_output: Use the pooler output of the huggingface model if
        applicable and available. Otherwise, last hidden features will be flattened
        and used instead.

        :Default: True

    :param v_dropout_rate: Dropout rate of the mapping from the timm image model to
        the dimension of the fusion method.

        :Default: 0.25

    :param visual_features_out: Output dimension of the timm image model. Dimension
        will be linearly mapped to fusion_in dimension with activation and dropout
        as specified.

        :Default: 512

    """

    timm_model_name: str
    hf_model_name: Optional[str] = None
    image_size: int = 120
    channels: int = 3
    classes: int = 10
    class_names: Optional[Sequence[str]] = None
    network_type: ILMType = ILMType.IMAGE_CLASSIFICATION
    visual_features_out: int = 512
    fusion_in: int = 512
    fusion_out: Optional[int] = None  # if None, same as fusion_in
    fusion_hidden: int = 256
    v_dropout_rate: float = 0.25
    t_dropout_rate: float = 0.25
    fusion_dropout_rate: float = 0.25
    _fusion_method: str = "torch.mul"
    _fusion_activation: str = "nn.Tanh()"
    drop_rate: Optional[float] = 0.2
    use_pooler_output: bool = True
    max_sequence_length: int = 32
    load_pretrained_timm_if_available: bool = False
    load_pretrained_hf_if_available: bool = True
    custom_fusion_method: Optional[
        Union[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], str, Tuple[str, Callable]]
    ] = None
    custom_fusion_activation: Optional[Union[Callable[[torch.Tensor], torch.Tensor], str, Tuple[str, Callable]]] = None

    def __post_init__(self):
        if self.fusion_out is None:
            self.fusion_out = self.fusion_in
        # go over all properties and check if they are called "custom_"
        self_dict_keys = list(self.__dict__.keys())  # needed for iteration and deletion
        for key in self_dict_keys:
            if key.startswith("custom_"):
                # check if the custom function is set
                if self.__dict__[key] is not None:
                    # set the real function / call the property setter
                    setattr(self, key[7:], self.__dict__[key])
                    # delete the custom property
                    delattr(self, key)

    @property
    def fusion_activation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        # convert to callable if string
        fusion_activation = _str_to_callable(self._fusion_activation)
        assert isinstance(fusion_activation, nn.Module), "Fusion activation has to be a torch.nn.Module"
        return fusion_activation

    @fusion_activation.setter
    def fusion_activation(self, value: Union[str, Callable[[torch.Tensor], torch.Tensor], Tuple[str, Callable]]):
        if isinstance(value, str):
            self._fusion_activation = value
        elif callable(value):
            # convert to string
            self._fusion_activation = _callable_to_str(value)
        elif isinstance(value, tuple):
            self._fusion_activation = _callable_tuple_to_str(value)
        else:
            raise ValueError("Fusion activation has to be a string or a callable or a tuple of a string and a callable")

    @property
    def fusion_method(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        # convert to callable if string
        return _str_to_callable(self._fusion_method)

    @fusion_method.setter
    def fusion_method(
        self, value: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Tuple[str, Callable]]
    ):
        if isinstance(value, str):
            self._fusion_method = value
        elif callable(value):
            # convert to string
            self._fusion_method = _callable_to_str(value)
        elif isinstance(value, tuple):
            self._fusion_method = _callable_tuple_to_str(value)
        else:
            raise ValueError("Fusion method has to be a string or a callable or a tuple of a string and a callable")

    def dif(self, other):
        """
        Compares two ILMConfigurations and returns a dictionary with the differences.
        """
        if not isinstance(other, ILMConfiguration):
            return {"other": other, "self": self}
        differences = {}
        for key in self.__dict__.keys():
            if key not in other.__dict__.keys():
                # key is not in other
                differences[key] = self.__dict__[key]
            if self.__dict__[key] != other.__dict__[key]:
                # keys are different
                if not isinstance(self.__dict__[key], Callable):
                    differences[key] = self.__dict__[key]
                else:
                    # key is a callable
                    # check if their internal params are the same
                    if self.__dict__[key].__dict__ != other.__dict__[key].__dict__:
                        differences[key] = self.__dict__[key]
        return differences

    def __eq__(self, other):
        """
        Compares two ILMConfigurations and returns True if they are equal.
        """
        return self.dif(other) == {}

    def as_dict(self):
        """
        Returns the configuration as a dictionary.
        """
        return asdict(self)

    def to_json(self):
        """
        Returns the configuration as a json string.
        """
        import json

        dct = self.as_dict()
        return json.dumps(dct, indent=4)

    @classmethod
    def from_json(cls, json_string):
        """
        Loads a configuration from a json string.
        """
        import json

        self = cls.__new__(cls)
        self.__dict__ = json.loads(json_string)
        return self


class ConfigILM(nn.Module):
    def __init__(self, config: ILMConfiguration):
        """
        Creates a ConfigILM model according to the provided ILMConfiguration

        :param config: Configuration file of the model. See ILMConfiguration for details
        :returns: self
        """
        super().__init__()
        self.config = config
        if self.config.class_names is None:
            self.config.class_names = [str(i) for i in range(self.config.classes)]

        if config.network_type in [
            ILMType.VQA_CLASSIFICATION,
            ILMType.IMAGE_CLASSIFICATION,
        ]:
            # keyword arguments as expected by Timm lib
            timm_kwargs = {
                "num_classes": self.config.visual_features_out
                if config.network_type in [ILMType.VQA_CLASSIFICATION]
                else self.config.classes,
                "img_size": self.config.image_size,
                "in_chans": self.config.channels,
                "drop_rate": self.config.drop_rate,
                "drop_path_rate": self.config.drop_rate,
                "pretrained": self.config.load_pretrained_timm_if_available,
            }

            # create timm_model
            self.vision_encoder = _get_timm_model(config.timm_model_name, timm_kwargs)

        if config.network_type in [ILMType.VQA_CLASSIFICATION]:
            # create huggingface model
            assert config.hf_model_name is not None, "Requesting huggingface model but not specifying which"
            self.tokenizer, self.text_encoder = _get_hf_model(
                config.hf_model_name,
                load_pretrained_if_available=config.load_pretrained_hf_if_available,
            )

        if config.network_type == ILMType.VQA_CLASSIFICATION:
            # create fusion layer
            # get number of out-features
            text_features_out = self.text_encoder.config.hidden_size
            if not self.config.use_pooler_output:
                text_features_out *= self.config.max_sequence_length
            else:
                # check if text encoder has a pooler
                if not hasattr(self.text_encoder, "pooler"):
                    warnings.warn(
                        f"Text encoder '{config.hf_model_name}' has no pooler, " f"changing use_pooler_output to False"
                    )
                    text_features_out *= self.config.max_sequence_length
                    self.config.use_pooler_output = False

            # map text encoder -> fusion_in
            self.text_linear = nn.Linear(text_features_out, self.config.fusion_in)
            # map vision encoder -> fusion_in
            self.visual_linear = nn.Linear(self.config.visual_features_out, self.config.fusion_in)

            self.dropout_v = torch.nn.Dropout(self.config.v_dropout_rate)
            self.dropout_q = torch.nn.Dropout(self.config.t_dropout_rate)

            assert (
                self.config.fusion_out is not None
            ), "Fusion output dimension has to be specified - this should have been done in the configuration"

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
                            nn.Linear(self.config.fusion_out, self.config.fusion_hidden),
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

    def to(self, *args, **kwargs):
        """
        Moves the model parts as well as the fusion methods and activations to a device
        or casts to a different type.

        :param args: device, dtype or other specified formats. See `nn.module.to()`
        :param kwargs: device, dtype or other specified formats. See `nn.module.to()`
        """
        super().to(*args, **kwargs)
        if isinstance(self.config.fusion_method, nn.Module):
            self.config.fusion_method.to(*args, **kwargs)

        if isinstance(self.config.fusion_activation, nn.Module):
            self.config.fusion_activation.to(*args, **kwargs)

    def get_tokenizer(self):
        """
        Getter to the tokenizer of the text model if applicable.

        :return: Tokenizer to specified huggingface text model.
        :raises: AttributeError if no text model is used
        """
        if hasattr(self, "tokenizer"):
            return self.tokenizer
        else:
            raise AttributeError(
                f"ConfigILM of type {self.config.network_type} has no"
                f"tokenizer. Please use a different network type."
            )

    def _check_input(self, batch):
        """
        Helper function that checks if the batch has the right dimension for the
        specified model.

        :param batch: Input batch
        :raises: AssertionError if the shape does not fit to the configuration
        :raises: ValueError if the configuration type is not known
        """
        if self.config.network_type == ILMType.IMAGE_CLASSIFICATION:
            assert len(batch.shape) == 4, (
                f"For vision classification, input should be (b, c, h, w) (4 dims) "
                f"but has shape {batch.shape} ({len(batch)} dims)."
            )
        elif self.config.network_type == ILMType.VQA_CLASSIFICATION:
            assert len(batch) == 2, (
                f"For VQA classification, input should be (v, q) (2 dims) but is " f"{len(batch)} dims."
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
            if shape_text[1] != self.config.max_sequence_length and not self.config.use_pooler_output:
                warnings.warn(
                    f"Text input has a length of {shape_text[1]} but sequence length is"
                    f" {self.config.max_sequence_length} and no pooler is used. This "
                    f"will probably fail. Please use a pooler, set the sequence length "
                    f"when initializing the model or pad accordingly."
                )
        else:
            raise ValueError(f"Configuration type '{self.config.network_type}' unknown")

    def forward(self, batch):
        """
        Model forward function that decides which parts of the model to use based on the
        configuration after checking that the input works with this kind of network.

        :param batch: Input batch of single modality of list of batches of multiple
            modalities
        Note: The text input of the VQA model will be automatically masked based on the
            padding tokens of the tokenizer.
        :returns: logits of the network
        """
        # check that input is correct before running network
        self._check_input(batch)

        if self.config.network_type == ILMType.IMAGE_CLASSIFICATION:
            return self.vision_encoder(batch)
        elif self.config.network_type == ILMType.VQA_CLASSIFICATION:
            v, q = batch
            # visual path
            v = self.vision_encoder(v)
            v = self.dropout_v(v)
            v = self.visual_linear(v)
            v = self.config.fusion_activation(v)

            # text path
            attention_mask = (q != self.tokenizer.pad_token_id).float()
            if self.config.use_pooler_output:
                # linear input is QFormer.config.hidden_size
                q = self.text_encoder(q, attention_mask).pooler_output.flatten(start_dim=1)
            else:
                # linear input is QFormer.config.hidden_size * seq_length
                q = self.text_encoder(q, attention_mask).last_hidden_state.flatten(start_dim=1)
            q = self.dropout_q(q)
            q = self.text_linear(q)
            q = self.config.fusion_activation(q)

            # late fusion
            x = self.config.fusion_method(v, q)
            x = self.fusion(x)
            return x
