import dataclasses
import shutil
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import pytest
import torch
from appdirs import user_cache_dir
from requests.exceptions import ReadTimeout  # type: ignore

from configilm import ConfigILM
from configilm.extra.CustomTorchClasses import NoneActivation
from configilm.Fusion.LinearSumFusion import LinearSum

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
max_memory_usage = 8  # VRAM usage of largest model during test

# If the available VRAM is smaller than the largest model, some tests will fail due to
# CUDA Out-Of-Memory errors. Therefore, only use CUDA if there is enough VRAM available.
if torch.cuda.is_available():
    cuda_gb_vram = round(torch.cuda.get_device_properties("cuda:0").total_memory / 1024**3, 2)
    DEVICE = "cuda:0" if cuda_gb_vram >= max_memory_usage else "cpu"


def get_classification_batch(
    img_shape: Sequence = (12, 120, 120),
    batch_size: int = 32,
    classes: int = 1000,
):
    assert len(img_shape) == 3
    v = torch.ones((batch_size, img_shape[0], img_shape[1], img_shape[2]), device=DEVICE)
    lbl = torch.ones((batch_size, classes), device=DEVICE)
    return [v, lbl]


def get_vqa_batch(
    img_shape: Sequence = (12, 120, 120),
    text_tokens: int = 64,
    batch_size: int = 32,
    classes: int = 1000,
):
    v, a = get_classification_batch(img_shape=img_shape, batch_size=batch_size, classes=classes)
    q = torch.ones((batch_size, text_tokens), dtype=torch.int32, device=DEVICE)
    return [v, q, a]


@pytest.mark.parametrize("batch_size", [1, 32, 27, 13])
def test_bs(batch_size):
    config = ConfigILM.ILMConfiguration(timm_model_name=default_image_test_model)
    x, y = get_classification_batch((config.channels, config.image_size, config.image_size), batch_size)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Keyword 'img_size' unknown. Trying to " "ignore and restart creation.",
        )
        model = ConfigILM.ConfigILM(config=config)
        model.to(DEVICE)
    model(x)
    # pass


default_image_test_model = "resnet18"
default_text_test_model = "prajjwal1/bert-tiny"
tested_timm_models_120 = [
    "cait_xxs24_224",
    "coat_lite_tiny",
    "convit_tiny",
    "convmixer_768_32",
    "convnext_femto",
    "deit3_base_patch16_224",
    "deit_tiny_patch16_224",
    "eca_resnet33ts",
    "edgenext_xx_small",
    "efficientnet_b0",
    # "efficientnet_b4", removed in 0.4.10
    # "efficientnetv2_s", removed in 0.4.10
    "efficientnetv2_xl",
    "gmlp_ti16_224",
    "mixer_b16_224",
    "mobilenetv2_035",
    "mobilenetv3_small_050",
    "mobilevit_xxs",
    "mobilevitv2_050",
    "pit_ti_224",
    # "pit_ti_distilled_224", removed in 0.4.10
    "poolformer_s12",
    "pvt_v2_b0",
    # "pvt_v2_b4", removed in 0.4.10
    "res2net50_14w_8s",
    "res2next50",
    "resnest14d",
    "resnet10t",
    "resnet18",
    # "resnet34",  removed in 0.4.10
    # "resnet50",  removed in 0.4.10
    "resnetv2_50",
    "resnext26ts",
    # "semobilevit_s", removed in 0.4.0
    "resnet18.fb_ssl_yfcc100m_ft_in1k",  # name change in 0.4.0
    "tinynet_a",
    "vgg16",
    "visformer_tiny",
    "vit_small_patch16_18x2_224",
    "vit_tiny_patch16_224",
    "volo_d1_224",
    "wide_resnet50_2",
    "xcit_tiny_12_p8_224",
    "xcit_tiny_12_p8_224.fb_dist_in1k",  # name change in 0.4.0
    # new tests in 0.4.0
    "eva02_tiny_patch14_336.mim_in22k_ft_in1k",
    "convnext_base.clip_laion2b_augreg_ft_in12k_in1k",
    "convnextv2_tiny",
]

tested_timm_models_224 = [
    "coat_tiny",
    "efficientformer_l1",
    "nest_tiny_jx",  # name change in 0.4.0
    "levit_128",
    "levit_128s",
    "mvitv2_tiny",
    "swin_s3_tiny_224",
    "swin_tiny_patch4_window7_224",
    "swinv2_cr_tiny_224",
    # new tests in 0.4.0
    "coatnet_2_rw_224.sw_in12k_ft_in1k",
    "coatnet_nano_rw_224.sw_in1k",
    "convnext_atto",
    "convnext_femto",
    "convnext_nano",
    # "convnext_pico", removed in 0.4.10
    # "convnext_small", removed in 0.4.10
    # "convnext_tiny", removed in 0.4.10
    "convnextv2_atto",
    "convnextv2_femto",
    # "convnextv2_nano", removed in 0.4.10
    # "convnextv2_pico", removed in 0.4.10
    # "convnextv2_small", removed in 0.4.10
    "coatnext_nano_rw_224.sw_in1k",
    "maxvit_nano_rw_256.sw_in1k",
]

tested_timm_models_240 = [
    "crossvit_tiny_240",
]

tested_timm_models_256 = [
    "swinv2_tiny_window8_256",
]

tested_dims = [10, 256, 1024, 360, 1200]


def apply_timm(model: str, cls: int, bs: int, image_size: int):
    config = ConfigILM.ILMConfiguration(
        timm_model_name=model,
        drop_rate=None,
        image_size=image_size,
        classes=cls,
    )
    x, y = get_classification_batch(
        (config.channels, config.image_size, config.image_size),
        batch_size=bs,
        classes=cls,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Keyword 'img_size' unknown. Trying to " "ignore and restart creation.",
        )
        model = ConfigILM.ConfigILM(config=config)
        model.to(DEVICE)
    res = model(x)
    assert res.shape == y.shape, f"Shape is wrong: Should be {y.shape} but is {res.shape}"
    assert res.shape == (
        bs,
        cls,
    ), f"Shape is wrong: Should be {(bs, cls)} but is {res.shape}"


@pytest.mark.parametrize("model", tested_timm_models_120)
def test_timm_120(model):
    apply_timm(model=model, cls=10, bs=4, image_size=120)


@pytest.mark.parametrize("model", tested_timm_models_224)
def test_timm_224(model):
    apply_timm(model=model, cls=10, bs=4, image_size=224)


@pytest.mark.parametrize("model", tested_timm_models_240)
def test_timm_240(model):
    apply_timm(model=model, cls=10, bs=4, image_size=240)


@pytest.mark.parametrize("model", tested_timm_models_256)
def test_timm_256(model):
    apply_timm(model=model, cls=10, bs=4, image_size=256)


def apply_ilm(config: ConfigILM.ILMConfiguration, bs: int) -> bool:
    cls = config.classes
    v, q, a = get_vqa_batch(
        (config.channels, config.image_size, config.image_size),
        batch_size=bs,
        classes=cls,
        text_tokens=config.max_sequence_length,
    )
    i = 0

    model = None
    while True:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore",
                    category=UserWarning,
                    message="Keyword 'img_size' unknown. Trying to " "ignore and restart creation.",
                )
                warnings.filterwarnings(
                    action="ignore",
                    category=UserWarning,
                    message=r"Model '\S+' not available. Trying to " "download...",
                )
                warnings.filterwarnings(
                    action="ignore",
                    category=UserWarning,
                    message=r"Text encoder '\S+' has no pooler, " "changing use_pooler_output to False",
                )
                warnings.filterwarnings(
                    action="ignore",
                    category=UserWarning,
                    message="Tokenizer was initialized pretrained",
                )
                model = ConfigILM.ConfigILM(config=config)
                model.to(DEVICE)
        except ReadTimeout:
            # Model could not load, retry
            i += 1
            continue
        if i >= 4:
            # model could not be loaded after 5 tries
            return False
        break

    if model is not None:
        res = model((v, q))
        assert res.shape == a.shape, f"Shape is wrong: Should be {a.shape} but is {res.shape}"
        assert res.shape == (
            bs,
            cls,
        ), f"Shape is wrong: Should be {(bs, cls)} but is {res.shape}"
        return True
    return False


hf_models = ["distilbert-base-uncased", "prajjwal1/bert-tiny"]
hf_models_full = hf_models + [
    "albert-base-v2",
    "google/mobilebert-uncased",
    "huawei-noah/TinyBERT_General_4L_312D",
]


@pytest.mark.parametrize("hfmodel", hf_models_full)
def test_ilm_default(hfmodel):
    bs = 4
    config = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        image_size=120,
        channels=12,
        hf_model_name=hfmodel,
        classes=10,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
        max_sequence_length=32,
    )
    if not apply_ilm(config=config, bs=bs):
        pytest.skip("Download did not work")


@pytest.mark.parametrize("hfmodel", hf_models)
def test_ilm_dont_use_pooler(hfmodel):
    bs = 4
    config = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        image_size=120,
        channels=12,
        hf_model_name=hfmodel,
        classes=10,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
        max_sequence_length=32,
        use_pooler_output=False,
    )
    if not apply_ilm(config=config, bs=bs):
        pytest.skip("Download did not work")


def test_ilm_download():
    hf_model = "distilbert-base-uncased"
    path = Path(user_cache_dir(appname="configilm"))
    shutil.rmtree(path, ignore_errors=True)
    bs = 4
    config = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        image_size=120,
        channels=12,
        hf_model_name=hf_model,
        classes=10,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
        max_sequence_length=32,
    )
    if not apply_ilm(config=config, bs=bs):
        pytest.skip("Download did not work")


def test_ilm_untrained():
    bs = 4
    config = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        image_size=120,
        channels=12,
        hf_model_name=default_text_test_model,
        classes=10,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
        max_sequence_length=32,
        load_pretrained_hf_if_available=False,
    )
    apply_ilm(config=config, bs=bs)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 6])
def test_v_wrong_batchshape(n):
    config = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        drop_rate=None,
        image_size=120,
        classes=10,
    )
    shape = [16] * n
    x = torch.ones(shape, device=DEVICE)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Keyword 'img_size' unknown. Trying to " "ignore and restart creation.",
        )

        model = ConfigILM.ConfigILM(config=config)
        model.to(DEVICE)
    with pytest.raises(AssertionError):
        _ = model(x)


@pytest.mark.parametrize("img, txt", [(i, t) for i in range(1, 6, 1) for t in range(1, 6, 1)])
def test_ilm_wrong_batch_parts_shape(img, txt):
    bs, cls = 8, 10
    config = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        image_size=120,
        channels=bs,
        hf_model_name=default_text_test_model,
        classes=cls,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
        max_sequence_length=32,
    )
    img_shape = [bs] * img
    txt_shape = [bs] * txt
    v = torch.ones(img_shape, device=DEVICE)
    q = torch.ones(txt_shape, dtype=torch.int32, device=DEVICE)
    a = torch.ones((bs, cls), device=DEVICE)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Keyword 'img_size' unknown. Trying to " "ignore and restart creation.",
        )

        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Tokenizer was initialized pretrained",
        )
        model = ConfigILM.ConfigILM(config=config)
        model.to(DEVICE)
    if img != 4 or txt != 2:
        with pytest.raises(AssertionError):
            _ = model((v, q))
    else:
        res = model((v, q))
        assert res.shape == a.shape, f"Shape is wrong: Should be {a.shape} but is {res.shape}"
        assert res.shape == (
            bs,
            cls,
        ), f"Shape is wrong: Should be {(bs, cls)} but is {res.shape}"


def test_ilm_wrong_different_batch_sizes():
    config = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        image_size=120,
        channels=12,
        hf_model_name=default_text_test_model,
        classes=10,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
        max_sequence_length=32,
    )
    img_shape = [16] * 4
    txt_shape = [8] * 2
    v = torch.ones(img_shape, device=DEVICE)
    q = torch.ones(txt_shape, dtype=torch.int32, device=DEVICE)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Keyword 'img_size' unknown. Trying to " "ignore and restart creation.",
        )

        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Tokenizer was initialized pretrained",
        )
        model = ConfigILM.ConfigILM(config=config)
        model.to(DEVICE)
    with pytest.raises(AssertionError):
        _ = model((v, q))


def test_ilm_wrong_seq_length_no_pooler():
    config = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        image_size=120,
        channels=12,
        hf_model_name=default_text_test_model,
        classes=10,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
        max_sequence_length=32,
        use_pooler_output=False,
    )
    img_shape = [16] * 4
    txt_shape = [16] * 2
    v = torch.ones(img_shape, device=DEVICE)
    q = torch.ones(txt_shape, dtype=torch.int32, device=DEVICE)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Keyword 'img_size' unknown. Trying to " "ignore and restart creation.",
        )

        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Tokenizer was initialized pretrained",
        )
        model = ConfigILM.ConfigILM(config=config)
        model.to(DEVICE)
    with pytest.raises(RuntimeError):
        # This specific one config (hf_model_name="prajjwal1/bert-tiny")
        # will have a runtime error. This may not happen for other configs

        with pytest.warns(UserWarning, match=r".+ length of \d+ but sequence length is \d+ and .+"):
            _ = model((v, q))


def test_ilm_wrong_input_length():
    config = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        image_size=120,
        channels=12,
        hf_model_name=default_text_test_model,
        classes=10,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
        max_sequence_length=32,
    )
    img_shape = [16] * 4
    txt_shape = [16] * 2
    v = torch.ones(img_shape, device=DEVICE)
    q = torch.ones(txt_shape, dtype=torch.int32, device=DEVICE)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Keyword 'img_size' unknown. Trying to " "ignore and restart creation.",
        )

        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Tokenizer was initialized pretrained",
        )
        model = ConfigILM.ConfigILM(config=config)
        model.to(DEVICE)
    with pytest.raises(AssertionError):
        # This specific one config (hf_model_name="prajjwal1/bert-tiny") will have a
        # runtime error. This may not happen for other configs
        _ = model((v, q, q))


def test_failty_config():
    bs = 4
    config = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        image_size=120,
        channels=12,
        hf_model_name=default_text_test_model,
        classes=10,
        network_type=-5,
        max_sequence_length=32,
        load_pretrained_hf_if_available=False,
    )
    v, q, a = get_vqa_batch(
        (config.channels, config.image_size, config.image_size),
        batch_size=bs,
        classes=config.classes,
        text_tokens=config.max_sequence_length,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Keyword 'img_size' unknown. Trying to " "ignore and restart creation.",
        )

        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Tokenizer was initialized pretrained",
        )
        model = ConfigILM.ConfigILM(config=config)
        model.to(DEVICE)
    with pytest.raises(ValueError):
        _ = model((v, q))


def test_failed_network_connection_in_download(mocker):
    hf_model = "distilbert-base-uncased"
    path = Path(user_cache_dir(appname="configilm")).joinpath("pretrained_models", "huggingface_models", hf_model)
    shutil.rmtree(path)

    mocker.patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        new=lambda x: (_ for _ in ()).throw(FileNotFoundError),
    )
    bs = 4
    config = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        image_size=120,
        channels=12,
        hf_model_name=hf_model,
        classes=10,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
        max_sequence_length=32,
    )

    with pytest.raises(FileNotFoundError):
        apply_ilm(config=config, bs=bs)


def test_failed_network_connection_cached(mocker):
    hf_model = default_text_test_model
    path = Path(user_cache_dir(appname="configilm")).joinpath("pretrained_models", "huggingface_models", hf_model)
    if not path.exists():
        pytest.skip(f"HF model '{hf_model}' to test was not downloaded before test. " f"Cannot check cache.")

    mocker.patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        new=lambda x: (_ for _ in ()).throw(ConnectionError),
    )
    bs = 4
    config = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        image_size=120,
        channels=12,
        hf_model_name=hf_model,
        classes=10,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
        max_sequence_length=32,
    )
    # works even tho we cannot download
    apply_ilm(config=config, bs=bs)


def test_failed_hf_name(mocker):
    from requests import HTTPError  # type: ignore

    hf_model = "hf_mock_name/simulated_name"
    path = Path(user_cache_dir(appname="configilm")).joinpath("pretrained_models", "huggingface_models", hf_model)
    if path.exists():
        pytest.skip(f"HF model '{hf_model}' exists offline already. Cannot check download.")

    mocker.patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        new=lambda x: (_ for _ in ()).throw(HTTPError),
    )
    bs = 4
    config = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        image_size=120,
        channels=12,
        hf_model_name=hf_model,
        classes=10,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
        max_sequence_length=32,
    )

    with pytest.raises(HTTPError):
        apply_ilm(config=config, bs=bs)


def test_integration():
    """
    Tests an example code that loads pretrained weights and executes an image of ones on
    it to confirm that everything works and the right values are returned.
    """
    cfg = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        classes=1000,
        load_pretrained_timm_if_available=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Keyword 'img_size' unknown. Trying to " "ignore and restart creation.",
        )

        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Tokenizer was initialized pretrained",
        )
        model = ConfigILM.ConfigILM(config=cfg)
        model.to(DEVICE)
    model.eval()

    in_t = torch.ones((1, 3, 120, 120), device=DEVICE)
    out_t = model(in_t)

    assert torch.all(
        torch.isclose(
            out_t[0][:5],
            torch.tensor([-5.6796, -5.2999, -4.8613, -5.0192, -4.8003], device=DEVICE),
            atol=0.005,
        )
    )


@pytest.mark.parametrize("dim", tested_dims)
def test_fusion_same_dim_explicit(dim):
    """
    Tests if fusion functions with the same input and output dimensions work
    """
    cfg = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        hf_model_name=default_text_test_model,
        fusion_in=dim,
        fusion_out=dim,
        custom_fusion_method=torch.mul,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
    )
    assert apply_ilm(cfg, bs=4), f"Fusion with {dim}->{dim} failed"


@pytest.mark.parametrize(
    "fusion, activation",
    [
        (f, a)
        for f in [
            torch.mul,
            torch.add,
            "torch.mul",
            "torch.add",
            LinearSum([512, 512], 512),
            "configilm.Fusion.LinearSumFusion.LinearSum([512, 512], 512)",
        ]
        for a in [
            torch.nn.Tanh(),
            torch.nn.ReLU(),
            "torch.nn.Tanh()",
            "torch.nn.ReLU()",
            NoneActivation(),
            "configilm.extra.CustomTorchClasses.NoneActivation()",
        ]
    ],
)
def test_custom_fusion_activation_method(fusion, activation):
    """
    Tests if fusion functions with the same input and output dimensions work
    """

    if isinstance(fusion, LinearSum):
        # LinearSum works only as a string argument
        with pytest.raises(AttributeError):
            _ = ConfigILM.ILMConfiguration(
                timm_model_name=default_image_test_model,
                hf_model_name=default_text_test_model,
                custom_fusion_method=fusion,
                custom_fusion_activation=activation,
                network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
            )
    else:
        cfg = ConfigILM.ILMConfiguration(
            timm_model_name=default_image_test_model,
            hf_model_name=default_text_test_model,
            custom_fusion_method=fusion,
            custom_fusion_activation=activation,
            network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
        )
        assert apply_ilm(cfg, bs=4), f"Fusion with 512->512 failed with {fusion} and {activation}"


@pytest.mark.parametrize("dim", tested_dims)
def test_fusion_same_dim_implicit(dim):
    """
    Tests if fusion functions with the same input and output dimensions work
    """
    cfg = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        hf_model_name=default_text_test_model,
        fusion_in=dim,
        fusion_out=None,
        custom_fusion_method=torch.mul,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
    )
    assert apply_ilm(cfg, bs=4), f"Fusion with {dim}->{dim} failed"


@pytest.mark.parametrize("in_dim, out_dim", [(i, o) for i in tested_dims for o in tested_dims])
def test_fusion_dif_dim(in_dim, out_dim):
    """
    Tests if fusion functions with the same input and output dimensions work
    """

    class _CatNet(torch.nn.Module):
        def __init__(self, in_dim_cat, out_dim_lin):
            super().__init__()
            self.net = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("FFN", torch.nn.Linear(2 * in_dim_cat, out_dim_lin)),
                        ("Activation", torch.nn.ReLU()),
                    ]
                )
            )

        def forward(self, x1, x2):
            # concat vectors, not batches
            x = torch.cat((x1, x2), dim=-1)
            return self.net(x)

    cfg = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        hf_model_name=default_text_test_model,
        fusion_in=in_dim,
        fusion_out=out_dim,
        custom_fusion_method=(f"_CatNet(in_dim_cat={in_dim}, out_dim_lin={out_dim})", _CatNet),
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
    )
    assert apply_ilm(cfg, bs=4), f"Fusion with {in_dim}->{out_dim} failed"


def test_custom_activation():
    class CustomActivation(torch.nn.Module):
        def forward(self, x):
            return x

    cfg = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        hf_model_name=default_text_test_model,
        custom_fusion_activation=("CustomActivation", CustomActivation()),
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
    )
    assert apply_ilm(cfg, bs=4), "Custom activation failed"


def test_custom_fusion():
    class CustomFusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.op = torch.mul

        def forward(self, x1, x2):
            return self.op(x1, x2)

    cfg = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        hf_model_name=default_text_test_model,
        custom_fusion_method=("CustomFusion", CustomFusion()),
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
    )
    assert apply_ilm(cfg, bs=4), "Custom fusion failed"


def test_custom_fusion_not_importable():
    cfg = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        hf_model_name=default_text_test_model,
        custom_fusion_method="not.importable",
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
    )
    with pytest.raises(ImportError):
        assert apply_ilm(cfg, bs=4), "Custom fusion failed"


def test_custom_fusion_not_available():
    cfg = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        hf_model_name=default_text_test_model,
        custom_fusion_method="not_available",
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
    )
    with pytest.raises(NameError):
        assert apply_ilm(cfg, bs=4), "Custom fusion failed"


def test_custom_fusion_importable_not_available():
    cfg = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        hf_model_name=default_text_test_model,
        custom_fusion_method="torch.not_available",
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
    )
    with pytest.raises(AttributeError):
        assert apply_ilm(cfg, bs=4), "Custom fusion failed"


def test_custom_fusion_wrong_input():
    with pytest.raises(ValueError):
        _ = ConfigILM.ILMConfiguration(
            timm_model_name=default_image_test_model,
            hf_model_name=default_text_test_model,
            custom_fusion_method=1,
            network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
        )


def test_custom_activation_not_importable():
    cfg = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        hf_model_name=default_text_test_model,
        custom_fusion_activation="not.importable",
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
    )
    with pytest.raises(ImportError):
        assert apply_ilm(cfg, bs=4), "Custom fusion failed"


def test_custom_activation_not_available():
    cfg = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        hf_model_name=default_text_test_model,
        custom_fusion_activation="not_available",
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
    )
    with pytest.raises(NameError):
        assert apply_ilm(cfg, bs=4), "Custom fusion failed"


def test_custom_activation_importable_not_available():
    cfg = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        hf_model_name=default_text_test_model,
        custom_fusion_activation="torch.not_available",
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
    )
    with pytest.raises(AttributeError):
        assert apply_ilm(cfg, bs=4), "Custom fusion failed"


def test_custom_activation_wrong_input():
    with pytest.raises(ValueError):
        _ = ConfigILM.ILMConfiguration(
            timm_model_name=default_image_test_model,
            hf_model_name=default_text_test_model,
            custom_fusion_activation=1,
            network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
        )


def test_tokenizer_exists():
    cfg = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        hf_model_name=default_text_test_model,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Keyword 'img_size' unknown. Trying to " "ignore and restart creation.",
        )

        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Tokenizer was initialized pretrained",
        )
        model = ConfigILM.ConfigILM(config=cfg)
        model.to(DEVICE)
    _ = model.get_tokenizer()


def test_tokenizer_not_exists():
    cfg = ConfigILM.ILMConfiguration(
        timm_model_name=default_image_test_model,
        network_type=ConfigILM.ILMType.IMAGE_CLASSIFICATION,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Keyword 'img_size' unknown. Trying to " "ignore and restart creation.",
        )
        model = ConfigILM.ConfigILM(config=cfg)
        model.to(DEVICE)

    with pytest.raises(AttributeError):
        _ = model.get_tokenizer()


def test_configuration_default():
    cfg = ConfigILM.ILMConfiguration(timm_model_name="resnet18")
    assert cfg.channels == 3, f"Channels should be 3 but is {cfg.channels}"
    assert cfg.classes == 10, f"Classes should be 10 but is {cfg.classes}"
    assert cfg.image_size == 120, f"Image size should be 120 but is {cfg.image_size}"
    assert cfg.fusion_in == 512, f"Fusion in should be 512 but is {cfg.fusion_in}"
    assert cfg.fusion_out == 512, f"Fusion out should be 512 but is {cfg.fusion_out}"
    assert cfg.fusion_method == torch.mul, f"Fusion method should be torch.mul but is {cfg.fusion_method}"
    assert cfg.hf_model_name is None, f"HF model name should be None but is {cfg.hf_model_name}"
    assert isinstance(
        cfg.fusion_activation, torch.nn.Tanh
    ), f"Fusion activation should be nn.Tanh but is {cfg.fusion_activation}"


def test_configurations_equal():
    cfg1 = ConfigILM.ILMConfiguration(timm_model_name="resnet18")
    cfg2 = ConfigILM.ILMConfiguration(timm_model_name="resnet18")
    assert cfg1 == cfg2, "Configurations should be equal"
    assert cfg1.dif(cfg2) == {}, "Configurations should be equal"

    dct = dataclasses.asdict(cfg1)
    assert ConfigILM.ILMConfiguration(**dct) == cfg1, "Configurations should be equal"
    assert ConfigILM.ILMConfiguration(**dct).dif(cfg1) == {}, "Configurations should be equal"

    dct2 = cfg1.as_dict()
    assert ConfigILM.ILMConfiguration(**dct2) == cfg1, "Configurations should be equal"
    assert ConfigILM.ILMConfiguration(**dct2).dif(cfg1) == {}, "Configurations should be equal"
    assert dct == dct2, "Configurations should be equal"


def test_configurations_change():
    cfg1 = ConfigILM.ILMConfiguration(timm_model_name="resnet18")
    dct = dataclasses.asdict(cfg1)
    dct["timm_model_name"] = "resnet50"
    assert ConfigILM.ILMConfiguration(**dct) != cfg1, "Configurations should not be equal"
    assert ConfigILM.ILMConfiguration(**dct).dif(cfg1) == {
        "timm_model_name": "resnet50"
    }, "Configurations should not be equal"

    dct2 = cfg1.as_dict()
    assert ConfigILM.ILMConfiguration(**dct2) == cfg1, "Configurations should be equal"
    assert ConfigILM.ILMConfiguration(**dct2).dif(cfg1) == {}, "Configurations should be equal"


def test_configuration_json_serializable():
    cfg = ConfigILM.ILMConfiguration(timm_model_name="resnet18")
    dct = cfg.as_dict()
    assert isinstance(dct, dict), "Configuration should be serializable to dict"

    json_str = cfg.to_json()
    assert isinstance(json_str, str), "Configuration should be serializable to json"

    cfg2 = ConfigILM.ILMConfiguration.from_json(json_str)
    assert cfg == cfg2, "Configuration should be equal after serialization"
