import shutil
from pathlib import Path
from typing import Sequence

import pytest
import torch
from appdirs import user_cache_dir
from requests.exceptions import ReadTimeout  # type: ignore
from collections import OrderedDict

from configilm import ConfigILM


def get_classification_batch(
    img_shape: Sequence = (12, 120, 120),
    batch_size: int = 32,
    classes: int = 1000,
):
    assert len(img_shape) == 3
    v = torch.ones((batch_size, img_shape[0], img_shape[1], img_shape[2]))
    lbl = torch.ones((batch_size, classes))
    return [v, lbl]


def get_vqa_batch(
    img_shape: Sequence = (12, 120, 120),
    text_tokens: int = 64,
    batch_size: int = 32,
    classes: int = 1000,
):
    v, a = get_classification_batch(
        img_shape=img_shape, batch_size=batch_size, classes=classes
    )
    q = torch.ones((batch_size, text_tokens), dtype=torch.int32)
    return [v, q, a]


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 27, 13])
def test_bs(batch_size):
    config = ConfigILM.ILMConfiguration(timm_model_name="resnet18")
    x, y = get_classification_batch(
        (config.channels, config.image_size, config.image_size), batch_size
    )
    model = ConfigILM.ConfigILM(config=config)
    model(x)
    # pass


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
    "efficientnet_b4",
    "efficientnetv2_s",
    "efficientnetv2_xl",
    "gmlp_ti16_224",
    "mixer_b16_224",
    "mobilenetv2_035",
    "mobilenetv3_small_050",
    "mobilevit_xxs",
    "mobilevitv2_050",
    "pit_ti_224",
    "pit_ti_distilled_224",
    "poolformer_s12",
    "pvt_v2_b0",
    "pvt_v2_b4",
    "res2net50_14w_8s",
    "res2next50",
    "resnest14d",
    "resnet10t",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnetv2_50",
    "resnext26ts",
    "semobilevit_s",
    "ssl_resnet18",
    "tinynet_a",
    "vgg16",
    "visformer_tiny",
    "vit_small_patch16_18x2_224",
    "vit_tiny_patch16_224",
    "volo_d1_224",
    "wide_resnet50_2",
    "xcit_tiny_12_p8_224",
    "xcit_tiny_12_p8_224_dist",
]

tested_timm_models_224 = [
    "coat_tiny",
    "efficientformer_l1",
    "jx_nest_tiny",
    "levit_128",
    "levit_128s",
    "mvitv2_tiny",
    "swin_s3_tiny_224",
    "swin_tiny_patch4_window7_224",
    "swinv2_cr_tiny_224",
]

tested_timm_models_240 = [
    "crossvit_tiny_240",
]

tested_timm_models_256 = [
    "swinv2_tiny_window8_256",
]

tested_dims = [10, 100, 1000, 256, 512, 1024, 360, 1200]


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
    model = ConfigILM.ConfigILM(config=config)
    res = model(x)
    assert (
        res.shape == y.shape
    ), f"Shape is wrong: Should be {y.shape} but is {res.shape}"
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
            model = ConfigILM.ConfigILM(config=config)
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
        assert (
            res.shape == a.shape
        ), f"Shape is wrong: Should be {a.shape} but is {res.shape}"
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
        timm_model_name="resnet18",
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
        timm_model_name="resnet18",
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
        timm_model_name="resnet18",
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
        timm_model_name="resnet18",
        image_size=120,
        channels=12,
        hf_model_name="prajjwal1/bert-tiny",
        classes=10,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
        max_sequence_length=32,
        load_hf_if_available=False,
    )
    apply_ilm(config=config, bs=bs)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 6])
def test_v_wrong_batchshape(n):
    config = ConfigILM.ILMConfiguration(
        timm_model_name="resnet18", drop_rate=None, image_size=120, classes=10
    )
    shape = [16] * n
    x = torch.ones(shape)

    model = ConfigILM.ConfigILM(config=config)
    with pytest.raises(AssertionError):
        _ = model(x)


@pytest.mark.parametrize(
    "img, txt", [(i, t) for i in range(1, 6, 1) for t in range(1, 6, 1)]
)
def test_ilm_wrong_batch_parts_shape(img, txt):
    bs, cls = 8, 10
    config = ConfigILM.ILMConfiguration(
        timm_model_name="resnet18",
        image_size=120,
        channels=bs,
        hf_model_name="prajjwal1/bert-tiny",
        classes=cls,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
        max_sequence_length=32,
    )
    img_shape = [bs] * img
    txt_shape = [bs] * txt
    v = torch.ones(img_shape)
    q = torch.ones(txt_shape, dtype=torch.int32)
    a = torch.ones((bs, cls))

    model = ConfigILM.ConfigILM(config=config)
    if img != 4 or txt != 2:
        with pytest.raises(AssertionError):
            _ = model((v, q))
    else:
        res = model((v, q))
        assert (
            res.shape == a.shape
        ), f"Shape is wrong: Should be {a.shape} but is {res.shape}"
        assert res.shape == (
            bs,
            cls,
        ), f"Shape is wrong: Should be {(bs, cls)} but is {res.shape}"


def test_ilm_wrong_different_batch_sizes():
    config = ConfigILM.ILMConfiguration(
        timm_model_name="resnet18",
        image_size=120,
        channels=12,
        hf_model_name="prajjwal1/bert-tiny",
        classes=10,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
        max_sequence_length=32,
    )
    img_shape = [16] * 4
    txt_shape = [8] * 2
    v = torch.ones(img_shape)
    q = torch.ones(txt_shape, dtype=torch.int32)

    model = ConfigILM.ConfigILM(config=config)
    with pytest.raises(AssertionError):
        _ = model((v, q))


def test_ilm_wrong_seq_length_no_pooler():
    config = ConfigILM.ILMConfiguration(
        timm_model_name="resnet18",
        image_size=120,
        channels=12,
        hf_model_name="prajjwal1/bert-tiny",
        classes=10,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
        max_sequence_length=32,
        use_pooler_output=False,
    )
    img_shape = [16] * 4
    txt_shape = [16] * 2
    v = torch.ones(img_shape)
    q = torch.ones(txt_shape, dtype=torch.int32)

    model = ConfigILM.ConfigILM(config=config)
    with pytest.raises(RuntimeError):
        # This specific one config (hf_model_name="prajjwal1/bert-tiny")
        # will have a runtime error. This may not happen for other configs
        _ = model((v, q))


def test_ilm_wrong_input_length():
    config = ConfigILM.ILMConfiguration(
        timm_model_name="resnet18",
        image_size=120,
        channels=12,
        hf_model_name="prajjwal1/bert-tiny",
        classes=10,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION,
        max_sequence_length=32,
    )
    img_shape = [16] * 4
    txt_shape = [16] * 2
    v = torch.ones(img_shape)
    q = torch.ones(txt_shape, dtype=torch.int32)

    model = ConfigILM.ConfigILM(config=config)
    with pytest.raises(AssertionError):
        # This specific one config (hf_model_name="prajjwal1/bert-tiny") will have a
        # runtime error. This may not happen for other configs
        _ = model((v, q, q))


def test_failty_config():
    bs = 4
    config = ConfigILM.ILMConfiguration(
        timm_model_name="resnet18",
        image_size=120,
        channels=12,
        hf_model_name="prajjwal1/bert-tiny",
        classes=10,
        network_type=-5,
        max_sequence_length=32,
        load_hf_if_available=False,
    )
    v, q, a = get_vqa_batch(
        (config.channels, config.image_size, config.image_size),
        batch_size=bs,
        classes=config.classes,
        text_tokens=config.max_sequence_length,
    )
    model = ConfigILM.ConfigILM(config=config)
    with pytest.raises(ValueError):
        _ = model((v, q))


def test_failed_network_connection_in_download(mocker):
    hf_model = "distilbert-base-uncased"
    path = Path(user_cache_dir(appname="configilm")).joinpath(
        "pretrained_models", "huggingface_models", hf_model
    )
    shutil.rmtree(path)

    mocker.patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        new=lambda x: (_ for _ in ()).throw(FileNotFoundError),
    )
    bs = 4
    config = ConfigILM.ILMConfiguration(
        timm_model_name="resnet18",
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
    hf_model = "prajjwal1/bert-tiny"
    path = Path(user_cache_dir(appname="configilm")).joinpath(
        "pretrained_models", "huggingface_models", hf_model
    )
    if not path.exists():
        pytest.skip(
            f"HF model '{hf_model}' to test was not downloaded before test. "
            f"Cannot check cache."
        )

    mocker.patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        new=lambda x: (_ for _ in ()).throw(ConnectionError),
    )
    bs = 4
    config = ConfigILM.ILMConfiguration(
        timm_model_name="resnet18",
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
    path = Path(user_cache_dir(appname="configilm")).joinpath(
        "pretrained_models", "huggingface_models", hf_model
    )
    if path.exists():
        pytest.skip(
            f"HF model '{hf_model}' exists offline already. Cannot check download."
        )

    mocker.patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        new=lambda x: (_ for _ in ()).throw(HTTPError),
    )
    bs = 4
    config = ConfigILM.ILMConfiguration(
        timm_model_name="resnet18",
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
        timm_model_name="resnet18", classes=1000, load_timm_if_available=True
    )

    model = ConfigILM.ConfigILM(config=cfg)
    model.eval()

    in_t = torch.ones((1, 3, 120, 120))
    out_t = model(in_t)

    assert torch.all(
        torch.isclose(
            out_t[0][:5],
            torch.tensor([-0.2201, -0.1476, -1.3507, -1.2310, -0.3701]),
            atol=0.0001,
        )
    )


@pytest.mark.parametrize("dim", tested_dims)
def test_fusion_same_dim_explicit(dim):
    """
    Tests if fusion functions with the same input and output dimensions work
    """
    cfg = ConfigILM.ILMConfiguration(
        timm_model_name="resnet18",
        hf_model_name="prajjwal1/bert-tiny",
        fusion_in=dim,
        fusion_out=dim,
        fusion_method=torch.mul,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION
    )
    assert apply_ilm(cfg, bs=4), f"Fusion with {dim}->{dim} failed"


@pytest.mark.parametrize("dim", tested_dims)
def test_fusion_same_dim_implicit(dim):
    """
    Tests if fusion functions with the same input and output dimensions work
    """
    cfg = ConfigILM.ILMConfiguration(
        timm_model_name="resnet18",
        hf_model_name="prajjwal1/bert-tiny",
        fusion_in=dim,
        fusion_out=None,
        fusion_method=torch.mul,
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION
    )
    assert apply_ilm(cfg, bs=4), f"Fusion with {dim}->{dim} failed"


@pytest.mark.parametrize(
    "in_dim, out_dim", [(i, o) for i in tested_dims for o in tested_dims]
)
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
                        ("FFN", torch.nn.Linear(2*in_dim_cat, out_dim_lin)),
                        ("Activation", torch.nn.ReLU())
                    ]
                )
            )

        def forward(self, x1, x2):
            # concat vectors, not batches
            x = torch.cat((x1, x2), dim=-1)
            return self.net(x)

    cfg = ConfigILM.ILMConfiguration(
        timm_model_name="resnet18",
        hf_model_name="prajjwal1/bert-tiny",
        fusion_in=in_dim,
        fusion_out=out_dim,
        fusion_method=_CatNet(in_dim_cat=in_dim, out_dim_lin=out_dim),
        network_type=ConfigILM.ILMType.VQA_CLASSIFICATION
    )
    assert apply_ilm(cfg, bs=4), f"Fusion with {in_dim}->{out_dim} failed"
