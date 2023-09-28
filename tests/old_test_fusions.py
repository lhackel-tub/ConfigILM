import pytest
import torch

from configilm.Fusion._old.MIDFusion import MIDFusion
from configilm.Fusion._old.MLBFusion import MLBFusion
from configilm.Fusion._old.MUTANFusion import MUTANFusion
from configilm.Fusion._old.MUTANFusion import MUTANFusion2d


@pytest.mark.parametrize(
    "embed_dim, num_heads, fusion_dim, attention_drop_out, fusion_drop_out",
    [
        (e, h, f, da, df)
        for e in [32, 256, 768]
        for h in [1, 2, 8]
        for f in [128, 256]
        for da in [0.0, 0.25]
        for df in [0.0, 0.5]
    ],
)
def test_MIDF(
    embed_dim: int,
    num_heads: int,
    fusion_dim: int,
    attention_drop_out: float,
    fusion_drop_out: float,
):
    batch_size = 4
    in_1 = torch.rand((batch_size, embed_dim))
    in_2 = torch.rand((batch_size, embed_dim))
    ref_tensor = torch.ones((batch_size, embed_dim))
    opt = {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "fusion_dim": fusion_dim,
        "attention_drop_out": attention_drop_out,
        "fusion_drop_out": fusion_drop_out,
    }
    net = MIDFusion(opt)
    y = net(in_1, in_2)
    assert (
        y.shape == ref_tensor.shape
    ), f"Shape missmatch, should be {ref_tensor.shape} but is {y.shape}"


@pytest.mark.parametrize(
    "dim_v, test_dim, dim_h",
    [
        (v, ref, h)
        for v in [128, 256, 378, None]
        for ref in [100, 200, 256, 350, 500]
        for h in [256, 378]
    ],
)
def test_MLBF_v(dim_v, test_dim, dim_h):
    opt = {
        "dim_q": 256,
        "dim_h": dim_h,
        "activation_v": "relu",
        "activation_q": "relu",
        "dropout_v": 0.1,
        "dropout_q": 0.1,
    }
    if dim_v is not None:
        opt["dim_v"] = dim_v
    net = MLBFusion(opt)
    batch_size = 4
    in_2 = torch.rand((batch_size, 256))
    d_ref = dim_h if dim_v is not None else 256
    ref_tensor = torch.ones((batch_size, d_ref))

    in_1 = torch.rand((batch_size, test_dim))
    if dim_v is None:
        if test_dim == 256 and dim_h == test_dim:
            # works without mapping
            y = net(in_1, in_2)
            assert (
                y.shape == ref_tensor.shape
            ), f"Shape missmatch, should be {ref_tensor.shape} but is {y.shape}"
        else:
            with pytest.raises(RuntimeError):
                _ = net(in_1, in_2)
    else:
        if dim_v != test_dim:
            # fails in mapping -> input to network is not correct
            with pytest.raises(RuntimeError):
                _ = net(in_1, in_2)
        else:
            # works due to mapping
            y = net(in_1, in_2)
            assert (
                y.shape == ref_tensor.shape
            ), f"Shape missmatch, should be {ref_tensor.shape} but is {y.shape}"


@pytest.mark.parametrize(
    "dim_q, test_dim, dim_h",
    [
        (q, ref, h)
        for q in [128, 256, 378, None]
        for ref in [100, 200, 256, 350, 500]
        for h in [256, 378]
    ],
)
def test_MLBF_q(dim_q, test_dim, dim_h):
    opt = {
        "dim_v": 256,
        "dim_h": dim_h,
        "activation_v": "relu",
        "activation_q": "relu",
        "dropout_v": 0.1,
        "dropout_q": 0.1,
    }
    if dim_q is not None:
        opt["dim_q"] = dim_q
    net = MLBFusion(opt)
    batch_size = 4
    in_2 = torch.rand((batch_size, test_dim))
    d_ref = dim_h if dim_q is not None else 256
    ref_tensor = torch.ones((batch_size, d_ref))

    in_1 = torch.rand((batch_size, 256))
    if dim_q is None:
        if test_dim == 256 and dim_h == test_dim:
            # works without mapping
            y = net(in_1, in_2)
            assert (
                y.shape == ref_tensor.shape
            ), f"Shape missmatch, should be {ref_tensor.shape} but is {y.shape}"
        else:
            with pytest.raises(RuntimeError):
                _ = net(in_1, in_2)
    else:
        if dim_q != test_dim:
            # fails in mapping -> input to network is not correct
            with pytest.raises(RuntimeError):
                _ = net(in_1, in_2)
        else:
            # works due to mapping
            y = net(in_1, in_2)
            assert (
                y.shape == ref_tensor.shape
            ), f"Shape missmatch, should be {ref_tensor.shape} but is {y.shape}"


@pytest.mark.parametrize(
    "dim_v, dim_q, dim_mm",
    [
        (dv, dq, dmm)
        for dv in [128, 256]
        for dq in [100, 256]
        for dmm in [100, 128, 256, 378]
    ],
)
def test_MUTANF(dim_v: int, dim_q: int, dim_mm: int):
    opt = {
        "dim_v": dim_v,
        "dim_hv": 200,
        "dim_q": dim_q,
        "dim_hq": 201,
        "R": 3,
        "dim_mm": dim_mm,
        "dropout_v": 0.2,
        "dropout_q": 0.25,
        "activation_v": "selu",
        "activation_q": "relu",
        "dropout_hv": 0.3,
        "dropout_hq": 0.35,
        "activation_hv": "relu",
        "activation_hq": "elu",
        "activation_mm": "gelu",
    }
    net = MUTANFusion(opt)
    batch_size = 4
    in_1 = torch.rand((batch_size, dim_v))
    in_2 = torch.rand((batch_size, dim_q))
    out = torch.rand((batch_size, dim_mm))
    y = net(in_1, in_2)
    assert (
        y.shape == out.shape
    ), f"Shape missmatch, should be {out.shape} but is {y.shape}"


@pytest.mark.parametrize(
    "dim_v, dim_q, dim_mm",
    [
        (dv, dq, dmm)
        for dv in [128, 256]
        for dq in [100, 256]
        for dmm in [100, 128, 256, 378]
    ],
)
def test_MUTANF_no_v_embedd(dim_v: int, dim_q: int, dim_mm: int):
    opt = {
        "dim_v": dim_v,
        "dim_hv": 256,
        "dim_q": dim_q,
        "dim_hq": 201,
        "R": 3,
        "dim_mm": dim_mm,
        "dropout_v": 0.2,
        "dropout_q": 0.25,
        "activation_v": "selu",
        "activation_q": "relu",
        "dropout_hv": 0.3,
        "dropout_hq": 0.35,
        "activation_hv": "relu",
        "activation_hq": "elu",
        "activation_mm": "gelu",
    }
    net = MUTANFusion(opt, visual_embedding=False)
    batch_size = 4
    in_1 = torch.rand((batch_size, dim_v))
    in_2 = torch.rand((batch_size, dim_q))
    out = torch.rand((batch_size, dim_mm))
    if dim_v == 256:
        # we should not need embedding
        y = net(in_1, in_2)
        assert (
            y.shape == out.shape
        ), f"Shape missmatch, should be {out.shape} but is {y.shape}"
    else:
        with pytest.raises(RuntimeError):
            _ = net(in_1, in_2)


@pytest.mark.parametrize(
    "dim_v, dim_q, dim_mm",
    [
        (dv, dq, dmm)
        for dv in [128, 256]
        for dq in [100, 256]
        for dmm in [100, 128, 256, 378]
    ],
)
def test_MUTANF_no_q_embedd(dim_v: int, dim_q: int, dim_mm: int):
    opt = {
        "dim_v": dim_v,
        "dim_hv": 200,
        "dim_q": dim_q,
        "dim_hq": 256,
        "R": 3,
        "dim_mm": dim_mm,
        "dropout_v": 0.2,
        "dropout_q": 0.25,
        "activation_v": "selu",
        "activation_q": "relu",
        "dropout_hv": 0.3,
        "dropout_hq": 0.35,
        "activation_hv": "relu",
        "activation_hq": "elu",
        "activation_mm": "gelu",
    }
    net = MUTANFusion(opt, question_embedding=False)
    batch_size = 4
    in_1 = torch.rand((batch_size, dim_v))
    in_2 = torch.rand((batch_size, dim_q))
    out = torch.rand((batch_size, dim_mm))
    if dim_q == 256:
        # we should not need embedding
        y = net(in_1, in_2)
        assert (
            y.shape == out.shape
        ), f"Shape missmatch, should be {out.shape} but is {y.shape}"
    else:
        with pytest.raises(RuntimeError):
            _ = net(in_1, in_2)


def test_MUTANF_dim():
    dim_v, dim_q, dim_mm = 256, 257, 128
    opt = {
        "dim_v": dim_v,
        "dim_hv": 200,
        "dim_q": dim_q,
        "dim_hq": 256,
        "R": 3,
        "dim_mm": dim_mm,
        "dropout_v": 0.2,
        "dropout_q": 0.25,
        "activation_v": "selu",
        "activation_q": "relu",
        "dropout_hv": 0.3,
        "dropout_hq": 0.35,
        "activation_hv": "relu",
        "activation_hq": "elu",
        "activation_mm": "gelu",
    }
    net = MUTANFusion(opt)
    batch_size = 4
    in_1 = torch.rand((batch_size, dim_v, 1))
    in_2 = torch.rand((batch_size, dim_q))
    with pytest.raises(ValueError):
        _ = net(in_1, in_2)


def test_MUTAN2dF():
    dim_v1 = 100
    dim_v2 = 200
    dim_q1 = 300
    dim_q2 = 400
    dim_mm = 256
    batch_size = 4
    opt = {
        "dim_v": dim_v2,
        "dim_hv": 256,
        "dim_q": dim_q1 * dim_q2 // dim_v1,
        "dim_hq": 512,
        "R": 3,
        "dim_mm": dim_mm,
        "dropout_v": 0.2,
        "dropout_q": 0.25,
        "activation_v": "selu",
        "activation_q": "relu",
        "dropout_hv": 0.3,
        "dropout_hq": 0.35,
        "activation_hv": "relu",
        "activation_hq": "elu",
        "activation_mm": "gelu",
    }
    net = MUTANFusion2d(opt)
    in_1 = torch.rand((batch_size, dim_v1, dim_v2))
    in_2 = torch.rand((batch_size, dim_q1, dim_q2))
    out = torch.rand((batch_size, dim_v1, dim_mm))
    y = net(in_1, in_2)
    assert (
        y.shape == out.shape
    ), f"Shape missmatch, should be {out.shape} but is {y.shape}"
