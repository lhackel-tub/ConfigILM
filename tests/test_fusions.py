import pytest
import torch

from configilm.Fusion.MIDFusion import MIDFusion
from configilm.Fusion.MLBFusion import MLBFusion


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
