import pytest
import torch

from configilm.Fusion.MIDF import MIDF

embedding_dims = [32, 64, 256, 768]
heads = [1, 2, 4, 8]
fusion_dims = [128, 256]
attention_dropouts = [0.0, 0.25]
fusion_dropouts = [0.0, 0.5]


@pytest.mark.parametrize(
    "embed_dim, num_heads, fusion_dim, attention_drop_out, fusion_drop_out",
    [
        (e, h, f, da, df)
        for e in embedding_dims
        for h in heads
        for f in fusion_dims
        for da in attention_dropouts
        for df in fusion_dropouts
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
    opt = {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "fusion_dim": fusion_dim,
        "attention_drop_out": attention_drop_out,
        "fusion_drop_out": fusion_drop_out,
    }
    net = MIDF(opt)
    y = net(in_1, in_2)
    ref_tensor = torch.ones((batch_size, embed_dim))
    assert (
        y.shape == ref_tensor.shape
    ), f"Shape missmatch, should be {ref_tensor.shape} but is {y.shape}"
