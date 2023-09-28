import pytest
import torch

from configilm.Fusion.BlockFusion import Block
from configilm.Fusion.BlockTuckerFusion import BlockTucker
from configilm.Fusion.ConcatMLPFusion import ConcatMLP
from configilm.Fusion.LinearSumFusion import LinearSum
from configilm.Fusion.MFBFusion import MFB
from configilm.Fusion.MFHFusion import MFH
from configilm.Fusion.MLBFusion import MLB
from configilm.Fusion.MutanFusion import Mutan
from configilm.Fusion.TuckerFusion import Tucker


def assert_basic_fusion(fusion, dim_1, dim_2, dim_mm):
    for bs in [1, 2, 4, 8]:
        in_1 = torch.rand((bs, dim_1))
        in_2 = torch.rand((bs, dim_2))
        out = torch.rand((bs, dim_mm))

        y = fusion(in_1, in_2)
        assert (
            y.shape == out.shape
        ), f"Shape missmatch, should be {out.shape} but is {y.shape}"


@pytest.mark.parametrize(
    "dim_1, dim_2, dim_mm, dinter, do_in, do_pl, do_o, rank, pos_norm, chunks",
    [
        (d1, d2, dmm, dinter, do_in, do_pl, do_o, rank, pos_norm, chunks)
        for d1 in [128, 256]
        for d2 in [100, 256]
        for dmm in [64, 128]
        for dinter in [16, 32]
        for do_in in [0, 0.2]
        for do_pl in [0, 0.25]
        for do_o in [0, 0.3]
        for rank in [1, 4]
        for pos_norm in ["before_cat", "after_cat"]
        for chunks in [15, 16]
    ],
)
def test_block_fusion(
    dim_1, dim_2, dim_mm, dinter, do_in, do_pl, do_o, rank, pos_norm, chunks
):
    fusion = Block(
        input_dims=[dim_1, dim_2],
        output_dim=dim_mm,
        mm_dim=dinter,
        dropout_input=do_in,
        dropout_pre_lin=do_pl,
        dropout_output=do_o,
        rank=rank,
        chunks=chunks,  # must divide by mm_dim for normal operation
        pos_norm=pos_norm,
    )
    assert_basic_fusion(fusion=fusion, dim_1=dim_1, dim_2=dim_2, dim_mm=dim_mm)


@pytest.mark.parametrize(
    "dim_1, dim_2, shared",
    [
        (d1, d2, shared)
        for d1 in [128, 256]
        for d2 in [100, 256]
        for shared in [True, False]
    ],
)
def test_block_fusion_shared(dim_1, dim_2, shared):
    fusion = Block(
        input_dims=[dim_1, dim_2],
        output_dim=64,
        shared=shared,
        mm_dim=32,  # for faster runtime
        rank=4,
        chunks=16,  # must divide by mm_dim
    )
    if shared and dim_1 == dim_2 or not shared:
        assert_basic_fusion(fusion=fusion, dim_1=dim_1, dim_2=dim_2, dim_mm=64)
    else:
        with pytest.raises(RuntimeError):
            # shared with different dims fails
            assert_basic_fusion(fusion=fusion, dim_1=dim_1, dim_2=dim_2, dim_mm=64)


@pytest.mark.parametrize(
    "dim_1, dim_2, dim_mm, dinter, do_in, do_pl, do_o, pos_norm",
    [
        (d1, d2, dmm, dinter, do_in, do_pl, do_o, pos_norm)
        for d1 in [128, 256]
        for d2 in [100, 256]
        for dmm in [64, 128]
        for dinter in [16, 32]
        for do_in in [0, 0.2]
        for do_pl in [0, 0.25]
        for do_o in [0, 0.3]
        for pos_norm in ["before_cat", "after_cat"]
    ],
)
def test_block_tucker_fusion(
    dim_1, dim_2, dim_mm, dinter, do_in, do_pl, do_o, pos_norm
):
    fusion = BlockTucker(
        input_dims=[dim_1, dim_2],
        output_dim=dim_mm,
        mm_dim=dinter,
        dropout_input=do_in,
        dropout_pre_lin=do_pl,
        dropout_output=do_o,
        chunks=16,  # must divide by mm_dim
        pos_norm=pos_norm,
    )
    assert_basic_fusion(fusion=fusion, dim_1=dim_1, dim_2=dim_2, dim_mm=dim_mm)


@pytest.mark.parametrize(
    "dim_1, dim_2, shared",
    [
        (d1, d2, shared)
        for d1 in [128, 256]
        for d2 in [100, 256]
        for shared in [True, False]
    ],
)
def test_block_tucker_fusion_shared(dim_1, dim_2, shared):
    fusion = BlockTucker(
        input_dims=[dim_1, dim_2],
        output_dim=64,
        shared=shared,
        mm_dim=32,  # for faster runtime
        chunks=16,  # must divide by mm_dim
    )
    if shared and dim_1 == dim_2 or not shared:
        assert_basic_fusion(fusion=fusion, dim_1=dim_1, dim_2=dim_2, dim_mm=64)
    else:
        with pytest.raises(RuntimeError):
            # shared with different dims fails
            assert_basic_fusion(fusion=fusion, dim_1=dim_1, dim_2=dim_2, dim_mm=64)


@pytest.mark.parametrize(
    "dim_1, dim_2, dim_mm",
    [(d1, d2, dmm) for d1 in [128, 256] for d2 in [100, 256] for dmm in [64, 128]],
)
def test_concat_mlp_fusion(dim_1, dim_2, dim_mm):
    fusion = ConcatMLP(input_dims=[dim_1, dim_2], output_dim=dim_mm)
    assert_basic_fusion(fusion=fusion, dim_1=dim_1, dim_2=dim_2, dim_mm=dim_mm)


def test_concat_mlp_fusion_multi_d():
    dim_1 = 128
    dim_2 = 128
    dim_mm = 512
    fusion = ConcatMLP(input_dims=[dim_1, dim_2], output_dim=dim_mm)
    batch_size = 4
    in_1 = torch.rand((batch_size, 1, dim_1))
    in_2 = torch.rand((batch_size, dim_2))
    out = torch.rand((batch_size, 1, dim_mm))

    y = fusion(in_1, in_2)
    assert (
        y.shape == out.shape
    ), f"Shape missmatch, should be {out.shape} but is {y.shape}"

    in_1 = torch.rand((batch_size, dim_1))
    in_2 = torch.rand((batch_size, 1, dim_2))
    out = torch.rand((batch_size, 1, dim_mm))

    y = fusion(in_1, in_2)
    assert (
        y.shape == out.shape
    ), f"Shape missmatch, should be {out.shape} but is {y.shape}"


@pytest.mark.parametrize(
    "dim_1, dim_2, dim_mm, do_in, do_pl, do_o, norm",
    [
        (d1, d2, dmm, do_in, do_pl, do_o, norm)
        for d1 in [128, 256]
        for d2 in [100, 256]
        for dmm in [64, 128]
        for do_in in [0, 0.2]
        for do_pl in [0, 0.25]
        for do_o in [0, 0.3]
        for norm in [True, False]
    ],
)
def test_lin_sum_fusion(dim_1, dim_2, dim_mm, do_in, do_pl, do_o, norm):
    fusion = LinearSum(
        input_dims=[dim_1, dim_2],
        output_dim=dim_mm,
        dropout_input=do_in,
        dropout_pre_lin=do_pl,
        dropout_output=do_o,
        normalize=norm,
    )
    assert_basic_fusion(fusion=fusion, dim_1=dim_1, dim_2=dim_2, dim_mm=dim_mm)


@pytest.mark.parametrize(
    "dim_1, dim_2, dim_mm, do_in, do_pl, do_o, norm",
    [
        (d1, d2, dmm, do_in, do_pl, do_o, norm)
        for d1 in [128, 256]
        for d2 in [100, 256]
        for dmm in [64, 128]
        for do_in in [0, 0.2]
        for do_pl in [0, 0.25]
        for do_o in [0, 0.3]
        for norm in [True, False]
    ],
)
def test_mfb_fusion(dim_1, dim_2, dim_mm, do_in, do_pl, do_o, norm):
    fusion = MFB(
        input_dims=[dim_1, dim_2],
        output_dim=dim_mm,
        dropout_input=do_in,
        dropout_pre_norm=do_pl,
        dropout_output=do_o,
        normalize=norm,
    )
    assert_basic_fusion(fusion=fusion, dim_1=dim_1, dim_2=dim_2, dim_mm=dim_mm)


@pytest.mark.parametrize(
    "dim_1, dim_2, dim_mm, do_in, do_pl, do_o, norm",
    [
        (d1, d2, dmm, do_in, do_pl, do_o, norm)
        for d1 in [128, 256]
        for d2 in [100, 256]
        for dmm in [64, 128]
        for do_in in [0, 0.2]
        for do_pl in [0, 0.25]
        for do_o in [0, 0.3]
        for norm in [True, False]
    ],
)
def test_mfh_fusion(dim_1, dim_2, dim_mm, do_in, do_pl, do_o, norm):
    fusion = MFH(
        input_dims=[dim_1, dim_2],
        output_dim=dim_mm,
        dropout_input=do_in,
        dropout_pre_norm=do_pl,
        dropout_output=do_o,
        normalize=norm,
    )
    assert_basic_fusion(fusion=fusion, dim_1=dim_1, dim_2=dim_2, dim_mm=dim_mm)


@pytest.mark.parametrize(
    "dim_1, dim_2, dim_mm, do_in, do_pl, do_o, norm",
    [
        (d1, d2, dmm, do_in, do_pl, do_o, norm)
        for d1 in [128, 256]
        for d2 in [100, 256]
        for dmm in [64, 128]
        for do_in in [0, 0.2]
        for do_pl in [0, 0.25]
        for do_o in [0, 0.3]
        for norm in [True, False]
    ],
)
def test_mlb_fusion(dim_1, dim_2, dim_mm, do_in, do_pl, do_o, norm):
    fusion = MLB(
        input_dims=[dim_1, dim_2],
        output_dim=dim_mm,
        dropout_input=do_in,
        dropout_pre_lin=do_pl,
        dropout_output=do_o,
        normalize=norm,
    )
    assert_basic_fusion(fusion=fusion, dim_1=dim_1, dim_2=dim_2, dim_mm=dim_mm)


@pytest.mark.parametrize(
    "dim_1, dim_2, dim_mm, do_in, do_pl, do_o, norm, rank",
    [
        (d1, d2, dmm, do_in, do_pl, do_o, norm, rank)
        for d1 in [128, 256]
        for d2 in [100, 256]
        for dmm in [64, 128]
        for do_in in [0, 0.2]
        for do_pl in [0, 0.25]
        for do_o in [0, 0.3]
        for norm in [True, False]
        for rank in [1, 4]
    ],
)
def test_mutan_fusion(dim_1, dim_2, dim_mm, do_in, do_pl, do_o, norm, rank):
    fusion = Mutan(
        input_dims=[dim_1, dim_2],
        output_dim=dim_mm,
        dropout_input=do_in,
        dropout_pre_lin=do_pl,
        dropout_output=do_o,
        normalize=norm,
        rank=rank,
    )
    assert_basic_fusion(fusion=fusion, dim_1=dim_1, dim_2=dim_2, dim_mm=dim_mm)


@pytest.mark.parametrize(
    "dim_1, dim_2, shared",
    [
        (d1, d2, shared)
        for d1 in [128, 256]
        for d2 in [100, 256]
        for shared in [True, False]
    ],
)
def test_mutan_fusion_shared(dim_1, dim_2, shared):
    fusion = Mutan(
        input_dims=[dim_1, dim_2],
        output_dim=64,
        shared=shared,
        rank=4,  # for faster runtime
    )
    if shared and dim_1 == dim_2 or not shared:
        assert_basic_fusion(fusion=fusion, dim_1=dim_1, dim_2=dim_2, dim_mm=64)
    else:
        with pytest.raises(RuntimeError):
            # shared with different dims fails
            assert_basic_fusion(fusion=fusion, dim_1=dim_1, dim_2=dim_2, dim_mm=64)


@pytest.mark.parametrize(
    "dim_1, dim_2, dim_mm, dim_inter, do_in, do_pl, do_o, norm",
    [
        (d1, d2, dmm, dinter, do_in, do_pl, do_o, norm)
        for d1 in [128, 256]
        for d2 in [100, 256]
        for dmm in [64, 128]
        for dinter in [32, 48]
        for do_in in [0, 0.2]
        for do_pl in [0, 0.25]
        for do_o in [0, 0.3]
        for norm in [True, False]
    ],
)
def test_tucker_fusion(dim_1, dim_2, dim_mm, dim_inter, do_in, do_pl, do_o, norm):
    fusion = Tucker(
        input_dims=[dim_1, dim_2],
        output_dim=dim_mm,
        mm_dim=dim_inter,
        dropout_input=do_in,
        dropout_pre_lin=do_pl,
        dropout_output=do_o,
        normalize=norm,
    )
    assert_basic_fusion(fusion=fusion, dim_1=dim_1, dim_2=dim_2, dim_mm=dim_mm)


@pytest.mark.parametrize(
    "dim_1, dim_2, shared",
    [
        (d1, d2, shared)
        for d1 in [128, 256]
        for d2 in [100, 256]
        for shared in [True, False]
    ],
)
def test_tucker_fusion_shared(dim_1, dim_2, shared):
    fusion = Tucker(
        input_dims=[dim_1, dim_2],
        output_dim=64,
        shared=shared,
        mm_dim=32,  # for faster runtime
    )
    if shared and dim_1 == dim_2 or not shared:
        assert_basic_fusion(fusion=fusion, dim_1=dim_1, dim_2=dim_2, dim_mm=64)
    else:
        with pytest.raises(RuntimeError):
            # shared with different dims fails
            assert_basic_fusion(fusion=fusion, dim_1=dim_1, dim_2=dim_2, dim_mm=64)
