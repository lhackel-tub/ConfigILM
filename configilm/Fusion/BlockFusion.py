# based on
# https://github.com/Cadene/block.bootstrap.pytorch/blob/master/block/models/networks/fusions/fusions.py
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from configilm.Fusion.AbstractFusion import AbstractFusion
from configilm.Fusion.AbstractFusion import get_chunks
from configilm.Fusion.AbstractFusion import get_sizes_list


class Block(AbstractFusion):
    def __init__(
        self,
        input_dims: Sequence[int],
        output_dim: int,
        mm_dim: int = 1600,
        chunks: int = 20,
        rank: int = 15,
        shared: bool = False,
        dropout_input: float = 0.0,
        dropout_pre_lin: float = 0.0,
        dropout_output: float = 0.0,
        pos_norm: str = "before_cat",
    ):
        """
        Initializes internal Module state of Block Fusion of "BLOCK: Bilinear
        Superdiagonal Fusion for Visual Question Answering and Visual Relationship
        Detection". Limits complexity of intermediate states by dividing the
        intermediate dimension into chunks (blocks). Inverse bottleneck expansion of
        multi-modal fusion is defined by rank.

        :param input_dims: Sequence of dimensions of different inputs. Only the first
                           two dimensions are used
        :param output_dim: Dimension of output tensor
        :param mm_dim: intermediate multi-modal dimension
        :param chunks: number of chunks the intermediate dimension will be divided into.
                       Has to be smaller or equal then mm_dim. If mm_dim is not
                       divisible by chunks, the last chunk will be smaller
        :param rank: Rank of input merging matrix, factor to the size calculated by
                     mm_dim/chunks
        :param shared: flag if the input mappings are shared between both inputs. Only
                       works, if all input_dims are equal
        :param dropout_input: Dropout rate of the inputs
        :param dropout_pre_lin: Dropout rate before linear mapping
        :param dropout_output: Dropout rate before the output
        :param pos_norm: position of normalization, has to be "before_cat" or
                         "after_cat"
        :returns: Block-Fusion torch.nn module
        """
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.chunks = chunks
        self.rank = rank
        self.shared = shared
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert pos_norm in ["before_cat", "after_cat"]
        self.pos_norm = pos_norm
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
        merge_linears0, merge_linears1 = [], []
        self.sizes_list = get_sizes_list(mm_dim, chunks)
        for size in self.sizes_list:
            ml0 = nn.Linear(size, size * rank)
            merge_linears0.append(ml0)
            if self.shared:
                ml1 = ml0
            else:
                ml1 = nn.Linear(size, size * rank)
            merge_linears1.append(ml1)
        self.merge_linears0 = nn.ModuleList(merge_linears0)
        self.merge_linears1 = nn.ModuleList(merge_linears1)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor) -> torch.Tensor:
        """
        Forward call to Block-Fusion

        :param input_0: first modality input
        :param input_1: second modality input
        :return: multi modality output
        """
        x0 = self.linear0(input_0)
        x1 = self.linear1(input_1)
        bsize = x1.size(0)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        x0_chunks = get_chunks(x0, self.sizes_list)
        x1_chunks = get_chunks(x1, self.sizes_list)
        zs = []

        for chunk_id, m0, m1 in zip(range(len(self.sizes_list)), self.merge_linears0, self.merge_linears1):
            x0_c = x0_chunks[chunk_id]
            x1_c = x1_chunks[chunk_id]
            m = m0(x0_c) * m1(x1_c)  # bsize x split_size*rank
            m = m.view(bsize, self.rank, -1)
            z = torch.sum(m, 1)
            if self.pos_norm == "before_cat":
                z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
                z = F.normalize(z, p=2)
            zs.append(z)

        return self._cat_norm_lin(zs)
