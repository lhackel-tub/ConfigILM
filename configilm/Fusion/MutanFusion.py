# based on
# https://github.com/Cadene/block.bootstrap.pytorch/blob/master/block/models/networks/fusions/fusions.py
from typing import Sequence

import torch
import torch.nn as nn

from configilm.Fusion.AbstractFusion import AbstractFusion


class Mutan(AbstractFusion):
    def __init__(
        self,
        input_dims: Sequence[int],
        output_dim: int,
        mm_dim: int = 1600,
        rank: int = 15,
        shared: bool = False,
        normalize: bool = False,
        dropout_input: float = 0.0,
        dropout_pre_lin: float = 0.0,
        dropout_output: float = 0.0,
    ):
        """
        Initializes internal Module state of Multimodal-Tucker-Fusion of "Mutan:
        Multimodal tucker fusion for visual question answering". Uses Tucker
        decomposition for tensor complexity restriction. Inverse bottleneck expansion of
        multi-modal fusion is defined by rank.

        :param input_dims: Sequence of dimensions of different inputs. Only the first
                           two dimensions are used
        :param output_dim: Dimension of output tensor
        :param mm_dim: intermediate multi-modal dimension
        :param rank: Rank of input merging matrix, factor to the size calculated by
                     mm_dim
        :param shared: flag if the input mappings are shared between both inputs. Only
                       works, if all input_dims are equal
        :param normalize: flag if normalization should be applied or not
        :param dropout_input: Dropout rate of the inputs
        :param dropout_pre_lin: Dropout rate before linear mapping
        :param dropout_output: Dropout rate before the output
        :returns: Multimodal-Tucker-Fusion torch.nn module
        """
        super().__init__()
        self.input_dims = input_dims
        self.shared = shared
        self.mm_dim = mm_dim
        self.rank = rank
        self.output_dim = output_dim
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        self.normalize = normalize
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.merge_linear0 = nn.Linear(mm_dim, mm_dim * rank)
        if self.shared:
            self.linear1 = self.linear0
            self.merge_linear1 = self.merge_linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
            self.merge_linear1 = nn.Linear(mm_dim, mm_dim * rank)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor) -> torch.Tensor:
        """
        Forward call to Multimodal-Tucker-Fusion

        :param input_0: first modality input
        :param input_1: second modality input
        :return: multi modality output
        """
        x0, x1 = self._dual_linear(input_0, input_1, use_activation=False)

        m0 = self.merge_linear0(x0)
        m1 = self.merge_linear1(x1)
        m = m0 * m1
        m = m.view(-1, self.rank, self.mm_dim)
        z = torch.sum(m, 1)

        return self._norm_drop_lin(z)
