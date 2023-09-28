# based on
# https://github.com/Cadene/block.bootstrap.pytorch/blob/master/block/models/networks/fusions/fusions.py
from typing import Sequence

import torch
import torch.nn as nn

from configilm.Fusion.AbstractFusion import AbstractFusion


class LinearSum(AbstractFusion):
    def __init__(
        self,
        input_dims: Sequence[int],
        output_dim: int,
        mm_dim: int = 1200,
        activ_input: str = "relu",
        activ_output: str = "relu",
        normalize: bool = False,
        dropout_input: float = 0.0,
        dropout_pre_lin: float = 0.0,
        dropout_output: float = 0.0,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor) -> torch.Tensor:
        x0, x1 = self._dual_linear(input_0, input_1)
        z = x0 + x1
        return self._norm_lin_out(z)
