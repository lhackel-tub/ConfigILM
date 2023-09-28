# based on
# https://github.com/Cadene/block.bootstrap.pytorch/blob/master/block/models/networks/fusions/fusions.py
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from configilm.Fusion.AbstractFusion import AbstractFusion


class MFB(AbstractFusion):
    def __init__(
        self,
        input_dims: Sequence[int],
        output_dim: int,
        mm_dim: int = 1200,
        factor: int = 2,
        activ_input: str = "relu",
        activ_output: str = "relu",
        normalize: bool = False,
        dropout_input: float = 0.0,
        dropout_pre_norm: float = 0.0,
        dropout_output: float = 0.0,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.mm_dim = mm_dim
        self.factor = factor
        self.output_dim = output_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_norm = dropout_pre_norm
        self.dropout_output = dropout_output
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim * factor)
        self.linear1 = nn.Linear(input_dims[1], mm_dim * factor)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor) -> torch.Tensor:
        x0, x1 = self._dual_linear(input_0, input_1)

        z = x0 * x1

        z = self._drop_view_norm(z)

        z = self.linear_out(z)

        if self.activ_output:
            z = getattr(F, self.activ_output)(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z
