# based on
# https://github.com/Cadene/block.bootstrap.pytorch/blob/master/block/models/networks/fusions/fusions.py
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from configilm.Fusion.AbstractFusion import AbstractFusion


class MFH(AbstractFusion):
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
        """
        Initializes internal Module state of Generalized Multimodal Factorized
        High-order Pooling Fusion of "Beyond Bilinear: Generalized Multimodal Factorized
        High-order Pooling for Visual Question Answering".
        Linear mapping of inputs to higher dimension paired with a point-wise
        multiplications, poolings, additional linear mappings and an output mapping.

        :param input_dims: Sequence of dimensions of different inputs. Only the first
                           two dimensions are used
        :param output_dim: Dimension of output tensor
        :param mm_dim: intermediate multi-modal dimension
        :param factor: rank of linear input mappings / pooling rate on output
        :param activ_input: activation function after the first linear layer
        :param activ_output: activation function after the second linear layer
        :param dropout_input: Dropout rate of the inputs
        :param dropout_pre_norm: Dropout rate before normalization
        :param dropout_output: Dropout rate before the output
        :param normalize: flag if normalization should be applied or not
        :returns: MFH-Fusion torch.nn module
        """
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.factor = factor
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_norm = dropout_pre_norm
        self.dropout_output = dropout_output
        # Modules
        self.linear0_0 = nn.Linear(input_dims[0], mm_dim * factor)
        self.linear1_0 = nn.Linear(input_dims[1], mm_dim * factor)
        self.linear0 = self.linear0_0  # copy attribute for reuse in functions
        self.linear1 = self.linear1_0  # copy attribute for reuse in functions
        self.linear0_1 = nn.Linear(input_dims[0], mm_dim * factor)
        self.linear1_1 = nn.Linear(input_dims[1], mm_dim * factor)
        self.linear_out = nn.Linear(mm_dim * 2, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _activate_input(self, x0, x1):
        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)
        return x0, x1

    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor) -> torch.Tensor:
        """
        Forward call to Generalized Multimodal Factorized High-order Pooling Fusion

        :param input_0: first modality input
        :param input_1: second modality input
        :return: multi modality output
        """
        x0, x1 = self._dual_linear(input_0, input_1)

        z_0_skip = x0 * x1

        if self.dropout_pre_norm:
            z_0_skip = F.dropout(z_0_skip, p=self.dropout_pre_norm, training=self.training)

        z_0 = self._view_norm(z_0_skip)

        x0 = self.linear0_1(input_0)
        x1 = self.linear1_1(input_1)

        x0, x1 = self._activate_input(x0, x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z_1 = x0 * x1 * z_0_skip

        z_1 = self._drop_view_norm(z_1)

        cat_dim = z_0.dim() - 1
        z = torch.cat([z_0, z_1], cat_dim)
        z = self.linear_out(z)

        if self.activ_output:
            z = getattr(F, self.activ_output)(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z
