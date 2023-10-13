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
        """
        Initializes internal Module state of LinearSum-Fusion. Passes both modalities
        independent of each other through a linear layer to map into a common
        dimension. Results are added (point-wise) and mapped to the output dimension.
        Normalization, dropout and activations are applied if set.

        :param input_dims: Sequence of dimensions of different inputs. Only the first
                           two dimensions are used
        :param output_dim: Dimension of output tensor
        :param mm_dim: intermediate multi-modal dimension
        :param activ_input: activation function after the first linear layer
        :param activ_output: activation function after the second linear layer
        :param normalize: flag if normalization should be applied or not
        :param dropout_input: Dropout rate of the inputs
        :param dropout_pre_lin: Dropout rate before linear mapping
        :param dropout_output: Dropout rate before the output
        :returns: LinearSum-Fusion torch.nn module
        """
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
        """
        Forward call to Pointwise-Linear-Addition-Fusion

        :param input_0: first modality input
        :param input_1: second modality input
        :return: multi modality output
        """
        x0, x1 = self._dual_linear(input_0, input_1)
        z = x0 + x1
        return self._norm_lin_out(z)
