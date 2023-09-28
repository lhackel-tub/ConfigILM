# based on
# https://github.com/Cadene/block.bootstrap.pytorch/blob/master/block/models/networks/fusions/fusions.py
from typing import Iterable
from typing import List
from typing import Optional

import torch
import torch.nn as nn

from configilm.Fusion.blocks import MLP


class ConcatMLP(nn.Module):
    def __init__(
        self,
        input_dims: Iterable[int],
        output_dim: int,
        dimensions: Optional[List[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.input_dim = sum(input_dims)
        dimensions = [500, 500] if dimensions is None else dimensions
        self.dimensions = dimensions + [output_dim]
        self.activation = activation
        self.dropout = dropout
        # Modules
        self.mlp = MLP(self.input_dim, self.dimensions, self.activation, self.dropout)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor) -> torch.Tensor:
        if input_0.dim() == 3 and input_1.dim() == 2:
            input_1 = input_1.unsqueeze(1).reshape_as(input_0)
        if input_1.dim() == 3 and input_0.dim() == 2:
            input_0 = input_0.unsqueeze(1).reshape_as(input_1)
        z = torch.cat([input_0, input_1], dim=input_0.dim() - 1)
        z = self.mlp(z)
        return z
