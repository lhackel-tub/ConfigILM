# functions are partially based on
#   https://github.com/Cadene/vqa.pytorch/blob/master/vqa/models/fusion.py
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sizes_list(dim, chunks):
    split_size = (dim + chunks - 1) // chunks
    sizes_list = [split_size] * chunks
    sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim)  # Adjust last
    assert sum(sizes_list) == dim
    if sizes_list[-1] < 0:
        n_miss = sizes_list[-2] - sizes_list[-1]
        sizes_list[-1] = sizes_list[-2]
        for j in range(n_miss):
            sizes_list[-j - 1] -= 1
        assert sum(sizes_list) == dim, "Chunk sizes don't add up correctly"
        assert min(sizes_list) > 0, "Chunk size must be smaller or equal to dim"
    return sizes_list


def get_chunks(x, sizes):
    out = []
    begin = 0
    for s in sizes:
        y = x.narrow(1, begin, s)
        out.append(y)
        begin += s
    return out


class AbstractFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def _dual_linear(
        self, input_0: torch.Tensor, input_1: torch.Tensor, use_activation: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.linear0(input_0)
        x1 = self.linear1(input_1)

        if use_activation and self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        return x0, x1

    def _norm_lin_out(self, z: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)

        z = self.linear_out(z)

        if self.activ_output:
            z = getattr(F, self.activ_output)(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z

    def _view_norm(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(z.size(0), self.mm_dim, self.factor)
        z = z.sum(2)

        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)
        return z

    def _drop_view_norm(self, z: torch.Tensor) -> torch.Tensor:
        if self.dropout_pre_norm > 0:
            z = F.dropout(z, p=self.dropout_pre_norm, training=self.training)
        return self._view_norm(z)

    def _drop_lin(self, z: torch.Tensor) -> torch.Tensor:
        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)

        z = self.linear_out(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z

    def _norm_drop_lin(self, z: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)

        return self._drop_lin(z)

    def _cat_norm_lin(self, zs: List[torch.Tensor]) -> torch.Tensor:
        z = torch.cat(zs, 1)
        if self.pos_norm == "after_cat":
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)

        return self._drop_lin(z)

    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
