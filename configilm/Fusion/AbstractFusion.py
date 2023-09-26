# functions are partially based on
#   https://github.com/Cadene/vqa.pytorch/blob/master/vqa/models/fusion.py
from typing import Mapping
from typing import Optional

import torch.nn as nn


class AbstractFusion(nn.Module):
    def __init__(self, opt: Optional[Mapping] = None):
        super().__init__()
        if opt is None:
            opt = dict()
        self.opt = opt

    def forward(self, input_v, input_q):
        raise NotImplementedError
