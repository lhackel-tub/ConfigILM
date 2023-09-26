import torch.nn as nn
from typing import Mapping, Optional


class AbstractFusion(nn.Module):

    def __init__(self, opt: Optional[Mapping] = None):
        super(AbstractFusion, self).__init__()
        if opt is None:
            opt = dict()
        self.opt = opt

    def forward(self, input_v, input_q):
        raise NotImplementedError
