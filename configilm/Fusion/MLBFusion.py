# functions are partially based on
#   https://github.com/Cadene/vqa.pytorch/blob/master/vqa/models/fusion.py
import torch
import torch.nn.functional as F
from torch import nn

from configilm.Fusion.AbstractFusion import AbstractFusion
from configilm.util import Messages


class MLBFusion(AbstractFusion):
    def __init__(self, opt):
        super().__init__(opt)
        # Modules
        if "dim_v" in self.opt:
            self.linear_v = nn.Linear(self.opt["dim_v"], self.opt["dim_h"])
        else:
            Messages.warn("MLBFusion: no visual embedding before fusion")

        if "dim_q" in self.opt:
            self.linear_q = nn.Linear(self.opt["dim_q"], self.opt["dim_h"])
        else:
            Messages.warn("MLBFusion: no question embedding before fusion")

    def forward(self, input_v, input_q):
        # visual (image features)
        if "dim_v" in self.opt:
            x_v = F.dropout(input_v, p=self.opt["dropout_v"], training=self.training)
            x_v = self.linear_v(x_v)
            if "activation_v" in self.opt:
                x_v = getattr(F, self.opt["activation_v"])(x_v)
        else:
            x_v = input_v
        # question (text features)
        if "dim_q" in self.opt:
            x_q = F.dropout(input_q, p=self.opt["dropout_q"], training=self.training)
            x_q = self.linear_q(x_q)
            if "activation_q" in self.opt:
                x_q = getattr(F, self.opt["activation_q"])(x_q)
        else:
            x_q = input_q
        # hadamard product
        x_mm = torch.mul(x_q, x_v)
        return x_mm
