# functions are partially based on
#   https://github.com/xiaoyuan1996/GaLR/blob/main/layers/GaLR.py
from typing import Mapping
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from configilm.Fusion.AbstractFusion import AbstractFusion
from configilm.Fusion.blocks import GuidedSelfAttention
from configilm.Fusion.blocks import SelfAttention


class MIDFusion(AbstractFusion):
    # based on https://github.com/xiaoyuan1996/GaLR/blob/main/layers/GaLR.py#L20
    def __init__(self, opt: Optional[Mapping] = None):
        super().__init__()
        self.opt = opt if opt is not None else {}
        embed_dim = self.opt["embed_dim"]
        num_heads = self.opt["num_heads"]
        fusion_dim = self.opt["fusion_dim"]
        attention_drop_out = self.opt["attention_drop_out"]
        fusion_drop_out = self.opt["fusion_drop_out"]

        # local trans
        self.l2l_SA = SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, drop_out=attention_drop_out
        )

        # global trans
        self.g2g_SA = SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, drop_out=attention_drop_out
        )

        # local correction
        self.g2l_GSA = GuidedSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, drop_out=attention_drop_out
        )

        # global supplement
        self.l2g_GSA = GuidedSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, drop_out=attention_drop_out
        )

        # dynamic fusion
        self.dynamic_weight = nn.Sequential(
            nn.Linear(embed_dim, fusion_dim),
            nn.Sigmoid(),
            nn.Dropout(fusion_drop_out),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=0),
        )

    def forward(self, global_feature, local_feature):
        global_feature = torch.unsqueeze(global_feature, dim=1)
        local_feature = torch.unsqueeze(local_feature, dim=1)

        # global trans
        global_feature = self.g2g_SA(global_feature)
        # local trans
        local_feature = self.l2l_SA(local_feature)

        # local correction
        local_feature = self.g2l_GSA(local_feature, global_feature)

        # global supplement
        global_feature = self.l2g_GSA(global_feature, local_feature)

        global_feature_t = torch.squeeze(global_feature, dim=1)
        local_feature_t = torch.squeeze(local_feature, dim=1)

        global_feature = F.sigmoid(local_feature_t) * global_feature_t
        local_feature = global_feature_t + local_feature_t

        # dynamic fusion
        feature_gl = global_feature + local_feature
        dynamic_weight = self.dynamic_weight(feature_gl)

        weight_global = (
            dynamic_weight[:, 0]
            .reshape(feature_gl.shape[0], -1)
            .expand_as(global_feature)
        )

        weight_local = (
            dynamic_weight[:, 0]
            .reshape(feature_gl.shape[0], -1)
            .expand_as(global_feature)
        )

        final_feature = weight_global * global_feature + weight_local * local_feature

        return final_feature
