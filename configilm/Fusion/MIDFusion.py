# functions are partially based on
#   https://github.com/xiaoyuan1996/GaLR/blob/main/layers/GaLR.py
import torch
import torch.nn.functional as F
from torch import nn

from configilm.Fusion.AbstractFusion import AbstractFusion
from configilm.Fusion.blocks import GuidedSelfAttention
from configilm.Fusion.blocks import SelfAttention


class MIDF(AbstractFusion):
    # based on https://github.com/xiaoyuan1996/GaLR/blob/main/layers/GaLR.py#L20
    def __init__(
        self,
        input_dim: int,
        mm_dim: int = 1200,
        heads: int = 2,
        activ_fusion: str = "sigmoid",
        dropout_attention: float = 0.0,
        dropout_output: float = 0.0,
    ):
        """
        Initializes internal Module state of Multi-Level Information Dynamic fusion of
        "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local
        Information".
        Self-Attention and Guided Attention followed by a Multi-Layer Perceptron.
        The output of the MLP is used to weight the different modality features.

        :param input_dim: Dimension of different inputs. All the inputs are of the same
                          dimension
        :param mm_dim: intermediate multi-modal dimension
        :param heads: number of heads in the self-attention layer
        :param activ_fusion: activation function in the MLP
        :param dropout_attention: Dropout rate of the inputs in attention layer
        :param dropout_output: Dropout rate before the output
        :returns: MID-Fusion torch.nn module
        """
        super().__init__()
        # local trans
        self.l2l_SA = SelfAttention(embed_dim=input_dim, num_heads=heads, drop_out=dropout_attention)

        # global trans
        self.g2g_SA = SelfAttention(embed_dim=input_dim, num_heads=heads, drop_out=dropout_attention)

        # local correction
        self.g2l_GSA = GuidedSelfAttention(embed_dim=input_dim, num_heads=heads, drop_out=dropout_attention)

        # global supplement
        self.l2g_GSA = GuidedSelfAttention(embed_dim=input_dim, num_heads=heads, drop_out=dropout_attention)

        self.linear1 = nn.Linear(input_dim, mm_dim)
        self.activ_fusion = getattr(F, activ_fusion)

        # dynamic fusion
        self.dynamic_weight = nn.Sequential(
            nn.Dropout(dropout_output),
            nn.Linear(mm_dim, 2),
            nn.Softmax(dim=-1),
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

        global_feature = self.activ_fusion(local_feature_t) * global_feature_t
        local_feature = global_feature_t + local_feature_t

        # dynamic fusion
        feature_gl = global_feature + local_feature
        dynamic_weight = self.linear1(feature_gl)
        dynamic_weight = self.activ_fusion(dynamic_weight)
        dynamic_weight = self.dynamic_weight(dynamic_weight)

        weight_global = dynamic_weight[:, 0].reshape(feature_gl.shape[0], -1).expand_as(global_feature)

        weight_local = dynamic_weight[:, 0].reshape(feature_gl.shape[0], -1).expand_as(global_feature)

        final_feature = weight_global * global_feature + weight_local * local_feature

        return final_feature
