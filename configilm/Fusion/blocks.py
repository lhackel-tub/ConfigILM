# functions are partially based on
#   https://github.com/xiaoyuan1996/GaLR/blob/main/layers/GaLR_utils.py
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FullyConnectedLayer, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MultiLayerPerceptron, self).__init__()

        self.fc = FullyConnectedLayer(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        out = self.fc(x)
        return self.linear(out)


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FeedForwardNetwork(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: Optional[int] = None, drop_out: float = 0.):
        super(FeedForwardNetwork, self).__init__()
        out_size = in_size if out_size is None else out_size
        self.mlp = MultiLayerPerceptron(
            in_size=in_size,
            mid_size=hidden_size,
            out_size=out_size,
            dropout_r=drop_out,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self-Attention ----
# ------------------------

class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, drop_out: float = 0.):
        super(SelfAttention, self).__init__()

        self.mhatt = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ffn = FeedForwardNetwork(in_size=embed_dim, hidden_size=embed_dim * 4, drop_out=drop_out)

        self.dropout1 = nn.Dropout(drop_out)
        self.norm1 = LayerNorm(embed_dim)

        self.dropout2 = nn.Dropout(drop_out)
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, x, x_mask=None):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, attn_mask=x_mask, need_weights=False)[0]
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Guided Self-Attention ----
# -------------------------------

class GuidedSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, drop_out: float = 0.):
        super(GuidedSelfAttention, self).__init__()

        self.mhatt1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.mhatt2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ffn = FeedForwardNetwork(in_size=embed_dim, hidden_size=embed_dim * 4, drop_out=drop_out)

        self.dropout1 = nn.Dropout(drop_out)
        self.norm1 = LayerNorm(embed_dim)

        self.dropout2 = nn.Dropout(drop_out)
        self.norm2 = LayerNorm(embed_dim)

        self.dropout3 = nn.Dropout(drop_out)
        self.norm3 = LayerNorm(embed_dim)

    def forward(self, x, y, x_mask=None, y_mask=None):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, attn_mask=x_mask, need_weights=False)[0]
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, attn_mask=y_mask, need_weights=False)[0]
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x
