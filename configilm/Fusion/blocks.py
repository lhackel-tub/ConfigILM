# functions are partially based on
#   https://github.com/xiaoyuan1996/GaLR/blob/main/layers/GaLR_utils.py
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init


class MLP(nn.Module):
    def __init__(self, input_dim, dimensions, activation="relu", dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.dimensions = dimensions
        self.activation = activation
        self.dropout = dropout
        # Modules
        self.linears = nn.ModuleList([nn.Linear(input_dim, dimensions[0])])
        for din, dout in zip(dimensions[:-1], dimensions[1:]):
            self.linears.append(nn.Linear(din, dout))

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears) - 1:
                x = F.__dict__[self.activation](x)
                if self.dropout > 0:
                    x = F.dropout(x, self.dropout, training=self.training)
        return x


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super().__init__()
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
    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        out_size: Optional[int] = None,
        drop_out: float = 0.0,
    ):
        super().__init__()
        out_size = in_size if out_size is None else out_size
        self.mlp = MLP(
            input_dim=in_size,
            dimensions=[hidden_size, out_size],
            dropout=drop_out,
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self-Attention ----
# ------------------------


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, drop_out: float = 0.0):
        super().__init__()

        self.mhatt = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ffn = FeedForwardNetwork(in_size=embed_dim, hidden_size=embed_dim * 4, drop_out=drop_out)

        self.dropout1 = nn.Dropout(drop_out)
        self.norm1 = LayerNorm(embed_dim)

        self.dropout2 = nn.Dropout(drop_out)
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, x, x_mask=None):
        x = self.norm1(x + self.dropout1(self.mhatt(x, x, x, attn_mask=x_mask, need_weights=False)[0]))

        x = self.norm2(x + self.dropout2(self.ffn(x)))

        return x


# -------------------------------
# ---- Guided Self-Attention ----
# -------------------------------


class GuidedSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, drop_out: float = 0.0):
        super().__init__()

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
        x = self.norm1(x + self.dropout1(self.mhatt1(x, x, x, attn_mask=x_mask, need_weights=False)[0]))

        x = self.norm2(x + self.dropout2(self.mhatt2(y, y, x, attn_mask=y_mask, need_weights=False)[0]))

        x = self.norm3(x + self.dropout3(self.ffn(x)))

        return x
