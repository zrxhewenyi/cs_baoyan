import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def causal_mask(x):
    # 少用API

    # (B, seq_len, v_dim)
    mask_matrix = torch.zeros_like(x)

    # mask_matri = torch.tril(mask_matrix)
    aaa = torch.arange(x.shape[1])

    mask_matrix = aaa.unsqueeze(1) < aaa.unsqueeze(0)

    x = x.masked_fill(mask_matrix, -math.inf)

    return x

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, qk_dim, v_dim, dropout=0):
        super().__init__()

        self.q = nn.Linear(embed_dim, qk_dim)
        self.k = nn.Linear(embed_dim, qk_dim)
        self.v = nn.Linear(embed_dim, v_dim)

        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Linear(v_dim, embed_dim)

        self.apply(self._init_weights)

    def forward(self, x):
        # (B, Seq_len, qk_dim)
        Q = self.q(x)
        # (B, Seq_len, qk_dim)
        K = self.k(x)

        print(f"Q 矩阵：")
        print(Q)
        # (B, Seq_len, v_dim)
        V = self.v(x)

        print(f"V 矩阵：")
        print(V)


        # .mT 矩阵转置
        # (B, Seq_len, Seq_len)
        raw_scores = (Q @ K.mT) / math.sqrt(Q.shape[-1])

        # 因果掩码
        raw_scores = causal_mask(raw_scores)

        print(f"掩码矩阵：")

        # (B, seq_len, seq_len)
        scores = self.dropout(F.softmax(raw_scores, dim=-1))

        print(print(f"分数矩阵："))

        # (B, seq_len, embed_dim)
        return self.proj(scores @ V)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.ones_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)