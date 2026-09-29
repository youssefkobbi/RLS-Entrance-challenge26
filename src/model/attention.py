"""Handwritten multi-head self-attention.

Q/K/V projections, split heads, scaled dot-product attention, mask,
softmax, merge heads, output projection -- written with tensor
operations, not nn.MultiheadAttention. See tests/test_attention.py for
the required equivalence test against F.scaled_dot_product_attention.

The provided tests expect the last nn.Module defined in this file to be
Cls(d_model, n_heads), called as attention(x, mask) with x of shape
(B, T, d_model). `mask` is optional; when given it is a boolean tensor where
True means "this query may attend to this key".
"""

import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # d_k: size of each head
        self.qkv = nn.Linear(d_model, 3 * d_model)  # one matmul that produces Q, K and V
        self.out_proj = nn.Linear(d_model, d_model)  # mixes the heads back together
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """x: (B, T, d_model); mask: bool, broadcastable to (B, 1, T, T), True = may attend."""
        B, T, _ = x.shape

        # 1) project, then cut into Q, K, V
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # 3 x (B, T, d_model)

        # 2) split heads: (B, T, d_model) -> (B, H, T, d_head). Heads are contiguous channel slices.
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)

        # 3) scaled dot-product scores: how well does query i match key j?
        #    Dividing by sqrt(d_k) keeps the score variance ~1, so softmax does not saturate.
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # (B, H, T, T)

        # 4) mask: forbidden (query, key) pairs get -inf-like scores -> weight 0 after softmax
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        # 5) softmax over the KEYS (last dim): each query's weights sum to 1
        weights = torch.softmax(scores, dim=-1)  # (B, H, T, T)
        weights = self.attn_dropout(weights)

        # 6) weighted average of the values
        heads = weights @ v  # (B, H, T, d_head)

        # 7) merge heads: (B, H, T, d_head) -> (B, T, d_model), then output projection
        merged = heads.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(merged)  # (B, T, d_model)
