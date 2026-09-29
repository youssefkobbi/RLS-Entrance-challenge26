"""Transformer decoder block and stack.

Masked self-attention + MLP + residual connections + LayerNorm, stacked
N times, with positional embeddings over the combined [visual tokens] +
[letter tokens] sequence.

We use the "pre-LN" layout (LayerNorm BEFORE each sub-layer, as in GPT-2 and
Karpathy's nanoGPT), which trains more stably than the original post-LN:

    x = x + Attention(LN(x))
    x = x + MLP(LN(x))
"""

import torch
from torch import nn

from src.model.attention import MultiHeadSelfAttention


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):  # (B, S, d_model) -> (B, S, d_model)
        x = x + self.drop(self.attn(self.ln1(x), mask))  # residual around attention
        x = x + self.drop(self.mlp(self.ln2(x)))  # residual around the MLP
        return x


class TransformerDecoder(nn.Module):
    """Learned positional embeddings + N decoder blocks + final LayerNorm."""

    def __init__(self, d_model, n_heads, n_layers, d_ff, max_len, dropout=0.0):
        super().__init__()
        self.max_len = max_len
        self.pos_emb = nn.Embedding(max_len, d_model)  # one learned vector per absolute position
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """x: (B, S, d_model) = [visual tokens; letter embeddings]; mask: (S, S) bool."""
        S = x.shape[1]
        if S > self.max_len:
            raise ValueError(f"sequence length {S} exceeds max_len {self.max_len}")
        positions = torch.arange(S, device=x.device)  # (S,)
        x = self.drop(x + self.pos_emb(positions))  # (B, S, d_model) + (S, d_model) broadcast
        for block in self.blocks:
            x = block(x, mask)
        return self.ln_f(x)  # (B, S, d_model)
