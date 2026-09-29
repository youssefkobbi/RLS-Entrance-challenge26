"""Full vision-language model.

Wires together the CNN encoder, the adapter (flatten + linear projection
to visual tokens), the Transformer decoder, and the linear head that
produces next-letter logits.

The provided tests expect the last nn.Module defined in this file to be
constructible with no arguments and called as model(images, input_ids), with
input_ids of shape (B, T), returning logits of shape (B, T_out, 27) where
T_out >= T and the last T logits line up with input_ids.

Shapes (defaults: d_model=128, 64 visual tokens, T letters):

    images                 (B, 3, 64, 64)
    encoder                (B, 128, 8, 8)          feature map
    flatten                (B, 64, 128)            H'*W' = 64 positions, C = 128 channels
    adapter (Linear)       (B, 64, d_model)        visual tokens
    letter embedding       (B, T, d_model)
    concatenate            (B, 64 + T, d_model)    ONE sequence: [v1 ... v64  l1 ... lT]
    decoder (masked attn)  (B, 64 + T, d_model)
    keep last T + 1 rows   (B, T + 1, d_model)     row 0 = last visual token, rows 1..T = letters
    head (Linear)          (B, T + 1, 27)          logits

Row 0 (the last visual token) predicts letter 1; row i (letter i) predicts letter i + 1,
and the row of the last letter predicts <eos>.  So T_out = T + 1 and the last T logits
line up with input_ids.  The targets are therefore [l1 ... lL <eos>] for L input letters.

Attention mask (prefix-LM): True = "may attend"
                 keys: visual (N)      letters (T)
    queries visual    all True (or causal, option)    all False
    queries letters   all True                        lower-triangular (causal)
"""

import torch
from torch import nn

from src.model.decoder import TransformerDecoder
from src.model.encoder import CNNEncoder
from src.tokenizer import MAX_LETTERS, VOCAB_SIZE


def build_prefix_mask(n_visual, n_letters, visual_attention="bidirectional", device=None):
    """(S, S) bool mask, S = n_visual + n_letters.  True = query row may attend to key column."""
    size = n_visual + n_letters
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))  # causal everywhere
    if visual_attention == "bidirectional":
        mask[:n_visual, :n_visual] = True  # the image is given all at once: no "future" to hide
    elif visual_attention != "causal":
        raise ValueError(f"visual_attention must be 'bidirectional' or 'causal', got {visual_attention!r}")
    return mask


class Adapter(nn.Module):
    """Feature map (B, C, H', W') -> visual tokens (B, H'*W', d_model)."""

    def __init__(self, in_channels, d_model):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)

    def forward(self, feature_map):
        tokens = feature_map.flatten(2).transpose(1, 2)  # (B, C, H'*W') -> (B, H'*W', C)
        return self.proj(tokens)  # (B, H'*W', d_model)


class TinyVLM(nn.Module):
    def __init__(
        self,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        dropout=0.1,
        encoder_channels=(32, 64, 128),
        image_size=64,
        max_letters=MAX_LETTERS,
        vocab_size=VOCAB_SIZE,
        visual_attention="bidirectional",
    ):
        super().__init__()
        self.visual_attention = visual_attention
        self.encoder = CNNEncoder(channels=tuple(encoder_channels), image_size=image_size)
        self.n_visual = self.encoder.n_tokens  # N = H' * W' = 64
        self.adapter = Adapter(self.encoder.out_channels, d_model)
        self.tok_emb = nn.Embedding(vocab_size, d_model)  # letter embeddings
        self.decoder = TransformerDecoder(
            d_model, n_heads, n_layers, d_ff, max_len=self.n_visual + max_letters, dropout=dropout
        )
        self.head = nn.Linear(d_model, vocab_size)

    def encode_image(self, images):
        """(B, 3, 64, 64) -> visual tokens (B, N, d_model).  Computed once per image at generation time."""
        return self.adapter(self.encoder(images))

    def decode(self, visual_tokens, input_ids):
        """visual tokens (B, N, d_model) + letters (B, T) -> logits (B, T + 1, vocab)."""
        n, t = visual_tokens.shape[1], input_ids.shape[1]
        letters = self.tok_emb(input_ids)  # (B, T, d_model)
        x = torch.cat([visual_tokens, letters], dim=1)  # (B, N + T, d_model)
        mask = build_prefix_mask(n, t, self.visual_attention, device=x.device)  # (N + T, N + T)
        h = self.decoder(x, mask)  # (B, N + T, d_model)
        return self.head(h[:, n - 1 :])  # keep last visual token + letters: (B, T + 1, vocab)

    def forward(self, images, input_ids):
        return self.decode(self.encode_image(images), input_ids)
