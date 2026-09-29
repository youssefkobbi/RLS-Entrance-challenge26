"""CNN vision encoder.

Built from nn.Conv2d, activations, normalization and pooling (no
pretrained backbones). Turns a (B, 3, 64, 64) image into a feature map
(B, C, H', W').

The provided tests expect the last nn.Module defined in this file to be
constructible with no arguments and called as encoder(images).

Output-size formula for a conv / pool layer (per spatial dimension):
    out = floor((in + 2 * padding - kernel) / stride) + 1
* Conv2d(kernel=3, stride=1, padding=1) keeps the size:  (64 + 2 - 3) / 1 + 1 = 64
* MaxPool2d(kernel=2, stride=2) halves it:               (64 - 2) / 2 + 1     = 32
With three stages the size goes 64 -> 32 -> 16 -> 8, so the feature map is 8 x 8
= 64 positions, each becoming one visual token.
"""

import torch
from torch import nn


class ConvStage(nn.Module):
    """[Conv3x3 -> BatchNorm -> ReLU] x 2, then 2x2 max-pool (halves H and W)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),  # bias is redundant before BN
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):  # (B, C_in, H, W) -> (B, C_out, H / 2, W / 2)
        return self.block(x)


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, channels=(32, 64, 128), image_size=64):
        super().__init__()
        stages, prev = [], in_channels
        for c in channels:
            stages.append(ConvStage(prev, c))
            prev = c
        self.stages = nn.Sequential(*stages)
        self.out_channels = prev  # C
        self.out_size = image_size // 2 ** len(channels)  # H' = W' (each stage halves the size)
        self.n_tokens = self.out_size**2  # H' * W' = number of visual tokens

    def forward(self, images):  # (B, 3, 64, 64) -> (B, C, H', W') = (B, 128, 8, 8)
        return self.stages(images)
