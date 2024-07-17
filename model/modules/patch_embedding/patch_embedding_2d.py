from __future__ import annotations

import math
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
from einops import einsum, repeat


class PatchEmbedding2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 96,
        patch_size: Tuple[int, int] = (4, 4),
        norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm,
        ape: bool = False,
        ape_freq_base: float = 10000.0,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size

        self.use_ape = ape
        self.ape_freq_base = ape_freq_base
        self.pe: Optional[torch.Tensor] = None

        self.transform = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
            bias=False,
        )
        self.norm = None if norm_layer is None else norm_layer(self.out_channels)

    def ape(self, x: torch.Tensor) -> torch.Tensor:
        embed_dim, height, width = x.shape[-3:]

        if embed_dim % 4 != 0:
            raise ValueError(f"Embedding dimension must be divisible by 4 for 2D APE, got {embed_dim}")

        pe = torch.zeros(embed_dim, height, width, device=x.device)

        half_dim = embed_dim // 2
        denominator = torch.exp(
            torch.arange(0, half_dim, 2, device=x.device) * -(math.log(self.ape_freq_base) / half_dim)
        )

        pos_h = torch.arange(height, device=x.device)
        pos_w = torch.arange(width, device=x.device)
        pos_h = einsum(denominator, pos_h, "p, h -> p h")
        pos_w = einsum(denominator, pos_w, "p, w -> p w")

        a, b, c = 0, half_dim, 2 * half_dim
        pe[a + 0 : b : 2, :, :] = repeat(torch.sin(pos_h), "e h -> e h w", w=width)
        pe[a + 1 : b : 2, :, :] = repeat(torch.cos(pos_h), "e h -> e h w", w=width)
        pe[b + 0 : c : 2, :, :] = repeat(torch.sin(pos_w), "e w -> e h w", h=height)
        pe[b + 1 : c : 2, :, :] = repeat(torch.cos(pos_w), "e w -> e h w", h=height)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]

        assert H % self.patch_size[0] == 0, f"Input height {H} not divisible by merge size {self.patch_size[0]}"
        assert W % self.patch_size[1] == 0, f"Input width {W} not divisible by merge size {self.patch_size[1]}"

        x = self.transform(x)

        if self.use_ape and (self.pe is None or self.pe.shape[1:] != x.shape[1:]):
            self.pe = self.ape(x)

        if self.use_ape:
            x = x + self.pe

        if self.norm:
            x = self.norm(x.transpose(1, -1)).transpose(1, -1)

        return x

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, embed_dim={self.out_channels}, "
            f"patch_size={self.patch_size}, ape={self.use_ape})"
        )
