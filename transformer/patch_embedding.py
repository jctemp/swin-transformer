import math
from typing import Tuple, Union, Optional

import torch
import torch.nn as nn
from einops import repeat, einsum


class PatchEmbedding(nn.Module):

    def __init__(
        self,
        in_channels: int,
        patch_size: Union[Tuple[int, int], Tuple[int, int, int]] = (4, 4),
        embed_dim: int = 96,
        bias: bool = True,
        ape: bool = False,
        ape_freq_base: float = 10000.0,
        norm_layer: Optional[nn.Module] = nn.LayerNorm,
    ):
        super(PatchEmbedding, self).__init__()

        self.dims = len(patch_size)

        if self.dims not in [2, 3]:
            raise ValueError(f"len(patch_size) = {self.dims}, but expected [2,3]")

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.bias = bias
        self.ape = ape
        self.ape_freq_base = ape_freq_base
        self.pe: Optional[torch.Tensor] = None
        self.norm = None if norm_layer is None else norm_layer(self.in_channels)

        module = getattr(nn, f"Conv{self.dims}d")
        self.transform = module(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
            bias=bias,
        )

    @property
    def out_channels(self) -> int:
        return self.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, *spatial_dims = x.shape

        if len(spatial_dims) != len(self.patch_size):
            raise ValueError(
                f"Input spatial dimensions {len(spatial_dims)} don't match patch size dimensions {self.dims}"
            )

        for d, p in zip(spatial_dims, self.patch_size):
            if d % p != 0:
                raise ValueError(
                    f"Input dimension {d} is not divisible by patch size {p}"
                )

        x = self.transform(x)

        if self.ape:
            if self.pe is None or self.pe.shape[1:] != x.shape[1:]:
                self.pe = (
                    absolute_positional_encoding_2d
                    if self.dims == 2
                    else absolute_positional_encoding_3d
                )(x, freq_base=self.ape_freq_base)
            x = x + self.pe

        if self.norm:
            x = self.norm(x.transpose(-1, 1)).transpose(-1, 1)

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, embed_dim={self.embed_dim}, patch_size={self.patch_size}, ape={self.ape})"


# Attention Is All You Need - Vaswani et al.
def absolute_positional_encoding_2d(
    tensor: torch.Tensor,
    freq_base: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    embed_dim, width, height = tensor.shape[-3:]

    if embed_dim % 4 != 0:
        raise ValueError(
            f"Embedding dimension must be divisible by 4 for 2D APE, got {embed_dim}"
        )

    pe = torch.zeros(embed_dim, width, height, device=device)

    half_dim = embed_dim // 2
    denominator = torch.exp(
        torch.arange(0, half_dim, 2, device=device) * -(math.log(freq_base) / half_dim)
    )

    pos_w = torch.arange(width, device=device)
    pos_h = torch.arange(height, device=device)
    pos_w = einsum(denominator, pos_w, "d, w -> d w")
    pos_h = einsum(denominator, pos_h, "d, h -> d h")

    a, b, c = 0, half_dim, 2 * half_dim
    pe[a + 0 : b : 2, :, :] = repeat(torch.sin(pos_w), "d w -> d w h", h=height)
    pe[a + 1 : b : 2, :, :] = repeat(torch.cos(pos_w), "d w -> d w h", h=height)
    pe[b + 0 : c : 2, :, :] = repeat(torch.sin(pos_h), "d h -> d w h", w=width)
    pe[b + 1 : c : 2, :, :] = repeat(torch.cos(pos_h), "d h -> d w h", w=width)

    return pe


# Attention Is All You Need - Vaswani et al.
def absolute_positional_encoding_3d(
    tensor: torch.Tensor,
    freq_base: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    embed_dim, w, h, l = tensor.shape[-4:]

    if embed_dim % 6 != 0:
        raise ValueError(
            f"Embedding dimension must be divisible by 6 for 3D APE, got {embed_dim}"
        )

    pe = torch.zeros(embed_dim, w, h, l, device=device)

    third_dim = embed_dim // 3
    denom = torch.exp(
        torch.arange(0, third_dim, 2, device=device)
        * -(math.log(freq_base) / third_dim)
    )

    pos_w = torch.arange(w, device=device)
    pos_h = torch.arange(h, device=device)
    pos_l = torch.arange(l, device=device)
    pos_w = einsum(denom, pos_w, "d, w -> d w")
    pos_h = einsum(denom, pos_h, "d, h -> d h")
    pos_l = einsum(denom, pos_l, "d, l -> d l")

    a, b, c, d = 0, third_dim, 2 * third_dim, 3 * third_dim
    pe[a + 0 : b : 2, :, :, :] = repeat(torch.sin(pos_w), "d w -> d w h l", h=h, l=l)
    pe[a + 1 : b : 2, :, :, :] = repeat(torch.cos(pos_w), "d w -> d w h l", h=h, l=l)
    pe[b + 0 : c : 2, :, :, :] = repeat(torch.sin(pos_h), "d h -> d w h l", w=w, l=l)
    pe[b + 1 : c : 2, :, :, :] = repeat(torch.cos(pos_h), "d h -> d w h l", w=w, l=l)
    pe[c + 0 : d : 2, :, :, :] = repeat(torch.sin(pos_l), "d l -> d w h l", w=w, h=h)
    pe[c + 1 : d : 2, :, :, :] = repeat(torch.cos(pos_l), "d l -> d w h l", w=w, h=h)

    return pe
