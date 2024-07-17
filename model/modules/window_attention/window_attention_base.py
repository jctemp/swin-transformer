from __future__ import annotations
from abc import ABC, abstractmethod
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from einops import einsum, rearrange, repeat


class WindowMultiHeadAttention(ABC, nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        window_size: List[int],
        qkv_bias: bool = True,
        drop_attn: float = 0.1,
        drop_proj: float = 0.1,
        rpe: bool = True,
        shift: bool = False,
    ) -> None:
        super().__init__()

        if in_channels % num_heads != 0:
            raise ValueError(
                "embed_dim % num_heads should be zero. It was"
                f" {in_channels % num_heads}"
            )

        self.out_channels = in_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.use_rpe = rpe
        self.shift = shift

        self.dims = len(window_size)
        self.shift_size = tuple([w // 2 for w in window_size])
        self.inv_sqrt_dim = 1.0 / math.sqrt(in_channels // num_heads)
        self.rel_pos_bias_table = self.init_rpe()

        self.proj_query = nn.Linear(in_channels, in_channels, bias=qkv_bias)
        self.proj_key = nn.Linear(in_channels, in_channels, bias=qkv_bias)
        self.proj_value = nn.Linear(in_channels, in_channels, bias=qkv_bias)
        self.proj = nn.Linear(in_channels, in_channels)

        self.drop_attn = nn.Dropout(drop_attn)
        self.drop_proj = nn.Dropout(drop_proj)

        self.softmax = nn.Softmax(dim=-1)

    @abstractmethod
    def init_rpe(self) -> torch.Tensor:
        pass

    @abstractmethod
    def rpe(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def to_seq(self, x: torch.Tensor, spatial_dims: List[int]) -> torch.Tensor:
        pass

    @abstractmethod
    def to_spatial(self, x: torch.Tensor, spatial_dims: List[int]) -> torch.Tensor:
        pass

    def cycle_shift(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        return torch.roll(
            x,
            [s * (1 if reverse else -1) for s in self.shift_size],
            dims=tuple(range(-self.dims, 0)),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,  # (heads, [D], H, W) | 0: attend, -float("inf"): don't attend
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, *spatial_dims = query.shape
        
        if self.shift:
            query = self.cycle_shift(query)
            key = self.cycle_shift(key)
            value = self.cycle_shift(value)

        query = self.to_seq(query, spatial_dims)
        key = self.to_seq(key, spatial_dims)
        value = self.to_seq(value, spatial_dims)

        _, seq_len, _ = query.shape
        if seq_len != self.seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} doesn't match window size {self.seq_len}"
            )

        # Project to local attention dimension
        query = self.proj_query(query)
        key = self.proj_key(key)
        value = self.proj_value(value)

        # Reorder to easily process each attention head
        query = rearrange(query, "b n (h d) -> b h n d", h=self.num_heads)
        key = rearrange(key, "b n (h d) -> b h n d", h=self.num_heads)
        value = rearrange(value, "b n (h d) -> b h n d", h=self.num_heads)

        # Compute attention score (not normalised yet)
        score = einsum(query, key, "b h i d, b h j d -> b h i j") * self.inv_sqrt_dim

        # Add relative positional embedding like in original SwinTransformer
        # (does not follow Shaw et al. relative positional embedding paradigm)
        if self.use_rpe:
            score = self.rpe(score)

        # Mask unknown scores (relevant for decoders)
        if mask is not None:
            score = score.masked_fill(mask == -float("inf"), -1e9)

        # Compute attention weights
        context = self.softmax(score)
        context = self.drop_attn(context)

        # Attend values
        out = einsum(context, value, "b h i j, b h j d -> b h i d")
        out = rearrange(out, "b h n d -> b n (h d)")

        # Project to original dimension
        x = self.proj(out)
        x = self.drop_proj(x)

        x = self.to_spatial(x, spatial_dims)

        if self.shift:
            x = self.cycle_shift(x, reverse=True)

        return x, context
