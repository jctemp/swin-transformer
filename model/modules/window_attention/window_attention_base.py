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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,  # (heads, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, *spatial_dims = query.shape

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
            score_dims = len(score.shape)
            mask_dims = len(mask.shape)

            if mask_dims == score_dims:
                if (
                    score.shape[0] != mask.shape[0]
                    and score.shape[0] % mask.shape[0] == 0
                ):
                    batch_diff = score.shape[0] // mask.shape[0]
                    mask = repeat(mask, "b h nw1 nw2 -> (b d) h nw1 nw2", d=batch_diff)
            else:
                raise ValueError(
                    f"len(score.shape) ({len(score.shape)}) != len(mask.shape) ({len(mask.shape)})"
                )

            score = score + mask

        # Compute attention weights
        context = self.softmax(score)
        assert not torch.any(torch.isnan(context)), context
        context = self.drop_attn(context)

        # Attend values
        out = einsum(context, value, "b h i j, b h j d -> b h i d")
        out = rearrange(out, "b h n d -> b n (h d)")

        # Project to original dimension
        x = self.proj(out)
        x = self.drop_proj(x)

        x = self.to_spatial(x, spatial_dims)

        return x, context
