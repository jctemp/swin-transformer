import math
from typing import List, Optional

import torch
import torch.nn as nn
from einops import einsum, rearrange, repeat


class ParameterisedRelativePositionalEmbedding2d(nn.Module):
    def __init__(self, window_size: List[int], num_heads: int, std: float = 0.02):
        super(ParameterisedRelativePositionalEmbedding2d, self).__init__()

        if len(window_size) != 2:
            raise ValueError(
                f"window_size must have exactly 2 elements, got {len(window_size)}"
            )

        self.window_size = window_size

        # Create the embeddings for each token in a window
        self.rel_pos_bias_table = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros(
                    (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads
                ),
                std=std,
            )
        )

        # Compute absolute position (dimensions, ws[0], ws[1])
        abs_coords = torch.stack(
            torch.meshgrid([torch.arange(ws) for ws in window_size], indexing="ij")
        )
        abs_coords_flat = rearrange(abs_coords, "c ... -> c (...)")

        # Compute relative coordinates
        rel_coords = rearrange(abs_coords_flat, "c i -> c i 1") - rearrange(
            abs_coords_flat, "c j -> c 1 j"
        )
        rel_coords = rearrange(rel_coords, "c i j -> i j c")

        # Shift coordinates to start from 0
        rel_coords += (
            repeat(torch.tensor(window_size, dtype=torch.int32), "c -> 1 1 c") - 1
        )

        # Scale the height dimension
        rel_coords[..., 1] *= 2 * window_size[1] - 1

        # Unique indices
        rel_pos_idx = rel_coords.sum(-1)

        # The index values are not trainable parameteres, but are required in
        # state dict. Therefore, we register the tensor.
        self.register_buffer("rel_pos_idx", rel_pos_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rel_pos_bias = rearrange(
            self.rel_pos_bias_table[self.rel_pos_idx.view(-1)],
            "(a b) h -> 1 h a b",
            a=self.window_size[0] * self.window_size[1],
        )
        return x + rel_pos_bias

    def extra_repr(self) -> str:
        return f"window_size={self.window_size}"


class ParameterisedRelativePositionalEmbedding3d(nn.Module):
    def __init__(self, window_size: List[int], num_heads: int, std: float = 0.02):
        super(ParameterisedRelativePositionalEmbedding3d, self).__init__()

        if len(window_size) != 3:
            raise ValueError(
                f"window_size must have exactly 3 elements, got {len(window_size)}"
            )

        self.window_size = window_size

        # Create the embeddings for each token in a window
        self.rel_pos_bias_table = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros(
                    (2 * window_size[0] - 1)
                    * (2 * window_size[1] - 1)
                    * (2 * window_size[2] - 1),
                    num_heads,
                ),
                std=std,
            )
        )

        # Compute absolute position (dimensions, ws[0], ws[1])
        abs_coords = torch.stack(
            torch.meshgrid([torch.arange(ws) for ws in window_size], indexing="ij")
        )
        abs_coords_flat = rearrange(abs_coords, "c ... -> c (...)")

        # Compute relative coordinates
        rel_coords = rearrange(abs_coords_flat, "c i -> c i 1") - rearrange(
            abs_coords_flat, "c j -> c 1 j"
        )
        rel_coords = rearrange(rel_coords, "c i j -> i j c")

        # Shift coordinates to start from 0
        rel_coords += (
            repeat(torch.tensor(window_size, dtype=torch.int32), "c -> 1 1 c") - 1
        )

        # Scale the height dimension
        rel_coords[..., 2] *= (2 * window_size[2] - 1) * (2 * window_size[1] - 1)
        rel_coords[..., 1] *= 2 * window_size[1] - 1

        # Unique indices
        rel_pos_idx = rel_coords.sum(-1)

        # The index values are not trainable parameteres, but are required in
        # state dict. Therefore, we register the tensor.
        self.register_buffer("rel_pos_idx", rel_pos_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rel_pos_bias = rearrange(
            self.rel_pos_bias_table[self.rel_pos_idx.view(-1)],
            "(a b) h -> 1 h a b",
            a=self.window_size[0] * self.window_size[1] * self.window_size[2],
        )
        return x + rel_pos_bias

    def extra_repr(self) -> str:
        return f"window_size={self.window_size}"


class WindowMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: List[int],
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rpe: Optional[ParameterisedRelativePositionalEmbedding2d] = None,
    ):
        super(WindowMultiHeadAttention, self).__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim % num_heads should be zero. It was"
                f" {embed_dim % num_heads}"
            )

        self.dims = len(window_size)
        self.inv_sqrt_dim = 1.0 / math.sqrt(embed_dim)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.rpe = rpe(window_size)

        self.proj_query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.proj_key = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.proj_value = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    @property
    def out_channels(self):
        return self.embed_dim

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        _, seq_len, _ = query.shape  # b (Ww * Wh ...) d

        if seq_len != self.window_size[0] * self.window_size[1]:
            raise ValueError(
                f"Input sequence length {seq_len} doesn't match window size {self.window_size}"
            )

        query = self.proj_query(query)  # b n h d
        key = self.proj_key(key)
        value = self.proj_value(value)

        query = rearrange(query, "b n (h d) -> b h n d", h=self.num_heads)
        key = rearrange(key, "b n (h d) -> b h n d", h=self.num_heads)
        value = rearrange(value, "b n (h d) -> b h n d", h=self.num_heads)

        score = einsum(query, key, "b h i d, b h j d -> b h i j") * self.inv_sqrt_dim

        # adds relative positional embedding (does not follow the description of Shaw et al.)
        if self.rpe is not None:
            score = self.rpe(score)

        if mask is not None:
            assert mask.shape == score.shape[2:]
            assert (
                mask.shape == score.shape[-2:]
            ), f"Mask shape {mask.shape} doesn't match attention shape {score.shape[-2:]}"
            score = score.masked_fill(mask == 0, float("-inf"))

        context = self.softmax(score)
        context = self.attn_drop(context)

        out = einsum(context, value, "b h i j, b h j d -> b h i d")
        out = rearrange(out, "b h n d -> b n (h d)")

        x = self.proj(out)
        x = self.proj_drop(x)

        return x
