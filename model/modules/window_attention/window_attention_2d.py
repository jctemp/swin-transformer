from typing import List

import torch
import torch.nn as nn
from einops import rearrange, repeat

from . import WindowMultiHeadAttention


class WindowMultiHeadAttention2D(WindowMultiHeadAttention):
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
        super().__init__(
            in_channels,
            num_heads,
            window_size,
            qkv_bias,
            drop_attn,
            drop_proj,
            rpe,
        )
        self.seq_len = window_size[0] * window_size[1]

    def init_rpe(self) -> torch.Tensor:
        if len(self.window_size) != 2:
            raise ValueError(
                f"window_size must have exactly 2 elements, got {len(self.window_size)}"
            )

        # Create the embeddings for each token in a window
        rel_pos_bias_table = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                    self.num_heads,
                ),
            )
        )

        # Compute absolute position (dimensions, ws[0], ws[1])
        abs_coords = torch.stack(
            torch.meshgrid([torch.arange(ws) for ws in self.window_size], indexing="ij")
        )
        abs_coords_flat = rearrange(abs_coords, "c ... -> c (...)")

        # Compute relative coordinates
        rel_coords = rearrange(abs_coords_flat, "c i -> c i 1") - rearrange(
            abs_coords_flat, "c j -> c 1 j"
        )
        rel_coords = rearrange(rel_coords, "c i j -> i j c")

        # Shift coordinates to start from 0
        rel_coords += (
            repeat(torch.tensor(self.window_size, dtype=torch.int32), "c -> 1 1 c") - 1
        )

        # Scale the height dimension
        rel_coords[..., 1] *= 2 * self.window_size[1] - 1

        # Unique indices
        rel_pos_idx = rel_coords.sum(-1)

        # The index values are not trainable parameteres, but are required in
        # state dict. Therefore, we register the tensor.
        self.register_buffer("rel_pos_idx", rel_pos_idx)

        return rel_pos_bias_table

    def rpe(self, x: torch.Tensor) -> torch.Tensor:
        rel_pos_bias = rearrange(
            self.rel_pos_bias_table[self.rel_pos_idx.view(-1)],
            "(a b) h -> 1 h a b",
            a=self.window_size[0] * self.window_size[1],
        )
        return x + rel_pos_bias

    def to_seq(self, x: torch.Tensor, spatial_dims: List[int]) -> torch.Tensor:
        # Transform input (B C W H) to ("B W/M_w H/M_h" "M_w M_h" C)
        wm = spatial_dims[0] // self.window_size[0]
        hm = spatial_dims[1] // self.window_size[1]
        x = rearrange(x, "b c (wm w) (hm h) -> (b wm hm) (w h) c", wm=wm, hm=hm)
        return x

    def to_spatial(self, x: torch.Tensor, spatial_dims: List[int]) -> torch.Tensor:
        # Transform ("B W/M_w H/M_h" "M_w M_h" C) to (B C W H)
        wm = spatial_dims[0] // self.window_size[0]
        hm = spatial_dims[1] // self.window_size[1]
        x = rearrange(
            x,
            "(b wm hm) (w h) c -> b c (wm w) (hm h)",
            wm=wm,
            hm=hm,
            w=self.window_size[0],
            h=self.window_size[1],
        )
        return x