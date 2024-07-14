from __future__ import annotations
from typing import List, Tuple

from einops import rearrange, repeat
import itertools
import torch

from .swin_transformer_block_base import SwinTransformerBlock
from ..window_attention import WindowMultiHeadAttention2D


class SwinTransformerBlock2D(SwinTransformerBlock):
    def __init__(
        self,
        input_resolution: Tuple[int, int],
        in_channels: int,
        num_heads: int,
        window_size: List[int],
        qkv_bias: bool = True,
        mlp_ratio: float = 2.0,
        drop: float = 0.1,
        drop_attn: float = 0.1,
        drop_proj: float = 0.1,
        drop_path: float = 0.1,
        rpe: bool = True,
        shift: bool = False,
    ) -> None:
        super().__init__(in_channels, window_size, mlp_ratio, drop, drop_path, shift)
        self.register_buffer(
            "attn_mask",
            self.create_attn_mask(input_resolution, num_heads) if shift else None,
        )
        self.attn = WindowMultiHeadAttention2D(
            in_channels=in_channels,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            drop_attn=drop_attn,
            drop_proj=drop_proj,
            rpe=rpe,
        )

    def create_attn_mask(
        self, input_resolution: Tuple[int, int], num_heads: int
    ) -> torch.Tensor:
        W, H = input_resolution
        img_mask = torch.zeros((W, H))

        w_slices = (
            slice(0, -self.window_size[0]),
            slice(-self.window_size[0], -self.shift_size[0]),
            slice(-self.shift_size[0], None),
        )
        h_slices = (
            slice(0, -self.window_size[1]),
            slice(-self.window_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None),
        )

        for cnt, (w, h) in enumerate(itertools.product(w_slices, h_slices)):
            img_mask[w, h] = cnt

        attn_mask = rearrange(
            img_mask,
            "(w p1) (h p2) -> (w h) (p1 p2)",
            p1=self.window_size[0],
            p2=self.window_size[1],
        )

        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = repeat(attn_mask, "b nw1 nw2 -> b h nw1 nw2", h=num_heads)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        return attn_mask
