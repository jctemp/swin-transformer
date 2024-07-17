from __future__ import annotations
from typing import List, Optional, Tuple, Type

from einops import rearrange, repeat
import itertools
import torch
import torch.nn as nn

from .swin_transformer_block_base import SwinTransformerBlock
from ..window_attention import WindowMultiHeadAttention3D


class SwinTransformerBlock3D(SwinTransformerBlock):
    def __init__(
        self,
        input_resolution: Tuple[int, int, int],
        in_channels: int,
        num_heads: int,
        window_size: List[int],
        qkv_bias: bool = True,
        mlp_ratio: float = 2.0,
        drop: float = 0.1,
        drop_attn: float = 0.1,
        drop_path: float = 0.1,
        rpe: bool = True,
        shift: bool = False,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        attn_mask: Optional[torch.Tensor] = None, # 0 keep, -inf drop
    ) -> None:
        super().__init__(
            in_channels,
            window_size,
            mlp_ratio,
            drop,
            drop_path,
            norm_layer,
            act_layer,
        )
        self.register_buffer(
            "attn_mask",
            self.create_attn_mask(input_resolution, num_heads) if shift else None,
        )
        if shift and attn_mask is not None:
            self.attn_mask += attn_mask
        self.attn = WindowMultiHeadAttention3D(
            in_channels=in_channels,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            drop_attn=drop_attn,
            drop_proj=drop,
            rpe=rpe,
            shift=shift,
        )

    def create_attn_mask(
        self, input_resolution: Tuple[int, int, int], num_heads: int
    ) -> torch.Tensor:
        D, H, W = input_resolution
        img_mask = torch.zeros((D, H, W))

        d_slices = (
            slice(0, -self.window_size[0]),
            slice(-self.window_size[0], -self.shift_size[0]),
            slice(-self.shift_size[0], None),
        )
        h_slices = (
            slice(0, -self.window_size[1]),
            slice(-self.window_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None),
        )
        w_slices = (
            slice(0, -self.window_size[2]),
            slice(-self.window_size[2], -self.shift_size[2]),
            slice(-self.shift_size[2], None),
        )

        for cnt, (d, h, w) in enumerate(
            itertools.product(d_slices, h_slices, w_slices)
        ):
            img_mask[d, h, w] = cnt

        mask_windows = rearrange(
            img_mask,
            "(d p1) (h p2) (w p3) -> (d h w) (p1 p2 p3)",
            p1=self.window_size[0],
            p2=self.window_size[1],
            p3=self.window_size[2],
        )

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = repeat(attn_mask, "b nw1 nw2 -> b h nw1 nw2", h=num_heads)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -float("inf")).masked_fill(
            attn_mask == 0, float(0.0)
        )

        return attn_mask
