from typing import List, Tuple, Type
from einops import rearrange, repeat
import torch
import torch.nn as nn

from ..modules import *


class SimpleCrop3D(nn.Module):
    def __init__(self, crop_range: Tuple[int, int, int, int, int, int]):
        super().__init__()
        self.crop_range = crop_range  # (left, right, top, bottom, front, back)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[
            ...,
            self.crop_range[2] : self.crop_range[3],
            self.crop_range[0] : self.crop_range[1],
            self.crop_range[4] : self.crop_range[5],
        ]


class SwinTransformer3D(nn.Module):
    def __init__(
        self,
        input_resolution: Tuple[int, int, int],
        in_channels: int,
        out_channels: List[int] = [96 * 2**i for i in range(0, 4)],
        depths: List[int] = [2, 2, 18, 2],
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        merge_size: List[Tuple[int, int, int]] = [(2, 2, 2)] * 3,
        window_size: List[Tuple[int, int, int]] = [(7, 7, 7)] * 4,
        num_heads: List[int] = [4 * 2**i for i in range(0, 4)],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.1,
        drop_attn: float = 0.1,
        drop_path: float = 0.1,
        ape: bool = False,
        rpe: bool = True,
        patch_norm: bool = True,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        padding_layer: Type[nn.Module] = nn.ZeroPad3d,
    ):
        super().__init__()

        assert (
            len(out_channels)
            == len(depths)
            == len(num_heads)
            == len(window_size)
            == len(merge_size) + 1
        )

        self.stages = nn.ModuleList()
        in_res = input_resolution
        for i, depth in enumerate(depths):
            if i == 0:
                operation = PatchEmbedding3D(
                    in_channels=in_channels,
                    out_channels=out_channels[i],
                    patch_size=patch_size,
                    norm_layer=norm_layer if patch_norm else None,
                    ape=ape,
                )
                in_res = (
                    in_res[0] // patch_size[0],
                    in_res[1] // patch_size[1],
                    in_res[2] // patch_size[2],
                )
            else:
                operation = PatchMerging3D(
                    in_channels=out_channels[i - 1],
                    out_channels=out_channels[i],
                    merge_size=merge_size[i - 1],
                    norm_layer=norm_layer if patch_norm else None,
                )
                in_res = (
                    in_res[0] // merge_size[i - 1][0],
                    in_res[1] // merge_size[i - 1][1],
                    in_res[2] // merge_size[i - 1][2],
                )
            padding = (
                (
                    0
                    if in_res[0] % window_size[i][0] == 0
                    else window_size[i][0] - (in_res[0] % window_size[i][0])
                ),
                (
                    0
                    if in_res[1] % window_size[i][1] == 0
                    else window_size[i][1] - (in_res[1] % window_size[i][1])
                ),
                (
                    0
                    if in_res[2] % window_size[i][2] == 0
                    else window_size[i][2] - (in_res[2] % window_size[i][2])
                ),
            )
            in_res_w_pad = (
                in_res[0] + padding[0],
                in_res[1] + padding[1],
                in_res[2] + padding[2],
            )

            block_attn_mask = None
            if padding != (0, 0, 0):
                block_mask = torch.zeros(in_res_w_pad)
                for w in range(1, padding[0] + 1):
                    for h in range(0, in_res_w_pad[1]):
                        for d in range(0, in_res_w_pad[2]):
                            block_mask[-w, h, d] = float(-100.0)
                for h in range(1, padding[1] + 1):
                    for w in range(0, in_res_w_pad[0]):
                        for d in range(0, in_res_w_pad[2]):
                            block_mask[w, -h, d] = float(-100.0)
                for d in range(1, padding[2] + 1):
                    for w in range(0, in_res_w_pad[0]):
                        for h in range(0, in_res_w_pad[1]):
                            block_mask[w, h, -d] = float(-100.0)
                block_mask = rearrange(
                    block_mask,
                    "(w p1) (h p2) (d p3) -> (w h d) (p1 p2 p3)",
                    p1=window_size[i][0],
                    p2=window_size[i][1],
                    p3=window_size[i][2],
                )
                block_attn_mask = block_mask.unsqueeze(1) - block_mask.unsqueeze(2)
                block_attn_mask = repeat(
                    block_attn_mask, "b nw1 nw2 -> b h nw1 nw2", h=num_heads[i]
                )
                block_attn_mask = block_attn_mask.masked_fill(
                    block_attn_mask != 0, float(-100.0)
                ).masked_fill(block_attn_mask == 0, float(0.0))

            blocks: List[SwinTransformerBlock3D] = []
            for k in range(depth):

                pad = padding_layer((0, padding[0], 0, padding[1], 0, padding[2]))
                crop = SimpleCrop3D(
                    (
                        0,
                        in_res_w_pad[0] - padding[0],
                        0,
                        in_res_w_pad[1] - padding[1],
                        0,
                        in_res_w_pad[2] - padding[2],
                    )
                )
                block = SwinTransformerBlock3D(
                    input_resolution=in_res_w_pad,
                    in_channels=out_channels[i],
                    num_heads=num_heads[i],
                    window_size=list(window_size[i]),
                    qkv_bias=qkv_bias,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_attn=drop_attn,
                    drop_path=drop_path,
                    rpe=rpe,
                    shift=k % 2 == 0,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    attn_mask=block_attn_mask,
                )
                blocks.append(block)

            self.stages.append(nn.Sequential(operation, pad, *blocks, crop))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []

        for stage in self.stages:
            x = stage(x)
            outputs.append(x)

        return outputs
