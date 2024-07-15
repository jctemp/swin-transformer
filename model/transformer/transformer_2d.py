from typing import List, Tuple, Type
import torch
import torch.nn as nn

from ..modules import *


class SimpleCrop2D(nn.Module):
    def __init__(self, crop_range: Tuple[int, int, int, int]):
        super().__init__()
        self.crop_range = crop_range  # (left, right, top, bottom)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[
            ...,
            self.crop_range[2] : self.crop_range[3],
            self.crop_range[0] : self.crop_range[1],
        ]


class SwinTransformer2D(nn.Module):
    def __init__(
        self,
        input_resolution: Tuple[int, int],
        in_channels: int,
        out_channels: List[int] = [96 * 2**i for i in range(0, 4)],
        depths: List[int] = [2, 2, 18, 2],
        patch_size: Tuple[int, int] = (4, 4),
        merge_size: List[Tuple[int, int]] = [(2, 2)] * 3,
        window_size: List[Tuple[int, int]] = [(7, 7)] * 4,
        num_heads: List[int] = [4 * 2**i for i in range(0, 4)],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.1,
        drop_attn: float = 0.1,
        drop_path: float = 0.1,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        ape: bool = False,
        rpe: bool = True,
        patch_norm: bool = True,
        padding_layer: Type[nn.Module] = nn.ZeroPad2d,
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
                operation = PatchEmbedding2D(
                    in_channels=in_channels,
                    out_channels=out_channels[i],
                    patch_size=patch_size,
                    norm_layer=norm_layer if patch_norm else None,
                    ape=ape,
                )
                in_res = (
                    in_res[0] // patch_size[0],
                    in_res[1] // patch_size[1],
                )
            else:
                operation = PatchMerging2D(
                    in_channels=out_channels[i - 1],
                    out_channels=out_channels[i],
                    merge_size=merge_size[i - 1],
                    norm_layer=norm_layer if patch_norm else None,
                )
                in_res = (
                    in_res[0] // merge_size[i - 1][0],
                    in_res[1] // merge_size[i - 1][1],
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
            )
            in_res_w_pad = (
                in_res[0] + padding[0],
                in_res[1] + padding[1],
            )

            blocks: List[SwinTransformerBlock2D] = []
            for k in range(depth):

                pad = padding_layer((0, padding[0], 0, padding[1]))
                crop = SimpleCrop2D(
                    (0, in_res_w_pad[0] - padding[0], 0, in_res_w_pad[1] - padding[1])
                )
                block = SwinTransformerBlock2D(
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
                )
                blocks.append(block)

            self.stages.append(nn.Sequential(operation, pad, *blocks, crop))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []

        for stage in self.stages:
            x = stage(x)
            outputs.append(x)

        return outputs
