from __future__ import annotations
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn


class PatchMerging2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        merge_size: Tuple[int, int] = (2, 2),
        norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if out_channels is None else out_channels
        self.merge_size = merge_size

        self.transform = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=merge_size,
            stride=merge_size,
            padding="valid",
            bias=False,
        )
        self.norm = None if norm_layer is None else norm_layer(self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape

        assert (
            H % self.merge_size[0] == 0
        ), f"Input height {H} not divisible by merge size {self.merge_size[0]}"
        assert (
            W % self.merge_size[1] == 0
        ), f"Input width {W} not divisible by merge size {self.merge_size[1]}"

        x = self.transform(x)

        if self.norm:
            x = self.norm(x.transpose(1, -1)).transpose(1, -1)

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, merge_size={self.merge_size})"