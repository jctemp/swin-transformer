from __future__ import annotations
from abc import abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from timm.layers import DropPath


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        window_size: List[int],
        mlp_ratio: float = 2.0,
        drop: float = 0.1,
        drop_path: float = 0.1,
        shift: bool = False,
    ) -> None:
        super().__init__()

        self.norm_attn = nn.LayerNorm(in_channels)
        mlp_features = int(in_channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_features),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_features, in_channels),
            nn.Dropout(drop),
        )
        self.norm_mlp = nn.LayerNorm(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.dims = len(window_size)
        self.shift = shift
        self.window_size = window_size
        self.shift_size = tuple([w // 2 for w in window_size])
        self.attn_map: Optional[torch.Tensor] = None

    def cycle_shift(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        return torch.roll(
            x,
            [s * -1 if reverse else 1 for s in self.shift_size],
            dims=tuple(range(2, 2 + self.dims)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert self.attn is not None

        if self.shift:
            x = self.cycle_shift(x)

        res = x.clone()
        x, attn = self.attn(x, x, x, self.attn_mask)
        x = self.norm_attn(x.transpose(1, -1)).transpose(1, -1)
        x = self.drop_path(x) + res

        # for lookup later
        self.attn_map = attn

        res = x.clone()
        x = self.mlp(x.transpose(1, -1))
        x = self.norm_mlp(x).transpose(1, -1)
        x = self.drop_path(x) + res

        if self.shift:
            x = self.cycle_shift(x, reverse=True)

        return x

    @abstractmethod
    def create_attn_mask(self, input_resolution: Tuple) -> torch.Tensor:
        pass
