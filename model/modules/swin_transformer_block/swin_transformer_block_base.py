from __future__ import annotations
from abc import abstractmethod
from typing import List, Optional, Tuple, Type

from einops import repeat
import torch
import torch.nn as nn
from timm.layers import DropPath


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        drop: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        window_size: List[int],
        mlp_ratio: float = 2.0,
        drop: float = 0.1,
        drop_path: float = 0.1,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.norm_attn = norm_layer(in_channels)
        mlp_features = int(in_channels * mlp_ratio)
        self.mlp = MLP(in_channels, mlp_features, in_channels, drop=drop, act_layer=act_layer)
        self.norm_mlp = norm_layer(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.dims = len(window_size)
        self.window_size = window_size
        self.shift_size = tuple([w // 2 for w in window_size])
        self.attn_map: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.attn is not None

        res = x.clone()
        if self.attn_mask is None:
            x, attn = self.attn(x, x, x, None)
        else:
            x, attn = self.attn(x, x, x, repeat(self.attn_mask, "b ... -> (b n) ...", n=x.shape[0]))
        x = self.norm_attn(x.transpose(1, -1)).transpose(1, -1)
        x = self.drop_path(x) + res

        # for lookup later
        self.attn_map = attn

        res = x.clone()
        x = self.mlp(x.transpose(1, -1))
        x = self.norm_mlp(x).transpose(1, -1)
        x = self.drop_path(x) + res

        return x

    @abstractmethod
    def create_attn_mask(self, input_resolution: Tuple) -> torch.Tensor:
        pass
