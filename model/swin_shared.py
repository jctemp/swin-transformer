import enum
from typing import Optional, Type

import torch
import torch.nn as nn


class PatchMode(enum.Enum):
    CONCATENATE = "concatenate"
    CONVOLUTION = "convolution"


class RelativePositionalEmeddingMode(enum.Enum):
    BIAS = "bias"
    CONTEXT = "context"
    NONE = "none"


class FeedForward(nn.Module):
    """FeedForward

    The feed-forward module for the Swin Transformer is a simple two-layer MLP with an activation function. It is used
    to process the output of the multi-head self-attention module.

    References:
        Swin Transformer: Hierarchical Vision Transformer using Shifted Windows - Li et al.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        act_layer (nn.Module): Activation layer type. Defaults to nn.GELU.
        norm_layer (Optional[Type[nn.Module]], optional): Normalisation layer type. Defaults to nn.LayerNorm.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        act_layer: nn.Module = nn.GELU(),
        norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.norm = norm_layer(out_channels) if norm_layer is not None else nn.Identity()

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (B, N, C).

        Returns:
            (torch.Tensor): Output tensor (B, N, C).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x
