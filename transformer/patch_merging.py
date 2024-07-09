from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class PatchMerging(nn.Module):
    def __init__(
        self,
        in_channels: int,
        merge_size: Union[Tuple[int, int], Tuple[int, int, int]] = (2, 2),
        bias: bool = True,
        norm_layer: Optional[nn.Module] = nn.LayerNorm,
    ):
        super(PatchMerging, self).__init__()

        self.dims = len(merge_size)

        if self.dims not in [2, 3]:
            raise ValueError(f"len(merge_size) = {self.dims}, but expected [2,3]")

        self.in_channels = in_channels
        self.merge_size = merge_size
        self.bias = bias
        self.embed_dim = (merge_size[0] * merge_size[1]) // 2 * in_channels
        self.norm = None if norm_layer is None else norm_layer(self.in_channels)

        module = getattr(nn, f"Conv{self.dims}d")
        self.transform = module(
            in_channels=in_channels,
            out_channels=self.embed_dim,
            kernel_size=merge_size,
            stride=merge_size,
            padding="valid",
            bias=bias,
        )

    @property
    def embed_dim(self) -> int:
        return self.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, *spatial_dims = x.shape

        if len(spatial_dims) != self.dims:
            raise ValueError(
                f"Input spatial dimensions {len(spatial_dims)} don't match merge size dimensions {self.dims}"
            )

        for d, m in zip(spatial_dims, self.merge_size):
            if d % m != 0:
                raise ValueError(
                    f"Input dimension {d} is not divisible by merge size {m}"
                )

        x = self.transform(x)

        if self.norm:
            x = self.norm(x.transpose(-1, 1)).transpose(-1, 1)

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, merge_size={self.merge_size})"
