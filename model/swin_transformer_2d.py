import enum
import itertools
import math
from typing import List, Optional, Tuple, Type

import einops
import einops.layers.torch as elt
import torch
import torch.nn as nn
from dataclasses import dataclass
from timm.layers import DropPath


class PatchMode(enum.Enum):
    CONCATENATE = "concatenate"
    CONVOLUTION = "convolution"


class PatchEmbedding2D(nn.Module):
    """PatchEmbedding2D

    For a vision transformer, we need to convert an image into a sequence of patches. We can achieve this by using a
    convolutional layer with a kernel size equal to the patch size and a stride equal to the patch size. This will
    produce a grid of patches which undergo a projection to the desired embedding dimension. The output is then reshaped
    to a sequence of patches (B [H * W] C).

    References:
        An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale - Dosovitskiy et al.

    Args:
        input_size (Tuple[int, int]): Size of the input image (height, width).
        patch_size (Tuple[int, int]): Size of the patch (height, width).
        in_channels (int): Number of input channels.
        embed_dim (int): Embedding dimension.
        norm_layer (Optional[Type[nn.Module]]): Normalisation layer type.
        mode (PatchMode): Patch embedding mode.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        in_channels: int,
        embed_dim: int,
        norm_layer: Optional[Type[nn.Module]] = None,
        mode: PatchMode = PatchMode.CONVOLUTION,
    ) -> None:
        super().__init__()
        assert (
            input_size[0] % patch_size[0] == 0 and input_size[1] % patch_size[1] == 0
        ), f"Input size {input_size} must be divisible by the patch size {patch_size}"

        # Output transformation
        self.rearrange_input = (
            elt.Rearrange("b c (hdm hm) (wdm wm) -> b (hdm wdm) (hm wm c)", hm=patch_size[0], wm=patch_size[1])
            if mode == PatchMode.CONCATENATE
            else nn.Identity()
        )
        self.rearrange_output = (
            elt.Rearrange("b c h w -> b (h w) c") if mode == PatchMode.CONVOLUTION else nn.Identity()
        )

        # Patch embedding
        self.proj = (
            nn.Linear(patch_size[0] * patch_size[1] * in_channels, embed_dim)
            if mode == PatchMode.CONCATENATE
            else nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

        # Auxilliary
        self.input_size = input_size
        self.output_size = (input_size[0] // patch_size[0], input_size[1] // patch_size[1])
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor (B, N, C).
        """
        x = self.rearrange_input(x)
        x = self.proj(x)
        x = self.rearrange_output(x)
        x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    """PatchMerging2D

    The Swin Transformer uses a hierarchical structure to process an image. Each stage the input image (B [H * W] C) is
    grouped into non-overlapping patches, again. These patches are then merged into a single patch by concatenating them
    along the channel dimension. The output is then reshaped to a sequence of patches (B [H * W] C).

    References:
        Swin Transformer: Hierarchical Vision Transformer using Shifted Windows - Li et al.

    Args:
        input_size (Tuple[int, int]): Size of the input image (height, width).
        merge_size (Tuple[int, int]): Size of the merged patch (height, width).
        embed_dim (int): Embedding dimension.
        norm_layer (Optional[Type[nn.Module]]): Normalisation layer type.
        mode (PatchMode): Patch embedding mode.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        merge_size: Tuple[int, int],
        embed_dim: int,
        norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm,
        mode: PatchMode = PatchMode.CONCATENATE,
    ) -> None:
        super().__init__()
        assert (
            input_size[0] % merge_size[0] == 0 and input_size[1] % merge_size[1] == 0
        ), f"Input size {input_size} must be divisible by the merge size {merge_size}"

        # Input/Output transformations
        self.rearrange_input = (
            elt.Rearrange("b (h w) c -> b c h w", h=input_size[0], w=input_size[1])
            if mode == PatchMode.CONVOLUTION
            else elt.Rearrange(
                "b (hdm hm wdm wm) c -> b (hdm wdm) (hm wm c)",
                hdm=input_size[0] // merge_size[0],
                wdm=input_size[1] // merge_size[1],
                hm=merge_size[0],
                wm=merge_size[1],
            )
        )
        self.rearrange_output = (
            elt.Rearrange("b c h w -> b (h w) c") if mode == PatchMode.CONVOLUTION else nn.Identity()
        )

        merge_dim = merge_size[0] * merge_size[1] * embed_dim

        # Projection and Normalisation
        self.proj = (
            nn.Conv2d(embed_dim, merge_dim // 2, merge_size, merge_size)
            if mode == PatchMode.CONVOLUTION
            else nn.Linear(merge_dim, merge_dim // 2)
        )
        self.norm = norm_layer(merge_dim // 2) if norm_layer is not None else nn.Identity()

        # Auxilliary
        self.input_size = input_size
        self.output_size = (input_size[0] // merge_size[0], input_size[1] // merge_size[1])
        self.merge_size = merge_size
        self.in_channels = embed_dim
        self.out_channels = merge_dim // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (B, N, C).

        Returns:
            torch.Tensor: Output tensor (B, N', C').
        """
        x = self.rearrange_input(x)
        x = self.proj(x)
        x = self.rearrange_output(x)
        x = self.norm(x)
        return x


class WindowShift2D(nn.Module):
    """WindowShift2D

    For efficient attention computation, the Swin Transformer employs a cyclic shift operation on the input sequence.
    This operation shifts the input sequence by the half of the window size in both the height and width dimensions.
    The shift operation is performed by rolling the input tensor in the height and width dimensions.

    References:
        Swin Transformer: Hierarchical Vision Transformer using Shifted Windows - Li et al. (see figure 4)

    Args:
        input_size (Tuple[int, int]): Size of the input image (height, width).
        shift_size (Tuple[int, int]): Size of the shift (height, width).
        reverse (bool): Whether to reverse the shift operation.
    """

    def __init__(self, input_size: Tuple[int, int], shift_size: Tuple[int, int], reverse: bool = False) -> None:
        super().__init__()
        # Input/Output transformations
        self.expand = elt.Rearrange("b (h w) c -> b h w c", h=input_size[0], w=input_size[1])
        self.squeeze = elt.Rearrange("b h w c -> b (h w) c")

        # Shift parameters
        self.input_size = input_size
        self.output_size = input_size
        self.reverse = reverse
        self.shift_size = shift_size if reverse else (-shift_size[0], -shift_size[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (B, N, C).

        Returns:
            torch.Tensor: Output tensor (B, N, C).
        """
        x = self.expand(x)
        x = torch.roll(x, shifts=self.shift_size, dims=(1, 2))
        x = self.squeeze(x)
        return x


class RelativePositionalEmeddingMode(enum.Enum):
    BIAS = "bias"
    CONTEXT = "context"
    NONE = "none"


class Attention2D(nn.Module):
    """Attention2D

    This is the multi-head self-attention module for the Swin Transformer. It works only on 2D data. We cannot use the
    standard multi-head self-attention module because of the cyclic shift operation and the positional embedding. Due to
    the cyclic shift operation, the attention module has to compute a mask for the non-adjacent items in the windows.

    References:
        Swin Transformer: Hierarchical Vision Transformer using Shifted Windows - Li et al.
        Self-Attention with Relative Position Representations - Shaw et al.
        Rethinking and Improving Relative Position Encoding for Vision Transformer - Wu et al.

    Args:
        input_size (Tuple[int, int]): Size of the input image (height, width).
        window_size (Tuple[int, int]): Size of the window (height, width).
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): Whether to add bias to the QKV linear layer.
        qk_scale (Optional[float]): Scaling factor for QK attention.
        drop_attn (float): Attention dropout rate.
        drop_proj (float): Projection dropout rate.
        rpe_mode (RelativePositionalEmbeddingMode): Whether to use relative position encoding.
        shift (bool): Whether to use shifted windows.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        window_size: Tuple[int, int],
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_attn: float = 0.0,
        drop_proj: float = 0.0,
        rpe_mode: RelativePositionalEmeddingMode = RelativePositionalEmeddingMode.BIAS,
        shift: bool = False,
    ) -> None:
        super().__init__()
        assert (
            input_size[0] % window_size[0] == 0 and input_size[1] % window_size[1] == 0
        ), f"Input size {input_size} must be divisible by the window size {window_size}"
        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension {embed_dim} must be divisible by the number of heads {num_heads}"

        # Input/output transformations
        self.to_sequence = elt.Rearrange("b nh n c -> b n (nh c)")
        self.to_windows = elt.Rearrange(
            "b (hdm hm wdm wm) c -> (b hdm wdm) (hm wm) c",
            hdm=input_size[0] // window_size[0],
            wdm=input_size[1] // window_size[1],
            hm=window_size[0],
            wm=window_size[1],
        )
        self.to_spatial = elt.Rearrange(
            "(b hdm wdm) (hm wm) c -> b (hdm hm wdm wm) c",
            hdm=input_size[0] // window_size[0],
            wdm=input_size[1] // window_size[1],
            hm=window_size[0],
            wm=window_size[1],
        )

        # Relative position encoding
        #
        #   No parameter sharing across heads as it is not context aware
        #   Clippling is disable, however, if needed one can parameterise the MHSA module with it
        #
        #   Rethinking and Improving Relative Position Encoding for Vision Transformer - Wu et al. (2021)
        #   see figure 1 and eq. 13, 14, 15, 16
        max_distance = (window_size[0] - 1, window_size[1] - 1)
        self.bias_mode = rpe_mode == RelativePositionalEmeddingMode.BIAS
        self.context_mode = rpe_mode == RelativePositionalEmeddingMode.CONTEXT

        self.embedding_table = nn.Embedding(sum(2 * d + 1 for d in max_distance), num_heads) if self.bias_mode else None
        self.embedding_table_q = (
            nn.Embedding(sum(2 * d + 1 for d in max_distance), embed_dim) if self.context_mode else None
        )
        self.embedding_table_k = (
            nn.Embedding(sum(2 * d + 1 for d in max_distance), embed_dim) if self.context_mode else None
        )
        self.register_buffer("indices", None)  # to cleanly move to device
        self.indices = self._create_indices(window_size, max_distance) if self.bias_mode or self.context_mode else None

        self.reshape_embedding = (
            elt.Rearrange(
                "h w nh -> nh h w" if self.bias_mode else "n m (nh c) -> n m nh c",
                nh=num_heads,
            )
            if self.bias_mode or self.context_mode
            else nn.Identity()
        )

        # Shifted windows
        #
        #   Efficient attention computation of shifted windows
        #   Masking non-adjascent items in windows for self-attention
        #
        #   Swin Transformer: Hierarchical Vision Transformer using Shifted Windows - Li et al. (2021)
        #   see figure 4
        self.shift = shift
        self.shift_size = (window_size[0] // 2, window_size[1] // 2)
        self.register_buffer("shift_mask", torch.tensor(0.0))  # to cleanly move to device
        self.shift_mask = (
            self._create_shift_mask(input_size, window_size, self.shift_size) if shift else torch.tensor(0.0)
        )
        self.shift_win = WindowShift2D(input_size, self.shift_size, reverse=False) if shift else nn.Identity()
        self.shift_win_rev = WindowShift2D(input_size, self.shift_size, reverse=True) if shift else nn.Identity()

        self.to_broadcast_mask = elt.Rearrange("bw ... -> () bw () ...") if shift else nn.Identity()
        bw = (input_size[0] // window_size[0]) * (input_size[1] // window_size[1])
        self.to_broadcast_score = elt.Rearrange("(b bw) ... -> b bw ...", bw=bw)
        self.to_merged_score = elt.Rearrange("b bw ... -> (b bw) ...", bw=bw)

        # Multi-head self-attention
        self.num_heads = num_heads
        self.scale = qk_scale or embed_dim**-0.5
        self.inv_embed_dim = 1.0 / math.sqrt(embed_dim)

        self.proj_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.drop_attn = nn.Dropout(drop_attn)
        self.drop_proj = nn.Dropout(drop_proj)
        self.softmax = nn.Softmax(dim=-1)

        self.to_qkv = elt.Rearrange("b n (qkv nh c) -> qkv b nh n c", qkv=3, nh=num_heads)

        # Input/output channels
        self.input_size = input_size
        self.out_size = input_size
        self.in_channels = embed_dim
        self.out_channels = embed_dim
        self.window_size = window_size

        # Others
        self.attn_weights: torch.Tensor = torch.tensor(0)

    def _create_shift_mask(
        self, input_size: Tuple[int, int], window_size: Tuple[int, int], shift_size: Tuple[int, int]
    ) -> torch.Tensor:
        id_map = torch.zeros((1, input_size[0], input_size[1], 1))
        shift_size = (window_size[0] // 2, window_size[1] // 2)

        h_slices = [(0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None)]
        w_slices = [(0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None)]

        cnt = 0
        for h_start, h_stop in h_slices:
            for w_start, w_stop in w_slices:
                id_map[:, h_start:h_stop, w_start:w_stop, :] = cnt
                cnt += 1

        id_map = id_map.view(1, -1, 1).contiguous()
        id_windows = self.to_windows(id_map).squeeze(-1)
        id_diff_windows = id_windows.unsqueeze(1) - id_windows.unsqueeze(2)

        return id_diff_windows.masked_fill(id_diff_windows != 0, float(-1e9)).masked_fill(
            id_diff_windows == 0, float(0.0)
        )

    def _create_indices(self, window_size: Tuple[int, int], max_distance: Tuple[int, int]) -> torch.Tensor:
        offsets = [0] + list(itertools.accumulate((2 * d + 1 for d in max_distance[:-1])))

        h_abs_dist = torch.arange(window_size[0])
        w_abs_dist = torch.arange(window_size[1])
        h_abs_dist = einops.repeat(h_abs_dist, "p -> p w", w=window_size[1]).flatten()
        w_abs_dist = einops.repeat(w_abs_dist, "p -> h p", h=window_size[0]).flatten()

        h_rel_dist = h_abs_dist.unsqueeze(0) - h_abs_dist.unsqueeze(1)
        w_rel_dist = w_abs_dist.unsqueeze(0) - w_abs_dist.unsqueeze(1)

        h_idx = torch.clamp(h_rel_dist, -max_distance[0], max_distance[0]) + max_distance[0] + offsets[0]
        w_idx = torch.clamp(w_rel_dist, -max_distance[1], max_distance[1]) + max_distance[1] + offsets[1]

        return torch.stack([h_idx, w_idx])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (B, N, C).
            mask (Optional[torch.Tensor], optional): Mask tensor ([B, BW, HEAD] N, N). Defaults to None.
                BW is the number of windows.

        Returns:
            torch.Tensor: Output tensor (B, N, C).
        """
        if self.shift:
            x = self.shift_win(x)
        x = self.to_windows(x)
        qkv = self.to_qkv(self.proj_qkv(x))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        s = torch.einsum("b h n c, b h m c -> b h n m", q, k)

        if self.bias_mode:
            assert self.indices is not None, "Indices must be created for bias mode."
            assert self.embedding_table is not None, "Embedding table must be created for bias mode."
            biases = self.embedding_table(self.indices).sum(0)
            biases = self.reshape_embedding(biases)
            s = s + biases
        elif self.context_mode:
            assert self.indices is not None, "Indices must be created for context mode."
            assert self.embedding_table_q is not None, "Embedding table Q must be created for context mode."
            assert self.embedding_table_k is not None, "Embedding table K must be created for context mode."
            q_embedding = self.embedding_table_q(self.indices).sum(0)
            k_embedding = self.embedding_table_k(self.indices).sum(0)
            q_embedding = self.reshape_embedding(q_embedding)
            k_embedding = self.reshape_embedding(k_embedding)
            s = s + torch.einsum("b h n c, n m h c -> b h n m", q, k_embedding)
            s = s + torch.einsum("b h n c, n m h c -> b h n m", k, q_embedding)

        if self.shift:
            s = self.to_broadcast_score(s) + self.to_broadcast_mask(self.shift_mask)
            s = self.to_merged_score(s)

        if mask is not None:
            s = self.to_broadcast_score(s) + mask
            s = self.to_merged_score(s)

        a = self.softmax(s)
        self.attn_weights = a
        a = self.drop_attn(a)

        x = torch.einsum("b h n m, b h m c -> b h n c", a, v)
        x = self.to_sequence(x)

        x = self.proj(x)
        x = self.drop_proj(x)

        x = self.to_spatial(x)

        if self.shift:
            x = self.shift_win_rev(x)

        return x


class FeedForward2D(nn.Module):
    """FeedForward2D

    The feed-forward module for the Swin Transformer is a simple two-layer MLP with an activation function. It is used
    to process the output of the multi-head self-attention module.

    References:
        Swin Transformer: Hierarchical Vision Transformer using Shifted Windows - Li et al.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        act_layer (Type[nn.Module], optional): Activation layer type. Defaults to nn.GELU.
        norm_layer (Optional[Type[nn.Module]], optional): Normalisation layer type. Defaults to nn.LayerNorm.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
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


class SwinTransformerBlock2D(nn.Module):
    """SwinTransformerBlock2D

    The Swin Transformer block consists of a multi-head self-attention module followed by a feed-forward module. The
    output of the feed-forward module is added to the input tensor and passed through a residual connection. It uses
    drop path for stochastic depth.

    References:
        Swin Transformer: Hierarchical Vision Transformer using Shifted Windows - Li et al.
        FractalNet: Ultra-Deep Neural Networks without Residuals - Larsson et al.

    Args:
        input_size (Tuple[int, int]): Size of the input image (height, width).
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        window_size (Tuple[int, int]): Size of the window (height, width).
        mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding dim.
        qkv_bias (bool, optional): Whether to add bias to the QKV linear layer.
        qk_scale (Optional[float], optional): Scaling factor for QK attention.
        drop (float, optional): Dropout rate.
        drop_attn (float, optional): Attention dropout rate.
        drop_path (float, optional): Stochastic depth rate.
        rpe_mode (RelativePositionalEmbeddingMode): Relative position encoding mode.
        shift (bool, optional): Whether to use shifted windows.
        act_layer (Type[nn.Module], optional): Activation layer type.
        norm_layer (Optional[Type[nn.Module]], optional): Normalisation layer type.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        embed_dim: int,
        num_heads: int,
        window_size: Tuple[int, int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        drop_attn: float = 0.0,
        drop_path: float = 0.0,
        rpe_mode: RelativePositionalEmeddingMode = RelativePositionalEmeddingMode.BIAS,
        shift: bool = False,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension {embed_dim} must be divisible by the number of heads {num_heads}"

        # compute padding for input
        padding_size = (
            window_size[0] - input_size[0] % window_size[0] if input_size[0] % window_size[0] != 0 else 0,
            window_size[1] - input_size[1] % window_size[1] if input_size[1] % window_size[1] != 0 else 0,
        )
        pad_input_size = (input_size[0] + padding_size[0], input_size[1] + padding_size[1])
        self.pad = (
            nn.ConstantPad2d((0, padding_size[1], 0, padding_size[0]), 0) if sum(padding_size) > 0 else nn.Identity()
        )
        self.register_buffer("padding_mask", torch.tensor(0.0))  # to cleanly move to device
        self.padding_mask = self._create_padding_mask(pad_input_size, window_size, padding_size)
        self.rearrange_input = (
            elt.Rearrange("b (h w) c -> b c h w", h=input_size[0], w=input_size[1])
            if sum(padding_size) > 0
            else nn.Identity()
        )
        self.rearrange_pad_input = (
            elt.Rearrange("b (h w) c -> b c h w", h=pad_input_size[0], w=pad_input_size[1])
            if sum(padding_size) > 0
            else nn.Identity()
        )
        self.rearrange_output = elt.Rearrange("b c h w -> b (h w) c") if sum(padding_size) > 0 else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_attn = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
        self.attn = Attention2D(
            pad_input_size,
            window_size,
            embed_dim,
            num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_attn=drop_attn,
            drop_proj=drop,
            rpe_mode=rpe_mode,
            shift=shift,
        )
        self.norm_proj = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
        self.proj = FeedForward2D(embed_dim, int(embed_dim * mlp_ratio), embed_dim, act_layer, norm_layer)

        self.input_size = input_size
        self.output_size = input_size
        self.in_channels = embed_dim
        self.out_channels = embed_dim
        self.padding_size = padding_size

    def _create_padding_mask(
        self, input_size: Tuple[int, int], window_size: Tuple[int, int], pad_size: Tuple[int, int]
    ) -> torch.Tensor:
        id_map = torch.zeros((1, input_size[0], input_size[1], 1))
        cnt = 1
        for h, p in itertools.product(range(input_size[0]), range(1, pad_size[0] + 1)):
            id_map[:, h, -p, :] = cnt
            cnt += 1
        for w, p in itertools.product(range(input_size[1]), range(1, pad_size[1] + 1)):
            id_map[:, -p, w, :] = cnt
            cnt += 1
        id_map = id_map.view(1, -1, 1).contiguous()
        id_windows = einops.rearrange(
            (id_map),
            "b (hdm hm wdm wm) c -> (b hdm wdm) (hm wm) c",
            hdm=input_size[0] // window_size[0],
            wdm=input_size[1] // window_size[1],
            hm=window_size[0],
            wm=window_size[1],
        ).squeeze(-1)
        id_diff_windows = id_windows.unsqueeze(1) - id_windows.unsqueeze(2)
        id_diff_windows = einops.rearrange(
            id_diff_windows,
            "bw ... -> () bw () ...",
        )
        return id_diff_windows.masked_fill(id_diff_windows != 0, float(1)).masked_fill(id_diff_windows == 0, float(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (B, N, C).

        Returns:
            torch.Tensor: Output tensor (B, N, C).
        """

        x = self.rearrange_input(x)
        x = self.pad(x)
        x = self.rearrange_output(x)

        skip = x
        x = self.drop_path(self.attn(x, self.padding_mask)) + skip
        x = self.norm_attn(x)

        x = self.rearrange_pad_input(x)
        x = x[..., : self.input_size[0], : self.input_size[1]]
        x = self.rearrange_output(x)

        skip = x
        x = self.drop_path(self.proj(x)) + skip
        x = self.norm_proj(x)

        return x


class SwinTransformerStage2D(nn.Module):
    """SwinTransformerStage

    The Swin Transformer stage consists of a patch embedding module followed by a series of Swin Transformer blocks.
    First, the patch embedding model reduces the size of the input, yielding a higher-level representation. Then, the
    Swin Transformer blocks attend the input to prodcues a semantically rich encoding.

    References:
        Swin Transformer: Hierarchical Vision Transformer using Shifted Windows - Li et al.

    Args:
        input_size (Tuple[int, int]): Size of the input image (height, width).
        in_channels (int): Number of input channels.
        embed_dim (int): Embedding dimension.
        num_blocks (int): Number of blocks in the stage.
        patch_module (Type[PatchEmbedding2D] | Type[PatchMerging2D]): Patch embedding module type.
        patch_window_size (Tuple[int, int]): Size of the patch window (height, width).
        block_window_size (Tuple[int, int]): Size of the block window (height, width).
        num_heads (int): Number of attention heads.
        mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding dim.
        qkv_bias (bool, optional): Whether to add bias to the QKV linear layer.
        qk_scale (Optional[float], optional): Scaling factor for QK attention.
        drop (float, optional): Dropout rate.
        drop_attn (float, optional): Attention dropout rate.
        drop_path (Optional[List[float]], optional): Stochastic depth rate.
        rpe_mode (RelativePositionalEmbeddingMode): Whether to use relative position encoding.
        act_layer (Type[nn.Module], optional): Activation layer type.
        norm_layer_pre_block (Optional[Type[nn.Module]], optional): Normalisation in patch.
        norm_layer_block (Optional[Type[nn.Module]], optional): Normalisation in the block.
        patch_mode (PatchMode): Patch embedding mode.
    """

    def __init__(
        self,
        # Stage parameters
        input_size: Tuple[int, int],
        in_channels: int,
        embed_dim: int,
        num_blocks: int,
        patch_module: Type[PatchEmbedding2D] | Type[PatchMerging2D],
        # Window parameters
        patch_window_size: Tuple[int, int],
        block_window_size: Tuple[int, int],
        # Block parameters
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        drop_attn: float = 0.0,
        drop_path: Optional[List[float]] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        # Normalisation parameters
        norm_layer_pre_block: Optional[Type[nn.Module]] = nn.LayerNorm,
        norm_layer_block: Optional[Type[nn.Module]] = nn.LayerNorm,
        # Mode parameters
        patch_mode: PatchMode = PatchMode.CONCATENATE,
        rpe_mode: RelativePositionalEmeddingMode = RelativePositionalEmeddingMode.BIAS,
    ) -> None:
        super().__init__()

        assert (
            drop_path is None or len(drop_path) == num_blocks
        ), "Length of drop_path must be equal to the number of blocks."

        if patch_module == PatchEmbedding2D:
            patch_config = dict(
                input_size=input_size,
                patch_size=patch_window_size,
                in_channels=in_channels,
                embed_dim=embed_dim,
                norm_layer=norm_layer_pre_block,
                mode=patch_mode,
            )
        else:
            patch_config = dict(
                input_size=input_size,
                merge_size=patch_window_size,
                embed_dim=embed_dim,
                norm_layer=norm_layer_pre_block,
                mode=patch_mode,
            )
            in_channels = embed_dim
        self.patch = patch_module(**patch_config)  # type: ignore

        input_size = (input_size[0] // patch_window_size[0], input_size[1] // patch_window_size[1])
        self.blocks = nn.ModuleDict()
        for i in range(num_blocks):
            name = f"block_{i}"
            self.blocks[name] = SwinTransformerBlock2D(
                input_size,
                self.patch.out_channels,
                num_heads,
                block_window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                drop_attn=drop_attn,
                drop_path=0.0 if drop_path is None else drop_path[i],
                rpe_mode=rpe_mode,
                act_layer=act_layer,
                norm_layer=norm_layer_block,
                shift=i % 2 == 1,
            )

        self.input_size = (input_size[0] * patch_window_size[0], input_size[1] * patch_window_size[1])
        self.output_size = input_size
        self.in_channels = in_channels
        self.out_channels = self.patch.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (B, C, H, W) or (B, N, C). Depending on the patch module.

        Returns:
            torch.Tensor: Output tensor (B, N, C).
        """

        x = self.patch(x)
        for block in self.blocks.values():
            x = block(x)
        return x


@dataclass
class SwinTransformerConfig2D:
    """
    Configuration class for Swin Transformer.

    This dataclass holds all the necessary parameters to initialize and
    configure a Swin Transformer model.

    Attributes:
        input_size (tuple): Size of the input image (height, width).
        in_channels (int): Number of input channels.
        embed_dim (int): Embedding dimension.
        num_blocks (List[int]): Number of blocks in each stage.
        patch_window_size (List[tuple]): Patch window sizes for each stage.
        block_window_size (List[tuple]): Block window sizes for each stage.
        num_heads (List[int]): Number of attention heads for each stage.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        qkv_bias (bool): Whether to add bias to the QKV linear layer.
        qk_scale (Optional[float]): Scaling factor for QK attention.
        drop (float): Dropout rate.
        drop_attn (float): Attention dropout rate.
        drop_path (float): Stochastic depth rate.
        act_layer (type): Activation layer type.
        patch_mode (Optional[List[PatchMode]]): Patch embedding mode.
        rpe_mode (RelativePositonalEmbeddingMode): Whether to use relative position encoding.
    """

    # Stage parameters
    input_size: Tuple[int, int]
    in_channels: int
    embed_dim: int
    num_blocks: List[int]
    # Window parameters
    patch_window_size: List[Tuple[int, int]]
    block_window_size: List[Tuple[int, int]]
    # Block parameters
    num_heads: List[int]
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    qk_scale: Optional[float] = None
    drop: float = 0.0
    drop_attn: float = 0.0
    drop_path: float = 0.1
    act_layer: Type[nn.Module] = nn.GELU
    # Mode parameters
    patch_mode: Optional[List[PatchMode]] = None
    rpe_mode: RelativePositionalEmeddingMode = RelativePositionalEmeddingMode.BIAS

    def __post_init__(self):
        """
        Validate the configuration after initialization.

        Raises:
            ValueError: If the lengths of num_blocks, patch_window_size,
                        block_window_size, and num_heads are not equal.
        """
        if not (
            len(self.num_blocks) == len(self.patch_window_size) == len(self.block_window_size) == len(self.num_heads)
        ):
            raise ValueError(
                "Lengths of num_blocks, patch_window_size, " "block_window_size, and num_heads must be equal."
            )

        if not len(self.num_blocks) > 0:
            raise ValueError("At least one stage must be defined.")

        if self.patch_mode is None:
            self.patch_mode = [PatchMode.CONVOLUTION] + [PatchMode.CONCATENATE] * (len(self.num_blocks) - 1)
        else:
            assert len(self.patch_mode) == len(self.num_blocks), "Length of patch_mode must be equal to num_blocks."


class SwinTransformer2D(nn.Module):
    """SwinTransformer2D

    The Swin Transformer is a hierarchical vision transformer that uses shifted windows to process an image.

    References:
        Swin Transformer: Hierarchical Vision Transformer using Shifted Windows - Li et al.
        Deep Networks with Stochastic Depth - Huang et al.

    Args:
        config (SwinTransformerConfig2D): Configuration class for Swin Transformer.
    """

    def __init__(self, config: SwinTransformerConfig2D) -> None:
        super().__init__()

        assert config.patch_mode is not None, "Should not be None"

        self.stages = nn.ModuleDict()
        stochastic_depth_decay = [x.item() for x in torch.linspace(0, config.drop_path, sum(config.num_blocks))]
        out_channels = config.embed_dim
        input_size = config.input_size
        for i, (nb, pws, bws, nh) in enumerate(
            zip(config.num_blocks, config.patch_window_size, config.block_window_size, config.num_heads)
        ):
            name = f"stage_{i}"
            patch_module = PatchEmbedding2D if i == 0 else PatchMerging2D
            norm_layer_pre_block = None if i == 0 else nn.LayerNorm
            self.stages[name] = SwinTransformerStage2D(
                input_size,
                config.in_channels,
                out_channels,
                num_blocks=nb,
                patch_module=patch_module,
                patch_window_size=pws,
                block_window_size=bws,
                num_heads=nh,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                qk_scale=config.qk_scale,
                drop=config.drop,
                drop_attn=config.drop_attn,
                drop_path=stochastic_depth_decay[sum(config.num_blocks[:i]) : sum(config.num_blocks[: i + 1])],
                act_layer=config.act_layer,
                norm_layer_pre_block=norm_layer_pre_block,
                norm_layer_block=nn.LayerNorm,
                patch_mode=config.patch_mode[i],
                rpe_mode=config.rpe_mode,
            )
            input_size = (input_size[0] // pws[0], input_size[1] // pws[1])
            out_channels = self.stages[name].out_channels

        self.input_size = [s.input_size for s in self.stages.values()]
        self.output_size = [s.output_size for s in self.stages.values()]
        self.in_channels = [s.in_channels for s in self.stages.values()]
        self.out_channels = [s.out_channels for s in self.stages.values()]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = []
        for stage in self.stages.values():
            x = stage(x)
            out.append(x)
        return out
