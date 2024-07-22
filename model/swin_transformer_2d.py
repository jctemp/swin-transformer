import itertools
import math
from typing import List, Optional, Tuple, Type

import einops
import einops.layers.torch as elt
import torch
import torch.nn as nn
from timm.layers import DropPath


class PatchEmbedding2D(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        in_channels: int,
        embed_dim: int,
        norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        assert (
            input_size[0] % patch_size[0] == 0 and input_size[1] % patch_size[1] == 0
        ), f"Input size {input_size} must be divisible by the patch size {patch_size}"

        # Output transformation
        self.reshape = elt.Rearrange("b c h w -> b (h w) c")

        # Patch embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

        # Auxilliary
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.reshape(x)
        x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, int],
        merge_size: Tuple[int, int],
        embed_dim: int,
        norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert (
            input_size[0] % merge_size[0] == 0 and input_size[1] % merge_size[1] == 0
        ), f"Input size {input_size} must be divisible by the merge size {merge_size}"

        # Input/Output transformations
        self.expand = elt.Rearrange("b (h w) c -> b h w c", h=input_size[0], w=input_size[1])
        self.squeeze = elt.Rearrange("b h w c -> b (h w) c")

        # Merge parameters
        self.indices = list(itertools.product(*[list(range(0, i)) for i in merge_size]))
        self.merge_size = merge_size
        self.merge_factor = len(self.indices)
        self.merge_target = len(self.indices) // 2

        # Projection and normalization
        self.proj = nn.Linear(self.merge_factor * embed_dim, self.merge_target * embed_dim)
        self.norm = norm_layer(self.merge_factor * embed_dim) if norm_layer is not None else nn.Identity()

        # Auxilliary
        self.input_size = input_size
        self.merge_size = merge_size
        self.in_channels = embed_dim
        self.out_channels = embed_dim * self.merge_target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = torch.cat([x[:, h :: self.merge_size[0], w :: self.merge_size[1], :] for h, w in self.indices], dim=-1)
        x = self.squeeze(x)
        x = self.norm(x)
        x = self.proj(x)
        return x


class WindowShift2D(nn.Module):
    def __init__(self, input_size: Tuple[int, int], shift_size: Tuple[int, int], reversed: bool = False) -> None:
        super().__init__()
        # Input/Output transformations
        self.expand = elt.Rearrange("b (h w) c -> b h w c", h=input_size[0], w=input_size[1])
        self.squeeze = elt.Rearrange("b h w c -> b (h w) c")

        # Shift parameters
        self.shift_size = shift_size if reversed else (-shift_size[0], -shift_size[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = torch.roll(x, shifts=self.shift_size, dims=(1, 2))
        x = self.squeeze(x)
        return x


class Attention2D(nn.Module):
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
        rpe: bool = True,
        rpe_dist: Optional[Tuple[int, int]] = None,
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
        #   Clippling possible because window_size produces significant small sequences
        #
        #   Rethinking and Improving Relative Position Encoding for Vision Transformer - Wu et al. (2021)
        #   see figure 1 (a) and eq. 13 + 14
        self.rpe = rpe
        max_distance = (window_size[0] - 1, window_size[1] - 1) if rpe_dist is None else rpe_dist
        self.embedding_table = nn.Embedding(sum(2 * d + 1 for d in max_distance), num_heads)
        self.register_buffer("indices", torch.tensor(0))  # to cleanly move to device
        self.indices = self.create_bias_indices(window_size, max_distance)

        self.to_broadcast_embedding = elt.Rearrange("h w nh -> nh h w")

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
            self.create_shift_mask(input_size, window_size, self.shift_size) if shift else torch.tensor(0.0)
        )
        self.shift_win = WindowShift2D(input_size, self.shift_size, reversed=False) if shift else nn.Identity()
        self.shift_win_rev = WindowShift2D(input_size, self.shift_size, reversed=True) if shift else nn.Identity()

        self.to_broadcast_mask = elt.Rearrange("bw ... -> () bw () ...") if shift else nn.Identity()
        self.to_broadcast_score = (
            elt.Rearrange("(b bw) ... -> b bw ...", bw=self.shift_mask.shape[0]) if shift else nn.Identity()
        )
        self.to_merged_score = (
            elt.Rearrange("b bw ... -> (b bw) ...", bw=self.shift_mask.shape[0]) if shift else nn.Identity()
        )

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
        self.in_channels = embed_dim
        self.out_channels = embed_dim

        # Others
        self.attn_weights: torch.Tensor = torch.tensor(0)

    def create_shift_mask(
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

    def create_bias_indices(self, window_size: Tuple[int, int], max_distance: Tuple[int, int]) -> torch.Tensor:
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
        if self.shift:
            x = self.shift_win(x)

        x = self.to_windows(x)
        qkv = self.to_qkv(self.proj_qkv(x))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        s = torch.einsum("b h n c, b h m c -> b h n m", q, k)
        if self.rpe:
            biases = self.embedding_table(self.indices).sum(0)
            biases = self.to_broadcast_embedding(biases)
            s = s + biases

        if self.shift:
            s = self.to_broadcast_score(s) + self.to_broadcast_mask(self.shift_mask)
            s = self.to_merged_score(s)

        if mask is not None:
            s = s.masked_fill(mask == 0, float(-1e9))

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
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x


class SwinTransformerBlock2D(nn.Module):
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
        rpe: bool = True,
        rpe_dist: Optional[Tuple[int, int]] = None,
        shift: bool = False,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        assert (
            input_size[0] % window_size[0] == 0 and input_size[1] % window_size[1] == 0
        ), f"Input size {input_size} must be divisible by the window size {window_size}"
        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension {embed_dim} must be divisible by the number of heads {num_heads}"

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_attn = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
        self.attn = Attention2D(
            input_size,
            window_size,
            embed_dim,
            num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_attn=drop_attn,
            drop_proj=drop,
            rpe=rpe,
            rpe_dist=rpe_dist,
            shift=shift,
        )
        self.norm_proj = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
        self.proj = FeedForward2D(embed_dim, int(embed_dim * mlp_ratio), embed_dim, act_layer, norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.norm_attn(x)
        x = self.drop_path(self.attn(x)) + skip

        skip = x
        x = self.norm_proj(x)
        x = self.drop_path(self.proj(x)) + skip

        return x


class SwinTransformerStage(nn.Module):
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
        drop_path: float = 0.0,
        rpe: bool = True,
        rpe_dist: Optional[Tuple[int, int]] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        # Normalization parameters
        norm_layer_pre_block: Optional[Type[nn.Module]] = nn.LayerNorm,
        norm_layer_block: Optional[Type[nn.Module]] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        if patch_module == PatchEmbedding2D:
            patch_config = dict(
                input_size=input_size,
                patch_size=patch_window_size,
                in_channels=in_channels,
                embed_dim=embed_dim,
                norm_layer=norm_layer_pre_block,
            )
        else:
            patch_config = dict(
                input_size=input_size,
                merge_size=patch_window_size,
                embed_dim=embed_dim,
                norm_layer=norm_layer_pre_block,
            )
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
                drop_path=drop_path,
                rpe=rpe,
                rpe_dist=rpe_dist,
                act_layer=act_layer,
                norm_layer=norm_layer_block,
                shift=i % 2 == 1,
            )

        self.in_channels = in_channels
        self.out_channels = self.patch.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x)
        for block in self.blocks.values():
            x = block(x)
        return x


class SwinTransformer2D(nn.Module):
    def __init__(
        self,
        # Stage parameters
        input_size: Tuple[int, int],
        in_channels: int,
        embed_dim: int,
        num_blocks: List[int],
        # Window parameters
        patch_window_size: List[Tuple[int, int]],
        block_window_size: List[Tuple[int, int]],
        # Block parameters
        num_heads: List[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        drop_attn: float = 0.0,
        drop_path: float = 0.1,
        rpe: bool = True,
        act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        assert (
            len(num_blocks) == len(block_window_size) == len(patch_window_size) == len(num_heads)
        ), "Number of stages, block window sizes, patch window sizes, and number of heads must match"
        assert len(num_blocks) > 0, "At least one stage is required"

        self.stages = nn.ModuleDict()
        out_channels = embed_dim
        print(f"Pre: {input_size}, {in_channels}")
        for i, (nb, pws, bws, nh) in enumerate(zip(num_blocks, patch_window_size, block_window_size, num_heads)):
            name = f"stage_{i}"
            patch_module = PatchEmbedding2D if i == 0 else PatchMerging2D
            norm_layer_pre_block = None if i == 0 else nn.LayerNorm
            self.stages[name] = SwinTransformerStage(
                input_size,
                in_channels,
                out_channels,
                num_blocks=nb,
                patch_module=patch_module,
                patch_window_size=pws,
                block_window_size=bws,
                num_heads=nh,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                drop_attn=drop_attn,
                drop_path=drop_path,
                rpe=rpe,
                act_layer=act_layer,
                norm_layer_pre_block=norm_layer_pre_block,
                norm_layer_block=nn.LayerNorm,
            )
            input_size = (input_size[0] // pws[0], input_size[1] // pws[1])
            out_channels = self.stages[name].out_channels
            print(f"Stage {i}: {input_size}, {out_channels}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = []
        for stage in self.stages.values():
            x = stage(x)
            out.append(x)
        return out
