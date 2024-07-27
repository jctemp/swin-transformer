import torch
import torch.nn as nn

from .swin_transformer_3d import (
    Attention3D,
    PatchEmbedding3D,
    PatchMerging3D,
    PatchMode,
    RelativePositionalEmeddingMode,
    SwinTransformer3D,
    SwinTransformerConfig3D,
    WindowShift3D,
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_patch_embedding_3d():
    input_size = (32, 224, 224)
    patch_size = (4, 16, 16)
    in_channels = 3
    embed_dim = 512

    x = torch.randn((2, in_channels, *input_size)).to(DEVICE)

    patch_mode = PatchMode.CONCATENATE
    pe = torch.jit.script(
        PatchEmbedding3D(input_size, patch_size, in_channels, embed_dim, nn.LayerNorm, patch_mode).to(DEVICE)
    )
    y = pe(x)  # type: ignore

    assert tuple(y.shape) == (  # type: ignore
        2,
        int(torch.prod(torch.tensor(input_size)) / torch.prod(torch.tensor(patch_size))),
        embed_dim,
    )

    patch_mode = PatchMode.CONVOLUTION
    pe = torch.jit.script(
        PatchEmbedding3D(input_size, patch_size, in_channels, embed_dim, nn.LayerNorm, patch_mode).to(DEVICE)
    )
    y = pe(x)  # type: ignore

    assert tuple(y.shape) == (  # type: ignore
        2,
        int(torch.prod(torch.tensor(input_size)) / torch.prod(torch.tensor(patch_size))),
        embed_dim,
    )


def test_patch_merging_3d():
    input_size = (4, 4, 6)
    merge_size = (2, 2, 3)
    embed_dim = 12

    x = torch.randn((2, input_size[0] * input_size[1] * input_size[2], embed_dim)).to(DEVICE)

    patch_mode = PatchMode.CONCATENATE
    pm = torch.jit.script(PatchMerging3D(input_size, merge_size, embed_dim, nn.LayerNorm, patch_mode).to(DEVICE))
    y = pm(x)  # type: ignore

    assert tuple(y.shape) == ((2, 2 * 2 * 2, embed_dim * 2 * 3))  # type: ignore

    patch_mode = PatchMode.CONVOLUTION
    pm = torch.jit.script(PatchMerging3D(input_size, merge_size, embed_dim, nn.LayerNorm, patch_mode).to(DEVICE))
    y = pm(x)  # type: ignore

    assert tuple(y.shape) == ((2, 2 * 2 * 2, embed_dim * 2 * 3))  # type: ignore


def test_window_shift_3d():
    input_size = (4, 4, 6)
    shift_size = (1, 1, 1)
    embed_dim = 12

    x = torch.randn((2, input_size[0] * input_size[1] * input_size[2], embed_dim)).to(DEVICE)
    cs = torch.jit.script(WindowShift3D(input_size, shift_size).to(DEVICE))
    csr = torch.jit.script(WindowShift3D(input_size, shift_size, reverse=True).to(DEVICE))
    assert torch.equal(x, csr(cs(x)))  # type: ignore


def test_attention_3d():
    input_size = (4, 6, 6)
    window_size = (2, 2, 2)
    embed_dim = 12
    num_heads = 4

    x = torch.randn((2, input_size[0] * input_size[1] * input_size[2], embed_dim)).to(DEVICE)

    for mode in [
        RelativePositionalEmeddingMode.BIAS,
        RelativePositionalEmeddingMode.CONTEXT,
        RelativePositionalEmeddingMode.NONE,
    ]:
        attn = torch.jit.script(
            Attention3D(
                input_size,
                window_size,
                embed_dim,
                num_heads,
                qkv_bias=True,
                qk_scale=num_heads**-0.5,
                drop_attn=0.0,
                drop_proj=0.0,
                rpe_mode=mode,
                shift=True,
            ).to(DEVICE)
        )
        x_out = attn(x)  # type: ignore
        a = attn.attn_weights  # type: ignore
        m = attn.shift_mask  # type: ignore

        assert x_out.shape == x.shape  # type: ignore
        assert torch.equal(
            a[:18, 0].masked_fill_(a[:18, 0] != 0, 1), m.masked_fill_(m == 0, 1).masked_fill_(m == -1e9, 0)
        )


def test_swin_transformer_3d():
    batch = 2
    input_size = (32, 224, 224)
    in_channels = 3
    embed_dim = 96
    num_blocks = [2, 6, 2]
    patch_window_size = [(4, 4, 4), (2, 2, 2), (2, 2, 2)]
    block_window_size = [(2, 7, 7), (2, 7, 7), (2, 7, 7)]
    num_heads = [6, 6, 6]

    x = torch.randn((batch, in_channels, *input_size)).to(DEVICE)
    config = SwinTransformerConfig3D(
        input_size,
        in_channels,
        embed_dim,
        num_blocks,
        patch_window_size,
        block_window_size,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        drop_attn=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        patch_mode=[PatchMode.CONVOLUTION] + [PatchMode.CONCATENATE] * (len(num_blocks) - 1),
        rpe_mode=RelativePositionalEmeddingMode.CONTEXT,
    )

    model = torch.jit.script(SwinTransformer3D(config).to(DEVICE))

    x_out = model(x)  # type: ignore

    ins = input_size
    dims = embed_dim
    for xo, pws in zip(x_out, patch_window_size):  # type: ignore
        ins = (ins[0] // pws[0], ins[1] // pws[1], ins[2] // pws[2])
        assert xo.shape == (batch, ins[0] * ins[1] * ins[2], dims)
        dims = dims * 4
