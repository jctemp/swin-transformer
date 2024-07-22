
import torch
import torch.nn as nn

from .swin_transformer_2d import (
    Attention2D,
    PatchEmbedding2D,
    PatchMerging2D,
    SwinTransformer2D,
    WindowShift2D,
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_patch_embedding_2d():
    input_size = (224, 224)
    patch_size = (16, 16)
    embed_dim = 3
    embed_dim = 512

    x = torch.randn((2, embed_dim, *input_size)).to(DEVICE)
    pe = torch.jit.script(PatchEmbedding2D(input_size, patch_size, embed_dim, embed_dim, nn.LayerNorm).to(DEVICE))
    pe = pe(x)  # type: ignore

    assert tuple(pe.shape) == (  # type: ignore
        2,
        int(torch.prod(torch.tensor(input_size)) / torch.prod(torch.tensor(patch_size))),
        embed_dim,
    )


def test_patch_merging_2d():
    input_size = (4, 6)
    merge_size = (2, 3)
    embed_dim = 12

    x = torch.randn((2, input_size[0] * input_size[1], embed_dim)).to(DEVICE)
    pm = torch.jit.script(PatchMerging2D(input_size, merge_size, embed_dim, nn.LayerNorm).to(DEVICE))
    pm = pm(x)  # type: ignore

    assert tuple(pm.shape) == ((2, 2 * 2, 2 * 3 * 6))  # type: ignore


def test_window_shift_2d():
    input_size = (4, 6)
    shift_size = (1, 1)
    embed_dim = 12

    x = torch.randn((2, input_size[0] * input_size[1], embed_dim)).to(DEVICE)
    cs = torch.jit.script(WindowShift2D(input_size, shift_size).to(DEVICE))
    csr = torch.jit.script(WindowShift2D(input_size, shift_size, reversed=True).to(DEVICE))
    assert torch.equal(x, csr(cs(x)))  # type: ignore


def test_attention_2d():
    # Example
    input_size = (6, 6)
    window_size = (2, 2)
    embed_dim = 12
    num_heads = 4

    x = torch.randn((2, input_size[0] * input_size[1], embed_dim)).to(DEVICE)
    attn = torch.jit.script(
        Attention2D(
            input_size,
            window_size,
            embed_dim,
            num_heads,
            qkv_bias=True,
            qk_scale=num_heads**-0.5,
            drop_attn=0.0,
            drop_proj=0.0,
            rpe=True,
            rpe_dist=(input_size[0] // window_size[0], input_size[1] // window_size[1]),
            shift=True,
        ).to(DEVICE)
    )
    x_out = attn(x)  # type: ignore
    a = attn.attn_weights  # type: ignore
    m = attn.shift_mask  # type: ignore

    assert x_out.shape == x.shape  # type: ignore
    assert torch.equal(a[:9, 0].masked_fill_(a[:9, 0] != 0, 1), m.masked_fill_(m == 0, 1).masked_fill_(m == -1e9, 0))


def test_swin_transformer_2d():
    # Example
    batch = 2
    input_size = (224, 224)
    in_channels = 3
    embed_dim = 96
    num_blocks = [2, 6, 2]
    patch_window_size = [(4, 4), (2, 2), (2, 2)]
    block_window_size = [(7, 7), (7, 7), (7, 7)]
    num_heads = [6, 6, 6]

    x = torch.randn((batch, in_channels, *input_size)).to(DEVICE)
    model = torch.jit.script(
        SwinTransformer2D(
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
            rpe=True,
            act_layer=nn.GELU,
        ).to(DEVICE)
    )

    x_out = model(x)  # type: ignore

    ins = input_size
    dims = embed_dim
    for xo, pws in zip(x_out, patch_window_size):  # type: ignore
        ins = (ins[0] // pws[0], ins[1] // pws[1])
        assert xo.shape == (batch, ins[0] * ins[1], dims)
        dims = dims * 2
