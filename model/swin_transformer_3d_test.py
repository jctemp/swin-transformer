import pytest
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


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize(
    "input_size,patch_size,in_channels,embed_dim,patch_mode",
    [
        ((32, 224, 224), (4, 16, 16), 3, 512, PatchMode.CONCATENATE),
        ((32, 224, 224), (4, 16, 16), 3, 512, PatchMode.CONVOLUTION),
        ((64, 128, 128), (8, 8, 8), 1, 256, PatchMode.CONCATENATE),
        ((64, 128, 128), (8, 8, 8), 1, 256, PatchMode.CONVOLUTION),
    ],
)
def test_patch_embedding_3d(device, input_size, patch_size, in_channels, embed_dim, patch_mode):
    x = torch.randn((2, in_channels, *input_size)).to(device)
    pe = PatchEmbedding3D(input_size, patch_size, in_channels, embed_dim, nn.LayerNorm, patch_mode).to(device)
    y = pe(x)

    expected_shape = (
        2,
        int(torch.prod(torch.tensor(input_size)) / torch.prod(torch.tensor(patch_size))),
        embed_dim,
    )

    assert y.shape == expected_shape, f"Expected shape {expected_shape}, got {y.shape}"
    assert y.device.type == device.type, f"Expected device type {device.type}, got {y.device.type}"
    assert not torch.isnan(y).any(), "Output contains NaN values"
    assert not torch.isinf(y).any(), "Output contains infinite values"


@pytest.mark.parametrize(
    "input_size,merge_size,embed_dim,patch_mode",
    [
        ((4, 4, 6), (2, 2, 3), 12, PatchMode.CONCATENATE),
        ((4, 4, 6), (2, 2, 3), 12, PatchMode.CONVOLUTION),
        ((8, 8, 8), (2, 2, 2), 64, PatchMode.CONCATENATE),
        ((8, 8, 8), (2, 2, 2), 64, PatchMode.CONVOLUTION),
    ],
)
def test_patch_merging_3d(device, input_size, merge_size, embed_dim, patch_mode):
    x = torch.randn((2, input_size[0] * input_size[1] * input_size[2], embed_dim)).to(device)
    pm = PatchMerging3D(input_size, merge_size, embed_dim, nn.LayerNorm, patch_mode).to(device)
    y = pm(x)

    expected_shape = (
        2,
        (input_size[0] // merge_size[0]) * (input_size[1] // merge_size[1]) * (input_size[2] // merge_size[2]),
        embed_dim * merge_size[0] * merge_size[1] * merge_size[2] // 2,
    )

    assert y.shape == expected_shape, f"Expected shape {expected_shape}, got {y.shape}"
    assert y.device.type == device.type, f"Expected device type {device.type}, got {y.device.type}"
    assert not torch.isnan(y).any(), "Output contains NaN values"
    assert not torch.isinf(y).any(), "Output contains infinite values"


@pytest.mark.parametrize(
    "input_size,shift_size,embed_dim",
    [
        ((4, 4, 6), (1, 1, 1), 12),
        ((8, 8, 8), (2, 2, 2), 64),
    ],
)
def test_window_shift_3d(device, input_size, shift_size, embed_dim):
    x = torch.randn((2, input_size[0] * input_size[1] * input_size[2], embed_dim)).to(device)
    cs = WindowShift3D(input_size, shift_size).to(device)
    csr = WindowShift3D(input_size, shift_size, reverse=True).to(device)

    shifted = cs(x)
    restored = csr(shifted)

    assert torch.allclose(x, restored, atol=1e-6), "Window shift operation is not reversible"
    assert shifted.shape == x.shape, f"Expected shape {x.shape}, got {shifted.shape}"
    assert shifted.device.type == device.type, f"Expected device type {device.type}, got {shifted.device.type}"


@pytest.mark.parametrize(
    "input_size,window_size,embed_dim,num_heads,rpe_mode",
    [
        ((4, 6, 6), (2, 2, 2), 12, 4, RelativePositionalEmeddingMode.BIAS),
        ((4, 6, 6), (2, 2, 2), 12, 4, RelativePositionalEmeddingMode.CONTEXT),
        ((4, 6, 6), (2, 2, 2), 12, 4, RelativePositionalEmeddingMode.NONE),
        ((8, 8, 8), (4, 4, 4), 64, 8, RelativePositionalEmeddingMode.BIAS),
    ],
)
def test_attention_3d(device, input_size, window_size, embed_dim, num_heads, rpe_mode):
    x = torch.randn((2, input_size[0] * input_size[1] * input_size[2], embed_dim)).to(device)
    attn = Attention3D(
        input_size,
        window_size,
        embed_dim,
        num_heads,
        qkv_bias=True,
        qk_scale=num_heads**-0.5,
        drop_attn=0.0,
        drop_proj=0.0,
        rpe_mode=rpe_mode,
        shift=True,
    ).to(device)

    x_out = attn(x)
    a = attn.attn_weights
    m = attn.shift_mask

    assert x_out.shape == x.shape, f"Expected shape {x.shape}, got {x_out.shape}"
    assert x_out.device.type == device.type, f"Expected device type {device.type}, got {x_out.device.type}"
    assert not torch.isnan(x_out).any(), "Output contains NaN values"
    assert not torch.isinf(x_out).any(), "Output contains infinite values"

    # Check if the attention mask is applied correctly
    mask_check = torch.equal(
        a[:27, 0].masked_fill_(a[:27, 0] != 0, 1)[: m.shape[0]], m.masked_fill_(m == 0, 1).masked_fill_(m == -1e9, 0)
    )
    assert mask_check, "Attention mask is not applied correctly"


@pytest.mark.parametrize(
    "config",
    [
        SwinTransformerConfig3D(
            input_size=(32, 224, 224),
            in_channels=3,
            embed_dim=96,
            num_blocks=[2, 6, 2],
            patch_window_size=[(4, 4, 4), (2, 2, 2), (2, 2, 2)],
            block_window_size=[(2, 7, 7), (2, 7, 7), (2, 7, 7)],
            num_heads=[6, 6, 6],
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            drop_attn=0.0,
            drop_path=0.0,
            act_layer=nn.GELU(),
            patch_mode=[PatchMode.CONVOLUTION] + [PatchMode.CONCATENATE] * 2,
            rpe_mode=RelativePositionalEmeddingMode.CONTEXT,
        ),
        SwinTransformerConfig3D(
            input_size=(64, 128, 128),
            in_channels=1,
            embed_dim=48,  # Changed from 64 to 48
            num_blocks=[2, 2, 2, 2],
            patch_window_size=[(4, 4, 4), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
            block_window_size=[(4, 4, 4), (4, 4, 4), (4, 4, 4), (4, 4, 4)],
            num_heads=[4, 8, 16, 32],
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.1,
            drop_attn=0.1,
            drop_path=0.1,
            act_layer=nn.GELU(),
            patch_mode=[PatchMode.CONVOLUTION] + [PatchMode.CONCATENATE] * 3,
            rpe_mode=RelativePositionalEmeddingMode.BIAS,
        ),
    ],
)
def test_swin_transformer_3d(device, config):
    batch = 2
    x = torch.randn((batch, config.in_channels, *config.input_size)).to(device)
    model = SwinTransformer3D(config).to(device)

    x_out = model(x)

    assert len(x_out) == len(config.num_blocks), f"Expected {len(config.num_blocks)} outputs, got {len(x_out)}"

    ins = config.input_size
    dims = config.embed_dim
    for i, (xo, pws) in enumerate(zip(x_out, config.patch_window_size)):
        ins = tuple(ins[j] // pws[j] for j in range(3))
        expected_shape = (batch, ins[0] * ins[1] * ins[2], dims)
        assert xo.shape == expected_shape, f"Stage {i}: Expected shape {expected_shape}, got {xo.shape}"
        assert xo.device.type == device.type, f"Stage {i}: Expected device type {device.type}, got {xo.device.type}"
        assert not torch.isnan(xo).any(), f"Stage {i}: Output contains NaN values"
        assert not torch.isinf(xo).any(), f"Stage {i}: Output contains infinite values"
        dims *= 4 if i < len(config.num_blocks) - 1 else 1  # Quadruple the dimensions except for the last stage

if __name__ == "__main__":
    pytest.main([__file__])