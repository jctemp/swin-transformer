import pytest
import torch
from model.modules import (
    SwinTransformerBlock,
    SwinTransformerBlock2D,
    SwinTransformerBlock3D,
)


@pytest.mark.parametrize(
    "Block, dims", [(SwinTransformerBlock2D, 2), (SwinTransformerBlock3D, 3)]
)
class TestSwinTransformerBlock:

    @pytest.fixture
    def block_params(self, dims):
        input_resolution = (56, 56) if dims == 2 else (56, 56, 56)
        return {
            "input_resolution": input_resolution,
            "in_channels": 96,
            "num_heads": 4,
            "window_size": [7] * dims,
            "mlp_ratio": 2.0,
            "qkv_bias": True,
            "drop": 0.1,
            "drop_attn": 0.1,
            "drop_proj": 0.1,
            "drop_path": 0.1,
            "rpe": True,
            "shift": False,
        }

    def test_initialization(self, Block, dims, block_params):
        block = Block(**block_params)
        assert isinstance(block, SwinTransformerBlock)
        assert block.dims == dims
        assert len(block.window_size) == dims

    @pytest.mark.parametrize("shift", [True, False])
    def test_forward(self, Block, dims, block_params, shift):
        block_params["shift"] = shift
        block = Block(**block_params)

        # Create input tensor
        batch_size = 2
        if dims == 2:
            x = torch.randn(batch_size, block_params["in_channels"], 56, 56)
        else:
            x = torch.randn(batch_size, block_params["in_channels"], 56, 56, 56)

        output = block(x)

        # Check output shape
        assert output.shape == x.shape

        # Check that output is different from input
        assert not torch.allclose(output, x)

    def test_attention_map(self, Block, dims, block_params):
        block = Block(**block_params)

        # Create input tensor
        batch_size = 2
        if dims == 2:
            x = torch.randn(batch_size, block_params["in_channels"], 56, 56)
        else:
            x = torch.randn(batch_size, block_params["in_channels"], 56, 56, 56)

        _ = block(x)

        # Check that attention map is stored
        assert block.attn_map is not None

    @pytest.mark.parametrize("shift", [True, False])
    def test_shifting(self, Block, dims, block_params, shift):
        block_params['shift'] = shift
        block1 = Block(**block_params)
        block2 = Block(**block_params)
        
        batch_size = 2
        if dims == 2:
            x = torch.randn(batch_size, block_params['in_channels'], 56, 56)
        else:
            x = torch.randn(batch_size, block_params['in_channels'], 56, 56, 56)
        
        output1 = block1(x)
        output2 = block2(x)

        assert not torch.allclose(output1, output2)

    def test_attn_mask(self, Block, dims, block_params):
        block_params['shift'] = True
        block = Block(**block_params)

        # Check that attn_mask is created when shift is True
        assert block.attn_mask is not None

        # Check the shape of attn_mask
        expected_mask_size = 49 if dims == 2 else 343  # 7^2 for 2D, 7^3 for 3D
        assert block.attn_mask.shape[-2:] == (expected_mask_size, expected_mask_size)

        # Check that attn_mask contains only 0.0 and -100.0
        unique_values = torch.unique(block.attn_mask)
        assert set(unique_values.tolist()) == {0.0, -100.0}