import pytest
import torch
from model.modules import (
    WindowMultiHeadAttention,
    WindowMultiHeadAttention2D,
    WindowMultiHeadAttention3D,
)


@pytest.mark.parametrize(
    "Attention, dims",
    [(WindowMultiHeadAttention2D, 2), (WindowMultiHeadAttention3D, 3)],
)
class TestWindowMultiHeadAttention:

    @pytest.fixture
    def attention_params(self, dims):
        return {
            "in_channels": 96,
            "num_heads": 4,
            "window_size": [7] * dims,
            "qkv_bias": True,
            "drop_attn": 0.1,
            "drop_proj": 0.1,
            "rpe": True,
        }

    def test_initialization(self, Attention, dims, attention_params):
        attn = Attention(**attention_params)
        assert isinstance(attn, WindowMultiHeadAttention)
        assert attn.out_channels == attention_params["in_channels"]
        assert attn.num_heads == attention_params["num_heads"]
        assert len(attn.window_size) == dims

    def test_forward(self, Attention, dims, attention_params):
        attn = Attention(**attention_params)

        # Create input tensors
        batch_size = 2
        if dims == 2:
            x = torch.randn(batch_size, attention_params["in_channels"], 56, 56)
        else:
            x = torch.randn(batch_size, attention_params["in_channels"], 56, 56, 56)

        output, context = attn(x, x, x)

        # Check output shape
        assert output.shape == x.shape

        # Check context shape
        expected_context_shape = (
            batch_size * (56 // 7) ** dims,
            attention_params["num_heads"],
            7**dims,
            7**dims,
        )
        assert context.shape == expected_context_shape

    def test_mask(self, Attention, dims, attention_params):
        attn = Attention(**attention_params)

        # Create input tensors
        batch_size = 2
        if dims == 2:
            x = torch.randn(batch_size, attention_params["in_channels"], 56, 56)
            mask = torch.ones(attention_params["num_heads"], 7*7, 7*7)
            mask[:, :24, :24] = 0  # mask out roughly a quarter
        else:
            x = torch.randn(batch_size, attention_params["in_channels"], 56, 56, 56)
            mask = torch.ones(attention_params["num_heads"], 7*7*7, 7*7*7)
            mask[:, :171, :171] = 0  # mask out roughly half

        output, context = attn(x, x, x, mask)

        # Check that masked areas have zero attention
        masked_area = 24 if dims == 2 else 171
        assert torch.all(context[:, :, :masked_area, :masked_area] == 0)

    @pytest.mark.parametrize("rpe", [True, False])
    def test_rpe(self, Attention, dims, attention_params, rpe):
        attention_params["rpe"] = rpe
        attn = Attention(**attention_params)

        # Create input tensors
        batch_size = 2
        if dims == 2:
            x = torch.randn(batch_size, attention_params["in_channels"], 56, 56)
        else:
            x = torch.randn(batch_size, attention_params["in_channels"], 56, 56, 56)

        output_rpe, _ = attn(x, x, x)

        attention_params["rpe"] = not rpe
        attn_no_rpe = Attention(**attention_params)
        output_no_rpe, _ = attn_no_rpe(x, x, x)

        # Outputs should be different when rpe is changed
        assert not torch.allclose(output_rpe, output_no_rpe)

    def test_attention_values(self, Attention, dims, attention_params):
        attention_params["drop_attn"] = 0.0
        attn = Attention(**attention_params)

        # Create input tensors
        batch_size = 2
        if dims == 2:
            x = torch.randn(batch_size, attention_params["in_channels"], 56, 56)
        else:
            x = torch.randn(batch_size, attention_params["in_channels"], 56, 56, 56)

        _, context = attn(x, x, x)

        # Check that attention values sum to 1 for each query, allowing for small numerical errors
        assert torch.allclose(context.sum(dim=-1), torch.ones_like(context.sum(dim=-1)), atol=1e-6, rtol=1e-5)
