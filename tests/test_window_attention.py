import pytest
import torch
from model.modules import WindowMultiHeadAttention2D, WindowMultiHeadAttention3D


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
            "shift": False,
        }

    def test_initialization(self, Attention, dims, attention_params):
        attn = Attention(**attention_params)
        assert isinstance(attn, Attention)
        assert attn.out_channels == attention_params["in_channels"]
        assert attn.num_heads == attention_params["num_heads"]
        assert len(attn.window_size) == dims

    @pytest.mark.parametrize("shift", [True, False])
    def test_forward(self, Attention, dims, attention_params, shift):
        attention_params["shift"] = shift
        attn = Attention(**attention_params)

        batch_size = 2
        if dims == 2:
            x = torch.randn(batch_size, attention_params["in_channels"], 56, 56)
        else:
            x = torch.randn(batch_size, attention_params["in_channels"], 56, 56, 56)

        output, context = attn(x, x, x)
        assert not torch.isnan(output).any()
        assert not torch.isnan(context).any()
        assert output.shape == x.shape

    def test_cycle_shift(self, Attention, dims, attention_params):
        attention_params["shift"] = True
        attn = Attention(**attention_params)

        batch_size = 2
        if dims == 2:
            x = torch.randn(batch_size, attention_params["in_channels"], 56, 56)
        else:
            x = torch.randn(batch_size, attention_params["in_channels"], 56, 56, 56)

        shifted = attn.cycle_shift(x)
        assert shifted.shape == x.shape
        assert not torch.allclose(x, shifted)

        reversed_shift = attn.cycle_shift(shifted, reverse=True)
        assert reversed_shift.shape == x.shape

        print(x[0, 0, 0, :10])
        print(shifted[0, 0, 0, :10])
        print(reversed_shift[0, 0, 0, :10])

        assert torch.all(torch.eq(x, reversed_shift))

    def test_rpe(self, Attention, dims, attention_params):
        attn = Attention(**attention_params)

        score = torch.randn(2, attention_params["num_heads"], 49, 49)
        if dims == 3:
            score = torch.randn(2, attention_params["num_heads"], 343, 343)

        rpe_score = attn.rpe(score)
        assert rpe_score.shape == score.shape
        assert not torch.allclose(score, rpe_score)

    def test_to_seq_and_to_spatial(self, Attention, dims, attention_params):
        attn = Attention(**attention_params)

        batch_size = 2
        if dims == 2:
            x = torch.randn(batch_size, attention_params["in_channels"], 56, 56)
            spatial_dims = [56, 56]
        else:
            x = torch.randn(batch_size, attention_params["in_channels"], 56, 56, 56)
            spatial_dims = [56, 56, 56]

        seq = attn.to_seq(x, spatial_dims)
        assert seq.dim() == 3  # (batch * windows, tokens, channels)

        spatial = attn.to_spatial(seq, spatial_dims)
        assert spatial.shape == x.shape
        assert torch.allclose(x, spatial)
