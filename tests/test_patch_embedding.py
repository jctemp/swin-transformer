import pytest
import torch
import torch.nn as nn
from model.modules import PatchEmbedding2D, PatchEmbedding3D

@pytest.mark.parametrize("PatchEmbedding, dims", [
    (PatchEmbedding2D, 2),
    (PatchEmbedding3D, 3)
])
class TestPatchEmbedding:

    @pytest.fixture
    def embed_params(self, dims):
        return {
            'in_channels': 3,
            'out_channels': 96,
            'patch_size': (4,) * dims,
            'norm_layer': nn.LayerNorm,
            'ape': False,
            'ape_freq_base': 10000.0
        }

    def test_initialization(self, PatchEmbedding, dims, embed_params):
        embed = PatchEmbedding(**embed_params)
        assert isinstance(embed, (PatchEmbedding2D, PatchEmbedding3D))
        assert embed.in_channels == embed_params['in_channels']
        assert embed.out_channels == embed_params['out_channels']
        assert embed.patch_size == embed_params['patch_size']
        assert embed.use_ape == embed_params['ape']
        assert embed.ape_freq_base == embed_params['ape_freq_base']

    def test_output_shape(self, PatchEmbedding, dims, embed_params):
        embed = PatchEmbedding(**embed_params)
        
        # Create input tensor
        batch_size = 2
        if dims == 2:
            x = torch.randn(batch_size, embed_params['in_channels'], 224, 224)
        else:
            x = torch.randn(batch_size, embed_params['in_channels'], 224, 224, 224)
        
        output = embed(x)
        
        expected_shape = [batch_size, embed_params['out_channels']] + [224 // 4] * dims
        assert list(output.shape) == expected_shape
        assert not torch.any(torch.isnan(output))

    @pytest.mark.parametrize("ape", [True, False])
    def test_positional_encoding(self, PatchEmbedding, dims, embed_params, ape):
        embed_params['ape'] = ape
        embed = PatchEmbedding(**embed_params)
        
        # Create input tensor
        batch_size = 2
        if dims == 2:
            x = torch.randn(batch_size, embed_params['in_channels'], 224, 224)
        else:
            x = torch.randn(batch_size, embed_params['in_channels'], 224, 224, 224)
        
        output = embed(x)
        assert not torch.any(torch.isnan(output))
        
        if ape:
            assert embed.pe is not None
            assert embed.pe.shape == output.shape[1:]
        else:
            assert embed.pe is None

    def test_normalization(self, PatchEmbedding, dims, embed_params):
        embed = PatchEmbedding(**embed_params)
        
        # Create input tensor
        batch_size = 2
        if dims == 2:
            x = torch.randn(batch_size, embed_params['in_channels'], 224, 224)
        else:
            x = torch.randn(batch_size, embed_params['in_channels'], 224, 224, 224)
        
        output = embed(x)
        assert not torch.any(torch.isnan(output))
        
        # Check if output is normalized
        assert torch.allclose(output.mean(), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(output.std(), torch.tensor(1.0), atol=1e-1)

    def test_invalid_input_size(self, PatchEmbedding, dims, embed_params):
        embed = PatchEmbedding(**embed_params)
        
        # Create input tensor with invalid size
        batch_size = 2
        if dims == 2:
            x = torch.randn(batch_size, embed_params['in_channels'], 225, 225)
        else:
            x = torch.randn(batch_size, embed_params['in_channels'], 225, 225, 225)
        
        with pytest.raises(AssertionError):
            embed(x)

    def test_repr(self, PatchEmbedding, dims, embed_params):
        embed = PatchEmbedding(**embed_params)
        repr_str = repr(embed)
        assert str(embed_params['in_channels']) in repr_str
        assert str(embed_params['out_channels']) in repr_str
        assert str(embed_params['patch_size']) in repr_str
        assert str(embed_params['ape']) in repr_str
        