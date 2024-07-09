import torch
from . import PatchEmbedding

def test_patch_embedding():
    batch_size, channels, height, width = 1, 3, 224, 224
    patch_size = (16, 16)
    embed_dim = 768
    
    x = torch.randn(batch_size, channels, height, width)
    model = PatchEmbedding(in_channels=channels, patch_size=patch_size, embed_dim=embed_dim, ape=True)
    
    output = model(x)
    
    expected_shape = (batch_size, embed_dim, height // patch_size[0], width // patch_size[1])
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"