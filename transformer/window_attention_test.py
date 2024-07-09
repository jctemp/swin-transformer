import torch
from . import WindowMultiHeadAttention

def test_window_attention():
    batch_size, seq_len, embed_dim = 1, 7 * 7, 96
    num_heads = 4
    window_size = [7, 7]
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    model = WindowMultiHeadAttention(embed_dim, num_heads, window_size)
    
    output = model(x, x, x)
    
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"