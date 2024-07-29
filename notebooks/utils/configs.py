model_embedding_none = {
    "input_size": (32, 32),
    "in_channels": 3,
    "embed_dim": 32,
    "num_blocks": [2, 4, 4, 2],
    "patch_window_size": [(2, 2)] * 4,
    "block_window_size": ([(4, 4)] * 3) + [(2, 2)],
    "num_heads": [2, 4, 8, 16],
    "drop_path": 0.1,
    "rpe_mode": "none",
}

model_embedding_bias = {
    "input_size": (32, 32),
    "in_channels": 3,
    "embed_dim": 32,
    "num_blocks": [2, 4, 4, 2],
    "patch_window_size": [(2, 2)] * 4,
    "block_window_size": ([(4, 4)] * 3) + [(2, 2)],
    "num_heads": [2, 4, 8, 16],
    "drop_path": 0.1,
    "rpe_mode": "bias",
}
model_embedding_context = {
    "input_size": (32, 32),
    "in_channels": 3,
    "embed_dim": 32,
    "num_blocks": [2, 4, 4, 2],
    "patch_window_size": [(2, 2)] * 4,
    "block_window_size": ([(4, 4)] * 3) + [(2, 2)],
    "num_heads": [2, 4, 8, 16],
    "drop_path": 0.1,
    "rpe_mode": "context",
}
model_merge_concatenation = {
    "input_size": (32, 32),
    "in_channels": 3,
    "embed_dim": 32,
    "num_blocks": [2, 4, 4, 2],
    "patch_window_size": [(2, 2)] * 4,
    "block_window_size": ([(4, 4)] * 3) + [(2, 2)],
    "num_heads": [2, 4, 8, 16],
    "drop_path": 0.1,
    "patch_mode": ["concatenate"] * 4,
}
model_merge_convolution = {
    "input_size": (32, 32),
    "in_channels": 3,
    "embed_dim": 32,
    "num_blocks": [2, 4, 4, 2],
    "patch_window_size": [(2, 2)] * 4,
    "block_window_size": ([(4, 4)] * 3) + [(2, 2)],
    "num_heads": [2, 4, 8, 16],
    "drop_path": 0.1,
    "patch_mode": ["convolution"] * 4,
}
model_odd_windows = {
    "input_size": (32, 32),
    "in_channels": 3,
    "embed_dim": 32,
    "num_blocks": [2, 4, 4, 2],
    "patch_window_size": [(2, 2)] * 4,
    "block_window_size": [(3, 3)] * 4,
    "num_heads": [2, 4, 8, 16],
    "drop_path": 0.1,
    "rpe_mode": "bias",
}

model_reference = {
    "num_classes": 10,
    "img_size": 32,
    "in_chans": 3,
    "embed_dim": 32,
    "depths": [2, 4, 4, 2],
    "patch_size": 2,
    "window_size": 4,
    "num_heads": [2, 4, 8, 16],
}

MODEL_CONFIGS = {
    "odd_windows": model_odd_windows,
    "embedding_none": model_embedding_none,
    "embedding_bias": model_embedding_bias,
    "embedding_context": model_embedding_context,
    "merge_concatenate": model_merge_concatenation,
    "merge_convolution": model_merge_convolution,
    "reference": model_reference,
}
