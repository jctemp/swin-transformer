import torch
from . import PatchMerging


def test_patch_merging():
    batch_size, channels, height, width = 1, 96, 56, 56
    merge_size = (2, 2)

    x = torch.randn(batch_size, channels, height, width)
    model = PatchMerging(in_channels=channels, merge_size=merge_size)

    output = model(x)

    expected_shape = (
        batch_size,
        channels * 2,
        height // merge_size[0],
        width // merge_size[1],
    )
    assert (
        output.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {output.shape}"
