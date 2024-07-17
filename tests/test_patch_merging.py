import pytest
import torch
import torch.nn as nn
from model.modules import PatchMerging2D, PatchMerging3D


@pytest.mark.parametrize(
    "PatchMerging, dims", [(PatchMerging2D, 2), (PatchMerging3D, 3)]
)
class TestPatchMerging:

    @pytest.fixture
    def merge_params(self, dims):
        return {
            "in_channels": 96,
            "out_channels": 192,
            "merge_size": (2,) * dims,
            "norm_layer": nn.LayerNorm,
        }

    def test_initialization(self, PatchMerging, dims, merge_params):
        merge = PatchMerging(**merge_params)
        assert isinstance(merge, (PatchMerging2D, PatchMerging3D))
        assert merge.in_channels == merge_params["in_channels"]
        assert merge.out_channels == merge_params["out_channels"]
        assert merge.merge_size == merge_params["merge_size"]

    def test_output_shape(self, PatchMerging, dims, merge_params):
        merge = PatchMerging(**merge_params)

        # Create input tensor
        batch_size = 2
        if dims == 2:
            x = torch.randn(batch_size, merge_params["in_channels"], 56, 56)
        else:
            x = torch.randn(batch_size, merge_params["in_channels"], 56, 56, 56)

        output = merge(x)

        expected_shape = [batch_size, merge_params["out_channels"]] + [56 // 2] * dims
        assert list(output.shape) == expected_shape
        assert not torch.any(torch.isnan(output))

    def test_normalization(self, PatchMerging, dims, merge_params):
        merge = PatchMerging(**merge_params)

        # Create input tensor
        batch_size = 2
        if dims == 2:
            x = torch.randn(batch_size, merge_params["in_channels"], 56, 56)
        else:
            x = torch.randn(batch_size, merge_params["in_channels"], 56, 56, 56)

        output = merge(x)
        assert not torch.any(torch.isnan(output))

        # Check if output is normalized
        assert torch.allclose(output.mean(), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(output.std(), torch.tensor(1.0), atol=1e-1)

    def test_invalid_input_size(self, PatchMerging, dims, merge_params):
        merge = PatchMerging(**merge_params)

        # Create input tensor with invalid size
        batch_size = 2
        if dims == 2:
            x = torch.randn(batch_size, merge_params["in_channels"], 57, 57)
        else:
            x = torch.randn(batch_size, merge_params["in_channels"], 57, 57, 57)

        with pytest.raises(AssertionError):
            merge(x)

    def test_repr(self, PatchMerging, dims, merge_params):
        merge = PatchMerging(**merge_params)
        repr_str = repr(merge)
        assert str(merge_params["in_channels"]) in repr_str
        assert str(merge_params["out_channels"]) in repr_str
        assert str(merge_params["merge_size"]) in repr_str
