import pytest
import torch
import torch.nn as nn
from model import SwinTransformer2D, SwinTransformer3D

class TestSwinTransformer2D:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.default_params = {
            "input_resolution": (224, 224),
            "in_channels": 3,
            "out_channels": [96, 192, 384, 768],
            "depths": [2, 2, 18, 2],
            "patch_size": (4, 4),
            "merge_size": [(2, 2)] * 3,
            "window_size": [(7, 7)] * 4,
            "num_heads": [3, 6, 12, 24],
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "drop": 0.1,
            "drop_attn": 0.1,
            "drop_path": 0.1,
            "norm_layer": nn.LayerNorm,
            "ape": False,
            "rpe": True,
            "patch_norm": True,
        }

    def test_output_shape(self):
        model = SwinTransformer2D(**self.default_params)
        x = torch.randn(1, 3, 224, 224)
        outputs = model(x)
        
        expected_shapes = [
            (1, 96, 56, 56),
            (1, 192, 28, 28),
            (1, 384, 14, 14),
            (1, 768, 7, 7),
        ]
        
        assert len(outputs) == len(expected_shapes), f"Number of outputs doesn't match expected, {len(outputs)} != {len(expected_shapes)}"
        assert not torch.any(torch.isnan(outputs[-1]))
        
        for output, expected_shape in zip(outputs, expected_shapes):
            assert output.shape == expected_shape, f"Output shape {output.shape} doesn't match expected {expected_shape}"

    def test_invalid_params(self):
        invalid_params = self.default_params.copy()
        invalid_params["out_channels"] = [96, 192, 384]  # Missing one output channel
        
        with pytest.raises(AssertionError):
            SwinTransformer2D(**invalid_params)

    @pytest.mark.parametrize("input_size", [(224, 224), (256, 256), (384, 384)])
    def test_different_input_sizes(self, input_size):
        model_params = self.default_params.copy()
        model_params["input_resolution"] = input_size
        model = SwinTransformer2D(**model_params)
        
        x = torch.randn(1, 3, *input_size)
        outputs = model(x)
        
        assert len(outputs) == 4, "Number of outputs should be 4"
        assert outputs[-1].shape[2] == input_size[0] // 32, "Final output height is incorrect"
        assert outputs[-1].shape[3] == input_size[1] // 32, "Final output width is incorrect"

    def test_no_patch_norm(self):
        model_params = self.default_params.copy()
        model_params["patch_norm"] = False
        model = SwinTransformer2D(**model_params)
        
        x = torch.randn(1, 3, 224, 224)
        outputs = model(x)
        
        assert len(outputs) == 4, "Number of outputs should be 4"

    @pytest.mark.parametrize("ape", [True, False])
    def test_ape(self, ape):
        model_params = self.default_params.copy()
        model_params["ape"] = ape
        model = SwinTransformer2D(**model_params)
        
        x = torch.randn(1, 3, 224, 224)
        outputs = model(x)
        
        assert len(outputs) == 4, "Number of outputs should be 4"

    def test_forward_pass(self):
        model = SwinTransformer2D(**self.default_params)
        x = torch.randn(1, 3, 224, 224)
        
        try:
            outputs = model(x)
        except Exception as e:
            pytest.fail(f"Forward pass failed with exception: {e}")
        
        assert isinstance(outputs, list), "Output should be a list"
        assert all(isinstance(output, torch.Tensor) for output in outputs), "All outputs should be torch.Tensor"

    def test_block_attn_mask(self):
        self.default_params["window_size"] = [(5, 5)] * 4
        model = SwinTransformer2D(**self.default_params)
        x = torch.randn(1, 3, 224, 224)
        outputs = model(x)
        
        assert len(outputs) == 4, "Number of outputs should be 4"
    
class TestSwinTransformer3D:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.default_params = {
            "input_resolution": (96, 96, 96),
            "in_channels": 1,
            "out_channels": [48, 96, 192, 384],
            "depths": [2, 2, 6, 2],
            "patch_size": (2, 2, 2),
            "merge_size": [(2, 2, 2)] * 3,
            "window_size": [(7, 7, 7)] * 4,
            "num_heads": [3, 6, 12, 24],
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "drop": 0.1,
            "drop_attn": 0.1,
            "drop_path": 0.1,
            "norm_layer": nn.LayerNorm,
            "ape": False,
            "rpe": True,
            "patch_norm": True,
        }

    def test_output_shape(self):
        model = SwinTransformer3D(**self.default_params)
        x = torch.randn(1, 1, 96, 96, 96)
        outputs = model(x)
        
        expected_shapes = [
            (1, 48, 48, 48, 48),
            (1, 96, 24, 24, 24),
            (1, 192, 12, 12, 12),
            (1, 384, 6, 6, 6),
        ]
        
        assert len(outputs) == len(expected_shapes), f"Number of outputs doesn't match expected, {len(outputs)} != {len(expected_shapes)}"
        
        for output, expected_shape in zip(outputs, expected_shapes):
            assert output.shape == expected_shape, f"Output shape {output.shape} doesn't match expected {expected_shape}"

    def test_invalid_params(self):
        invalid_params = self.default_params.copy()
        invalid_params["out_channels"] = [48, 96, 192]  # Missing one output channel
        
        with pytest.raises(AssertionError):
            SwinTransformer3D(**invalid_params)

    @pytest.mark.parametrize("input_size", [(96, 96, 96), (128, 128, 128), (64, 64, 64)])
    def test_different_input_sizes(self, input_size):
        model_params = self.default_params.copy()
        model_params["input_resolution"] = input_size
        model = SwinTransformer3D(**model_params)
        
        x = torch.randn(1, 1, *input_size)
        outputs = model(x)
        
        assert len(outputs) == 4, "Number of outputs should be 4"
        assert outputs[-1].shape[2] == input_size[0] // 16, "Final output depth is incorrect"
        assert outputs[-1].shape[3] == input_size[1] // 16, "Final output height is incorrect"
        assert outputs[-1].shape[4] == input_size[2] // 16, "Final output width is incorrect"

    def test_no_patch_norm(self):
        model_params = self.default_params.copy()
        model_params["patch_norm"] = False
        model = SwinTransformer3D(**model_params)
        
        x = torch.randn(1, 1, 96, 96, 96)
        outputs = model(x)
        
        assert len(outputs) == 4, "Number of outputs should be 4"

    @pytest.mark.parametrize("ape", [True, False])
    def test_ape(self, ape):
        model_params = self.default_params.copy()
        model_params["ape"] = ape
        model = SwinTransformer3D(**model_params)
        
        x = torch.randn(1, 1, 96, 96, 96)
        outputs = model(x)
        
        assert len(outputs) == 4, "Number of outputs should be 4"

    def test_forward_pass(self):
        model = SwinTransformer3D(**self.default_params)
        x = torch.randn(1, 1, 96, 96, 96)
        
        try:
            outputs = model(x)
        except Exception as e:
            pytest.fail(f"Forward pass failed with exception: {e}")
        
        assert isinstance(outputs, list), "Output should be a list"
        assert all(isinstance(output, torch.Tensor) for output in outputs), "All outputs should be torch.Tensor"

    @pytest.mark.parametrize("in_channels", [1, 3])
    def test_different_input_channels(self, in_channels):
        model_params = self.default_params.copy()
        model_params["in_channels"] = in_channels
        model = SwinTransformer3D(**model_params)
        
        x = torch.randn(1, in_channels, 96, 96, 96)
        outputs = model(x)
        
        assert len(outputs) == 4, "Number of outputs should be 4"
        assert outputs[0].shape[1] == self.default_params["out_channels"][0], "First output channel count is incorrect"

    def test_block_attn_mask(self):
        self.default_params["window_size"] = [(5, 5, 5)] * 4
        model = SwinTransformer3D(**self.default_params)
        x = torch.randn(1, 1, 96, 96, 96)
        outputs = model(x)
        
        assert len(outputs) == 4, "Number of outputs should be 4"