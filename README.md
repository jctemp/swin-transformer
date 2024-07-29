# Swin Transformer

This project implements the Swin Transformer (Shifted Window Transformer), a hierarchical vision transformer that uses shifted windows to process images or volumetric data.

## Overview

The Swin Transformer is a powerful and flexible architecture for various vision tasks. This implementation provides both 2D and 3D versions, making it suitable for a wide range of applications, from image processing to video analysis and 3D medical imaging. It is based on the official implementation with a personal spin.

## Features

- 2D and 3D Patch Embedding
- 2D and 3D Patch Merging
- 2D and 3D Window Shifting
- 2D and 3D Attention mechanism with relative position encoding
- Configurable architecture (number of stages, embedding dimensions, etc.)
- PyTorch implementation with TorchScript support

## Requirements

- Python 3.11
- PyTorch 2.3
- einops 0.8
- timm 1.0

## Installation

Clone this repository and install the required packages:

```bash
git clone https://github.com/jctemp/swin-transformer.git
cd swin-transformer
pip install -r requirements.txt
```

For NixOS users:

```bash
git clone https://github.com/jctemp/swin-transformer.git
cd swin-transformer
nix develop
. run setup
```

## Usage

Here's a basic example of how to use the 2D and 3D Swin Transformer:

### 2D Swin Transformer

```python
import torch
from swin_transformer_2d import SwinTransformer2D, SwinTransformerConfig2D

config = SwinTransformerConfig2D(
    input_size=(224, 224),
    in_channels=3,
    embed_dim=96,
    num_blocks=[2, 2, 6, 2],
    patch_window_size=[(4, 4), (2, 2), (2, 2), (2, 2)],
    block_window_size=[(7, 7), (7, 7), (7, 7), (7, 7)],
    num_heads=[3, 6, 12, 24],
)

model = SwinTransformer2D(config)
x = torch.randn(1, 3, 224, 224)
output = model(x)
```

### 3D Swin Transformer

```python
import torch
from swin_transformer_3d import SwinTransformer3D, SwinTransformerConfig3D

config = SwinTransformerConfig3D(
    input_size=(32, 224, 224),
    in_channels=3,
    embed_dim=96,
    num_blocks=[2, 2, 6, 2],
    patch_window_size=[(2, 4, 4), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
    block_window_size=[(2, 7, 7), (2, 7, 7), (2, 7, 7), (2, 7, 7)],
    num_heads=[3, 6, 12, 24],
)

model = SwinTransformer3D(config)
x = torch.randn(1, 3, 32, 224, 224)
output = model(x)
```

## Testing

To run the tests:

```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This implementation is based on the original Swin Transformer paper:
"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" by Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo.
The official implementation can be found at the following link: https://github.com/microsoft/Swin-Transformer

## Contact

For any questions or feedback, please open an issue on GitHub or contact me.
