# UResNet50 / AG-UResNet50 PyTorch models

## Encoder
- 2D input, default `1 x 240 x 240`
- ResNet50 encoder
- stem: `7x7 conv -> BN -> ReLU`
- max-pooling after the stem
- residual stage layout consistent with ResNet50: `3, 4, 6, 3` bottleneck blocks

## Decoder
- four decoder stages
- bilinear upsampling to the next skip resolution
- skip concatenation at each level
- decoder block pattern designed to mirror the user's diagram: `3x3 conv -> 1x1 conv -> dropout -> 3x3 conv -> dropout -> BN -> LeakyReLU`
- final head: `3x3 conv -> 3x3 conv -> 3x3 conv -> sigmoid`

## Attention gate
The attention gate is implemented exactly in the computational order shown in the AG diagram:
- `Wg(g)` and `Wx(x)`
- element-wise addition
- `ReLU`
- `1x1 conv`
- `sigmoid`
- resampling to skip resolution
- element-wise multiplication with the skip feature
