# puresound.nnet.lobe.cnn

CNN building blocks for 1D temporal convolution.

## Class: `DepthwiseSeparableConv1d`

Depthwise Separable 1D Convolution with residual skip connection, following the Conv-TasNet pattern.

### Architecture

```
Input
  ├─ Depthwise Conv1d (groups = channels, dilation supported)
  │     └─ Pointwise Conv1d (1×1 projection)
  │           └─ Normalization + Activation
  └─ Skip connection (identity or 1×1 projection if dimensions differ)
Output = residual_path + skip_connection
```

### Constructor

```python
DepthwiseSeparableConv1d(
    in_channel: int,
    hid_channel: int,
    kernel_size: int,
    dilation: int = 1,
    causal: bool = False,
    norm: str = "global_layer_norm",
    activation: str = "prelu",
)
```

**Parameters:**
- `in_channel` – Number of input and output channels (residual path preserves dimension)
- `hid_channel` – Intermediate pointwise projection dimension
- `kernel_size` – Depthwise convolution kernel size
- `dilation` – Dilation factor (exponentially growing in stacked TCNs)
- `causal` – If `True`, uses causal padding (no future look-ahead)
- `norm` – Normalization type (e.g., `"global_layer_norm"`, `"channel_layer_norm"`)
- `activation` – Activation function name

### `forward(x: Tensor) -> Tensor`

**Parameters:**
- `x` – Input tensor `[batch, in_channel, T]`

**Returns:** Output tensor `[batch, in_channel, T]` (same shape as input).

## Example

```python
from puresound.nnet.lobe.cnn import DepthwiseSeparableConv1d

conv = DepthwiseSeparableConv1d(
    in_channel=256,
    hid_channel=512,
    kernel_size=3,
    dilation=4,
    causal=False,
)

out = conv(features)  # [B, 256, T]
```
