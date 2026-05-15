# puresound.nnet.unet

U-Net architecture for time-frequency domain speech processing with encoder-decoder structure and skip connections.

## Class: `Unet`

A configurable 2D U-Net operating on time-frequency spectral feature maps. Used as the backbone for DPCRN, DPARN, and other architectures.

### Architecture

```
Input [B, C_in, F, T]
  └─ Encoder Stage 1: Conv2d → Norm → Activation  ─────────────────────────────┐ skip_1
       └─ Encoder Stage 2: Conv2d → Norm → Activation  ──────────────────────┐  │ skip_2
            └─ ...                                                             │  │
                 └─ Bottleneck (custom blocks or identity)                    │  │
            └─ Decoder Stage N: ConvTranspose2d → Norm → Activation ← skip_N ┘  │
       └─ Decoder Stage 2: ConvTranspose2d → Norm → Activation ← skip_2 ─────┘
  └─ Decoder Stage 1: ConvTranspose2d → Norm → Activation ← skip_1
       └─ Output [B, C_out, F, T]
```

### Constructor

```python
Unet(
    in_channel: int,
    out_channel: int,
    encoder_kernel: List[Tuple[int, int]],
    encoder_stride: List[Tuple[int, int]],
    encoder_channel: List[int],
    encoder_padding: Optional[List] = None,
    activation: str = "prelu",
    norm: str = "layer_norm_2d",
    multi_output: bool = False,
)
```

**Parameters:**
- `in_channel` – Number of input feature channels (e.g., 2 for stacked real/imag)
- `out_channel` – Number of output channels
- `encoder_kernel` – List of `(freq_kernel, time_kernel)` pairs for each encoder stage
- `encoder_stride` – List of `(freq_stride, time_stride)` pairs for each encoder stage
- `encoder_channel` – List of output channel counts for each encoder stage
- `encoder_padding` – Optional list of padding tuples (auto-computed if `None`)
- `activation` – Activation function name (passed to `get_activation()`)
- `norm` – Normalization type (passed to `get_norm()`)
- `multi_output` – If `True`, returns output at each decoder stage (for multi-scale supervision)

### `forward(x: Tensor, cond: Optional[List[Tensor]] = None) -> Union[Tensor, List[Tensor]]`

**Parameters:**
- `x` – Input tensor `[batch, in_channel, freq, time]`
- `cond` – Optional list of conditioning tensors per decoder stage

**Returns:**
- Single output tensor `[batch, out_channel, freq, time]` (if `multi_output=False`)
- List of output tensors at each decoder stage (if `multi_output=True`)

## Example

```python
from puresound.nnet.unet import Unet

unet = Unet(
    in_channel=2,
    out_channel=2,
    encoder_kernel=[(3, 3), (3, 3), (3, 3)],
    encoder_stride=[(2, 1), (2, 1), (2, 1)],
    encoder_channel=[16, 32, 64],
    activation="prelu",
    norm="layer_norm_2d",
)

output = unet(input_spec)  # [B, 2, F, T]
```
