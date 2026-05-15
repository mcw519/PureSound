# puresound.nnet.conv_tasnet

Conv-TasNet temporal convolution blocks for time-domain speech separation and enhancement.

**Reference:** Luo & Mesgarani, "Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation," IEEE/ACM TASLP, 2019.

## Class: `TCN`

A single Temporal Convolution Block used in the Conv-TasNet stack.

### Architecture

```
Input
  └─ 1×1 Conv (bottleneck projection)
       └─ Depthwise Separable Conv1d (dilation)
            └─ 1×1 Conv (output projection)
                 └─ Skip connection + residual output
```

### Constructor

```python
TCN(
    in_channel: int,
    hid_channel: int,
    kernel_size: int,
    dilation: int,
    skip_connection: bool = True,
    embed_dim: Optional[int] = None,
)
```

**Parameters:**
- `in_channel` – Input feature dimension
- `hid_channel` – Hidden (bottleneck) dimension
- `kernel_size` – Depthwise conv kernel size
- `dilation` – Dilation factor for the depthwise conv
- `skip_connection` – If `True`, add a skip path output in addition to residual
- `embed_dim` – If provided, concatenates a conditioning embedding before the depthwise conv

### `forward(x: Tensor, embed: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]`

**Parameters:**
- `x` – Input tensor `[batch, in_channel, T]`
- `embed` – Optional conditioning embedding `[batch, embed_dim]`

**Returns:** `(residual_output, skip_output)`

---

## Class: `GatedTCN`

A gated variant of the TCN block with FiLM (Feature-wise Linear Modulation) conditioning.

### Constructor

```python
GatedTCN(
    in_channel: int,
    hid_channel: int,
    kernel_size: int,
    dilation: int,
    embed_dim: Optional[int] = None,
)
```

### `forward(x: Tensor, embed: Optional[Tensor] = None) -> Tensor`

Applies gated temporal convolution with optional FiLM modulation from `embed`.

**Returns:** Output tensor `[batch, in_channel, T]`.

## Conv-TasNet Stack

To build a full Conv-TasNet stack, chain multiple `TCN` blocks with exponentially growing dilation:

```python
from puresound.nnet.conv_tasnet import TCN

stack = nn.ModuleList([
    TCN(in_channel=256, hid_channel=512, kernel_size=3, dilation=2**i)
    for i in range(8)
])
```
