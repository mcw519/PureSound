# puresound.nnet.dparn

Dual-Path Attention RNN (DPARN) — a U-Net architecture using attention-based intra-chunk processing and LSTM-based inter-chunk processing.

**Reference:** Inspired by DPARN architecture for speech enhancement in the time-frequency domain.

## Class: `DPARNblock2D`

A single Dual-Path Attention RNN processing block operating on 2D time-frequency feature maps.

### Architecture

- **Intra-chunk path**: Multi-head self-attention over the frequency axis (each time frame processed independently)
- **Inter-chunk path**: LSTM over the time axis (captures temporal dependencies)

### Constructor

```python
DPARNblock2D(
    in_channel: int,
    hid_channel: int,
    num_heads: int = 4,
    bidirectional: bool = True,
    causal: bool = False,
)
```

**Parameters:**
- `in_channel` – Number of input feature channels (frequency bins after encoding)
- `hid_channel` – Hidden dimension for LSTM
- `num_heads` – Number of attention heads for intra-chunk self-attention
- `bidirectional` – If `True`, the inter-chunk LSTM is bidirectional (non-causal)
- `causal` – If `True`, applies causal masking to the attention to prevent look-ahead

### `forward(x: Tensor) -> Tensor`

**Parameters:**
- `x` – Input tensor `[batch, channel, freq, time]`

**Returns:** Processed tensor `[batch, channel, freq, time]`.

---

## Class: `DPARN`

Full DPARN model extending the `Unet` architecture. Replaces the U-Net bottleneck with a stack of `DPARNblock2D` modules.

### Constructor

```python
DPARN(
    # U-Net parameters
    in_channel: int,
    out_channel: int,
    encoder_kernel: List[int],
    encoder_stride: List[int],
    encoder_channel: List[int],
    # DPARN bottleneck parameters
    num_blocks: int,
    hid_channel: int,
    num_heads: int = 4,
    bidirectional: bool = True,
    causal: bool = False,
)
```

### `forward(x: Tensor, embed: Optional[Tensor] = None) -> Tensor`

**Parameters:**
- `x` – Input spectrum `[batch, channel, freq, time]`
- `embed` – Optional conditioning embedding (unused in base DPARN)

**Returns:** Enhanced spectrum tensor of the same shape.

## Example

```python
from puresound.nnet.dparn import DPARN

model = DPARN(
    in_channel=2,
    out_channel=2,
    encoder_kernel=[3, 3, 3],
    encoder_stride=[2, 2, 1],
    encoder_channel=[16, 32, 64],
    num_blocks=4,
    hid_channel=64,
    num_heads=4,
    bidirectional=True,
)
```

## Streaming ONNX Runtime

DPARN can be exported as a feature-frame ONNX model for low-latency inference.
The streaming path keeps the offline model unchanged and wraps the DPARN
backbone with explicit frame state:

- CNN temporal caches for the downsampling path
- transpose-convolution pending caches for the upsampling path
- LSTM hidden and cell states for each DPARN block

The ONNX model consumes one complex STFT frame at a time:

```python
enhanced_frame, next_state = forward_frame(noisy_frame, state)
```

`noisy_frame` and `enhanced_frame` have shape `[batch, 257, 2]`. Audio
buffering, Hann STFT, iSTFT, and overlap-add are handled outside ONNX by
`puresound.streaming.StreamingDparnOrt`.

See [DPARN Streaming ONNX Runtime](../../streaming/dparn_onnx.md) for the
supported config, export command, runtime API, and Gradio demo workflow.
