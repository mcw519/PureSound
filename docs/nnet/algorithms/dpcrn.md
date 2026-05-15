# puresound.nnet.dpcrn

Dual-Path Conditional RNN (DPCRN) — a U-Net architecture with dual-path LSTM processing and optional speaker or content conditioning via FiLM modulation.

## Class: `DPRNNblock2D`

> **Deprecated**: Use `DPCRN` directly. This class is kept for backwards compatibility.

A dual-path RNN block with bidirectional intra-chunk LSTM and unidirectional inter-chunk LSTM, with optional FiLM conditioning.

### Constructor

```python
DPRNNblock2D(
    in_channel: int,
    hid_channel: int,
    embed_dim: Optional[int] = None,
    bidirectional: bool = True,
)
```

**Parameters:**
- `in_channel` – Input feature dimension
- `hid_channel` – LSTM hidden dimension
- `embed_dim` – If provided, enables FiLM conditioning with this embedding dimension
- `bidirectional` – If `True`, intra-chunk LSTM is bidirectional

### `forward(x: Tensor, embed: Optional[Tensor] = None) -> Tensor`

**Parameters:**
- `x` – Input tensor `[batch, channel, freq, time]`
- `embed` – Optional conditioning embedding `[batch, embed_dim]`

**Returns:** Processed tensor `[batch, channel, freq, time]`.

---

## Class: `DPCRN`

Full Dual-Path Conditional RNN model extending the `Unet` architecture.

### Architecture

```
Input Spectrum
  └─ U-Net Encoder (CNN downsampling with skip connections)
       └─ DPCRN Bottleneck (stack of DPRNNblock2D with conditioning)
            └─ U-Net Decoder (CNN upsampling + skip connections)
                 └─ Output Spectrum (mask or enhanced features)
```

### Constructor

```python
DPCRN(
    # U-Net parameters
    in_channel: int,
    out_channel: int,
    encoder_kernel: List[int],
    encoder_stride: List[int],
    encoder_channel: List[int],
    # DPCRN bottleneck parameters
    num_blocks: int,
    hid_channel: int,
    embed_dim: Optional[int] = None,
    bidirectional: bool = True,
)
```

**Parameters:**
- `in_channel` – Input spectrum channels (e.g., 2 for stacked real/imag)
- `out_channel` – Output channels
- `encoder_kernel/stride/channel` – U-Net encoder configuration
- `num_blocks` – Number of DPCRN blocks in the bottleneck
- `hid_channel` – LSTM hidden dimension per block
- `embed_dim` – Speaker/content embedding dimension for FiLM conditioning (optional)
- `bidirectional` – If `True`, intra-chunk LSTM is bidirectional (non-causal)

### `forward(x: Tensor, embed: Optional[Tensor] = None) -> Tensor`

**Parameters:**
- `x` – Input spectrum `[batch, channel, freq, time]`
- `embed` – Optional conditioning embedding `[batch, embed_dim]`

**Returns:** Output tensor of the same shape as input.

## Example

```python
from puresound.nnet.dpcrn import DPCRN

model = DPCRN(
    in_channel=2, out_channel=2,
    encoder_kernel=[3, 3, 3],
    encoder_stride=[2, 2, 1],
    encoder_channel=[16, 32, 64],
    num_blocks=4,
    hid_channel=64,
    embed_dim=256,   # Enable conditioning with 256-dim speaker embedding
)

# Inference with speaker embedding
enhanced = model(noisy_spec, embed=spk_embedding)
```
