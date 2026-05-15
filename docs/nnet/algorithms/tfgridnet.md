# puresound.nnet.tfgridnet

TF-GridNet time-frequency context module for speech enhancement.

**Reference:** Wang et al., "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation," IEEE/ACM TASLP, 2023.

## Class: `ContextFeature`

Unfolds time-frequency frames to create a local context window representation, enabling the model to process each TF bin with access to neighboring frames.

### Constructor

```python
ContextFeature(
    context_size: int,
    stride: int = 1,
)
```

**Parameters:**
- `context_size` – Number of frames to include in each local context window (must be odd for symmetric context)
- `stride` – Stride for the unfolding operation along the time axis (default: 1 = dense)

### `forward(x: Tensor) -> Tensor`

Unfolds time-frequency features to include local temporal context.

**Parameters:**
- `x` – Input spectrum `[batch, channel, freq, time]`

**Returns:** Context-expanded tensor `[batch, channel * context_size, freq, time]`.

**Notes:**
- Zero-padding is applied at the temporal boundaries to maintain sequence length
- The output can be used as input to frequency-axis or time-axis processing modules

## Use in TF-GridNet Architecture

`ContextFeature` is typically placed before an intra-frame processing module (e.g., LSTM over frequency) to provide each frame with information from neighboring time steps:

```
Input [B, C, F, T]
  └─ ContextFeature(context_size=3)   → [B, 3C, F, T]
       └─ Reshape to [B*T, 3C, F]
            └─ Frequency-axis LSTM    → [B*T, hid, F]
                 └─ Reshape back to [B, hid, F, T]
```

## Class: `TFGridNet`

> **Note:** The full TF-GridNet model (exported from `puresound.nnet` as `TFGridNet`) assembles `ContextFeature` with interleaved time-axis and frequency-axis processing layers, with a U-Net-style encoder-decoder around it.

## Example

```python
from puresound.nnet.tfgridnet import ContextFeature

ctx = ContextFeature(context_size=3)
x_ctx = ctx(spec_features)  # [B, 3*C, F, T]
```
