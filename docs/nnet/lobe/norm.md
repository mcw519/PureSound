# puresound.nnet.lobe.norm

Normalization layers with various strategies for speech processing models.

## Base Class: `_LayerNorm`

Extends `nn.Module`. Base layer normalization with learnable gain and bias. Not used directly; subclassed below.

---

## Class: `GlobLN`

**Global Layer Normalization** — normalizes across all spatial and temporal dimensions.

### Constructor

```python
GlobLN(channel_size: int)
```

**Parameters:**
- `channel_size` – Number of channels (feature dimension)

### `forward(x: Tensor) -> Tensor`

Normalizes `x` over the entire `[channel, time]` dimensions.

**Input shape:** `[batch, channel, time]`

---

## Class: `ChanLN`

**Channel-wise Layer Normalization** — normalizes independently per channel (across time only).

### Constructor

```python
ChanLN(channel_size: int)
```

### `forward(x: Tensor) -> Tensor`

Normalizes each channel independently over the time dimension.

**Input shape:** `[batch, channel, time]`

---

## Class: `InstantLN`

**Instant Layer Normalization** — normalizes across the feature (channel/frequency) dimension at each time step independently. Suitable for causal/streaming models.

### Constructor

```python
InstantLN(channel_size: int)
```

### `forward(x: Tensor) -> Tensor`

Normalizes across channels at each time step: `[batch, channel, time]` → normalization over `channel` dim per `time`.

**Input shape:** `[batch, channel, time]`

---

## Class: `LayerNorm2D`

**2D Layer Normalization** — normalizes over the channel dimension of 2D time-frequency feature maps.

### Constructor

```python
LayerNorm2D(channel_size: int)
```

### `forward(x: Tensor) -> Tensor`

Normalizes along the channel dimension of a 4D tensor.

**Input shape:** `[batch, channel, freq, time]`

---

## Helper Function: `get_norm`

### `get_norm(name: str, channel_size: int) -> nn.Module`

Factory function to create a normalization layer by name.

**Parameters:**
- `name` – Normalization type string
- `channel_size` – Number of input channels

**Supported values:**

| Name | Class |
|------|-------|
| `"global_layer_norm"` | `GlobLN` |
| `"channel_layer_norm"` | `ChanLN` |
| `"instant_layer_norm"` | `InstantLN` |
| `"layer_norm_2d"` | `LayerNorm2D` |
| `"batch_norm"` | `nn.BatchNorm1d` |

## Choosing a Normalization Strategy

| Strategy | Causal Safe | 2D Input | Recommended Use |
|----------|:-----------:|:--------:|-----------------|
| `GlobLN` | ❌ | ❌ | Non-causal 1D (Conv-TasNet) |
| `ChanLN` | ❌ | ❌ | Non-causal 1D |
| `InstantLN` | ✅ | ❌ | Causal/streaming 1D |
| `LayerNorm2D` | ❌ | ✅ | 2D time-frequency (U-Net) |

## Example

```python
from puresound.nnet.lobe.norm import get_norm

norm = get_norm("global_layer_norm", channel_size=256)
out = norm(features)  # [B, 256, T]
```
