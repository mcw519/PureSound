# puresound.nnet.lobe.dsp

Learnable DSP-inspired layers for differentiable signal processing.

## Class: `FrequecyEQLayer`

A trainable parametric equalizer implemented as a learnable biquad filter chain. Applies frequency-domain shaping as a differentiable neural network layer.

> **Note:** Exported from `puresound.nnet` as `FrequecyEQLayer` (spelling preserved for compatibility).

### Constructor

```python
FrequecyEQLayer(
    num_bands: int,
    sr: int,
    trainable: bool = True,
)
```

**Parameters:**
- `num_bands` – Number of biquad EQ bands (filter stages)
- `sr` – Audio sample rate in Hz
- `trainable` – If `True`, all filter parameters (frequencies, gains, Q factors) are learnable; if `False`, the layer uses fixed initial parameters

### `forward(wav: Tensor) -> Tensor`

Applies the learned EQ filtering to a waveform.

**Parameters:**
- `wav` – Input waveform tensor `[batch, 1, T]` or `[batch, T]`

**Returns:** EQ-filtered waveform tensor of the same shape.

### Learnable Parameters

When `trainable=True`, the following are learned:

| Parameter | Description |
|-----------|-------------|
| `center_freqs` | Center frequency for each EQ band (Hz) |
| `gains_db` | Per-band gain in dB |
| `q_factors` | Per-band Q factor (bandwidth control) |

## Use Cases

- **Post-processing EQ**: Learn frequency corrections as part of an end-to-end enhancement pipeline
- **Microphone compensation**: Adapt to target device frequency responses
- **Perceptual weighting**: Emphasize perceptually important frequency regions

## Example

```python
from puresound.nnet.lobe.dsp import FrequecyEQLayer

eq_layer = FrequecyEQLayer(num_bands=8, sr=16000, trainable=True)
wav_eq = eq_layer(enhanced_wav)
```
