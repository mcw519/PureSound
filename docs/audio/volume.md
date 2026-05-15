# puresound.audio.volume

Audio amplitude and volume manipulation utilities.

## Functions

### `calculate_rms(wav: Tensor) -> float`

Computes the Root Mean Square (RMS) amplitude of a waveform.

**Parameters:**
- `wav` – Input waveform tensor

**Returns:** RMS value (scalar float).

---

### `normalize_waveform(wav: Tensor, mode: str = "peak") -> Tensor`

Normalizes a waveform to a standard amplitude level.

**Parameters:**
- `wav` – Input waveform tensor
- `mode` – Normalization strategy:
  - `"peak"` – Normalize by peak amplitude to [-1, 1]
  - `"rms"` – Normalize by RMS to unit RMS
  - `"avg"` – Normalize by mean absolute value

**Returns:** Normalized waveform tensor.

---

### `rescale_waveform(wav: Tensor, target_level: float, mode: str = "dB") -> Tensor`

Rescales a waveform to a target amplitude level.

**Parameters:**
- `wav` – Input waveform tensor
- `target_level` – Target amplitude level
- `mode` – Level mode:
  - `"dB"` – `target_level` interpreted as dBFS (e.g., `-26.0`)
  - `"linear"` – `target_level` interpreted as linear RMS value

**Returns:** Rescaled waveform tensor.

---

### `rand_gain_distortion(wav: Tensor, min_gain: float, max_gain: float) -> Tensor`

Applies a random gain factor uniformly sampled from `[min_gain, max_gain]`.

**Parameters:**
- `wav` – Input waveform tensor
- `min_gain` – Minimum gain multiplier
- `max_gain` – Maximum gain multiplier

**Returns:** Gain-distorted waveform tensor.

---

### `wav_fade_in(wav: Tensor, duration: float, sr: int, mode: str = "linear") -> Tensor`

Applies a fade-in effect to the beginning of a waveform.

**Parameters:**
- `wav` – Input waveform tensor
- `duration` – Duration of the fade-in in seconds
- `sr` – Sample rate
- `mode` – Fade curve shape: `"linear"`, `"exponential"`, or `"logarithmic"`

**Returns:** Waveform with fade-in applied.

---

### `wav_fade_out(wav: Tensor, duration: float, sr: int, mode: str = "linear") -> Tensor`

Applies a fade-out effect to the end of a waveform.

**Parameters:**
- `wav` – Input waveform tensor
- `duration` – Duration of the fade-out in seconds
- `sr` – Sample rate
- `mode` – Fade curve shape: `"linear"`, `"exponential"`, or `"logarithmic"`

**Returns:** Waveform with fade-out applied.

---

### `wav_clipping(wav: Tensor, quantile: float = 0.99) -> Tensor`

Clips waveform amplitude at the specified quantile bounds, simulating hard clipping distortion.

**Parameters:**
- `wav` – Input waveform tensor
- `quantile` – Clipping threshold quantile (e.g., `0.99` clips the top and bottom 1%)

**Returns:** Clipped waveform tensor.

## Example

```python
from puresound.audio.volume import normalize_waveform, rescale_waveform, wav_fade_in

wav_norm = normalize_waveform(wav, mode="peak")
wav_rms  = rescale_waveform(wav, target_level=-26.0, mode="dB")
wav_fade = wav_fade_in(wav_rms, duration=0.01, sr=16000, mode="linear")
```
