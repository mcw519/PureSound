# puresound.audio.dsp

Digital Signal Processing utilities: resampling, biquad filter design, and parametric equalization.

## Functions

### `wav_resampling(wav: Tensor, orig_sr: int, target_sr: int, backend: str = "sox") -> Tensor`

Resamples an audio waveform from `orig_sr` to `target_sr`.

**Parameters:**
- `wav` – Input waveform tensor
- `orig_sr` – Original sample rate
- `target_sr` – Target sample rate
- `backend` – Resampling backend: `"sox"` (default) or `"torchaudio"`

**Returns:** Resampled waveform tensor.

---

### `get_biquad_params(filter_type: str, fc: float, sr: int, gain_db: float = 0.0, q: float = 0.707) -> Tuple`

Designs biquad filter coefficients for a given filter type.

**Parameters:**
- `filter_type` – One of `"highshelf"`, `"lowshelf"`, `"peaking"`, `"highpass"`, `"lowpass"`, `"notch"`
- `fc` – Cutoff/center frequency in Hz
- `sr` – Sample rate in Hz
- `gain_db` – Gain in dB (for shelf/peaking filters)
- `q` – Quality factor controlling bandwidth

**Returns:** `(b0, b1, b2, a1, a2)` biquad coefficients.

---

### `wav_apply_biquad_filter(wav: Tensor, b0, b1, b2, a1, a2) -> Tensor`

Applies a biquad IIR filter to a waveform using the provided coefficients.

**Parameters:**
- `wav` – Input waveform tensor
- `b0, b1, b2, a1, a2` – Biquad filter coefficients from `get_biquad_params()`

**Returns:** Filtered waveform tensor.

## Class: `ParametricEQ`

A series of biquad filters forming a full parametric equalizer.

### Constructor

```python
ParametricEQ(filter_configs: List[Dict])
```

Each entry in `filter_configs` is a dict with keys:
- `filter_type` – Filter type string
- `fc` – Center/cutoff frequency
- `sr` – Sample rate
- `gain_db` – Gain (optional)
- `q` – Q factor (optional)

### Methods

#### `forward(wav: Tensor) -> Tensor`

Applies the full EQ chain to the input waveform.

**Returns:** Equalized waveform.

---

#### `plot_eq(savefig: Optional[str] = None)`

Visualizes the combined frequency response curve of all biquad stages.

**Parameters:**
- `savefig` – If provided, saves the plot to this file path; otherwise displays interactively.

## Example

```python
from puresound.audio.dsp import ParametricEQ, wav_resampling

wav_16k = wav_resampling(wav, orig_sr=48000, target_sr=16000)

eq = ParametricEQ([
    {"filter_type": "highpass", "fc": 80, "sr": 16000, "q": 0.707},
    {"filter_type": "peaking",  "fc": 1000, "sr": 16000, "gain_db": 3.0, "q": 1.0},
])
wav_eq = eq.forward(wav_16k)
```
