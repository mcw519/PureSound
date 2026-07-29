# puresound.audio.impulse_response

Room Impulse Response (RIR) convolution and random IIR filtering utilities.

> **Note:** The source file is named `impulse_response.py` (typo preserved for backwards compatibility).

## Functions

### `wav_apply_rir(wav: Tensor, rir: Tensor, sr: int, mode: str = "full") -> Tensor`

Convolves a waveform with a Room Impulse Response to simulate reverberation.

**Parameters:**
- `wav` – Clean speech waveform tensor `[1, T]` or `[T]`
- `rir` – Room impulse response tensor `[1, T_rir]` or `[T_rir]`
- `sr` – Sample rate in Hz (used to determine early reflection boundary)
- `mode` – Reverberation mode:
  - `"full"` – Full convolution; output retains all late reflections
  - `"direct"` – Truncates RIR to direct path only (before first 2.5 ms)
  - `"early"` – Truncates RIR to direct path + early reflections (≤ 50 ms)

**Returns:** Reverberant waveform tensor, trimmed to the original input length.

**Notes:**
- Uses FFT-based convolution (`fftconvolve`) for efficiency
- In `"direct"` and `"early"` modes, the RIR is windowed before convolution

---

### `rand_add_2nd_filter_response(wav: Tensor, sr: int) -> Tensor`

Applies a random second-order IIR (biquad) filter to simulate microphone or channel frequency coloration.

**Parameters:**
- `wav` – Input waveform tensor
- `sr` – Sample rate in Hz

**Returns:** Filtered waveform tensor.

**Notes:**
- Filter parameters (type, frequency, Q, gain) are sampled randomly on each call
- Filter types may include highpass, lowpass, peaking, shelf variants

## Example

```python
from puresound.audio.impulse_response import wav_apply_rir, rand_add_2nd_filter_response
from puresound.audio.io import AudioIO

rir, _ = AudioIO.open("rir.wav")
clean, sr = AudioIO.open("clean.wav", target_sr=16000)

reverberant = wav_apply_rir(clean, rir, sr=sr, mode="early")
colored = rand_add_2nd_filter_response(clean, sr=sr)
```
