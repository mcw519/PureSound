# puresound.audio.spectrum

STFT analysis/synthesis utilities for complex spectrum manipulation.

## Functions

### `tensor_as_complex(real: Tensor, imag: Tensor) -> Tensor`

Combines real and imaginary parts into a PyTorch complex tensor.

**Parameters:**
- `real` – Real part tensor `[..., F, T]`
- `imag` – Imaginary part tensor `[..., F, T]`

**Returns:** Complex tensor `[..., F, T]`.

---

### `cpx_stft_as_mag_and_phase(cpx_stft: Tensor) -> Tuple[Tensor, Tensor]`

Extracts magnitude and phase from a complex STFT tensor.

**Parameters:**
- `cpx_stft` – Complex STFT tensor `[..., F, T]`

**Returns:** `(magnitude, phase)` where both are real tensors of the same shape.

---

### `mag_and_phase_as_cpx_stft(magnitude: Tensor, phase: Tensor) -> Tensor`

Reconstructs a complex STFT from magnitude and phase.

**Parameters:**
- `magnitude` – Magnitude tensor `[..., F, T]`
- `phase` – Phase tensor `[..., F, T]` (in radians)

**Returns:** Complex STFT tensor.

---

### `wav_to_stft(wav: Tensor, n_fft: int, hop_length: int, win_length: int, window: Optional[Tensor] = None) -> Tuple[Tensor, Dict]`

Computes the Short-Time Fourier Transform of a waveform.

**Parameters:**
- `wav` – Input waveform `[batch, samples]` or `[samples]`
- `n_fft` – FFT size
- `hop_length` – Hop size in samples
- `win_length` – Analysis window length
- `window` – Window function tensor (defaults to Hann window)

**Returns:**
- `cpx_stft` – Complex STFT tensor `[batch, F, T]`
- `meta` – Dict containing `n_fft`, `hop_length`, `win_length`, `original_length`

---

### `stft_to_wav(cpx_stft: Tensor, hop_length: int, win_length: int, original_length: Optional[int] = None, window: Optional[Tensor] = None) -> Tensor`

Reconstructs a waveform from a complex STFT using inverse STFT (overlap-add).

**Parameters:**
- `cpx_stft` – Complex STFT tensor `[batch, F, T]`
- `hop_length` – Hop size in samples
- `win_length` – Synthesis window length
- `original_length` – If provided, trims or pads output to this length
- `window` – Synthesis window tensor

**Returns:** Reconstructed waveform tensor.

## Example

```python
from puresound.audio.spectrum import wav_to_stft, stft_to_wav

cpx, meta = wav_to_stft(wav, n_fft=512, hop_length=128, win_length=512)
# ... process cpx_stft ...
reconstructed = stft_to_wav(cpx, hop_length=128, win_length=512,
                            original_length=meta["original_length"])
```
