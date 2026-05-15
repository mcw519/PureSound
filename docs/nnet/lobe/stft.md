# puresound.nnet.lobe.stft

STFT kernel generation, Mel filterbank construction, and overlap-add reconstruction utilities.

## Functions

### `create_fourier_kernels(n_fft: int, win_length: int, freq_scale: str = "linear", sr: int = 22050, fmin: float = 50.0, fmax: float = 8000.0) -> Tuple[Tensor, Tensor]`

Generates real and imaginary STFT kernels as 1D convolutional filter banks.

**Parameters:**
- `n_fft` – FFT size
- `win_length` – Analysis window length
- `freq_scale` – Frequency binning:
  - `"linear"` – Standard uniform frequency bins
  - `"log"` – Log-scale frequency bins (higher resolution at lower frequencies)
- `sr` – Sample rate (used for log-scale binning)
- `fmin` / `fmax` – Min/max frequency range for log-scale binning

**Returns:** `(real_kernels, imag_kernels)` each of shape `[n_fft//2+1, 1, win_length]`.

---

### `mel_filterbank(n_fft: int, n_mels: int, sr: int, fmin: float = 0.0, fmax: Optional[float] = None) -> Tensor`

Generates a Mel-scale triangular filterbank matrix.

**Parameters:**
- `n_fft` – FFT size (determines number of linear frequency bins = n_fft//2+1)
- `n_mels` – Number of Mel filter channels
- `sr` – Sample rate
- `fmin` / `fmax` – Frequency range for Mel filters

**Returns:** Filterbank matrix `[n_mels, n_fft//2+1]`.

---

### `extend_fbins(x: Tensor) -> Tensor`

Extends frequency bins at the edges to handle boundary effects in filterbank processing.

**Parameters:**
- `x` – Spectrum tensor `[batch, F, T]`

**Returns:** Extended spectrum tensor with padded edge bins.

---

### `overlap_add(frames: Tensor, hop_length: int, window: Optional[Tensor] = None) -> Tensor`

Reconstructs a time-domain signal from overlapping frames using the overlap-add method (WOLA).

**Parameters:**
- `frames` – Framed signal `[batch, frame_length, num_frames]`
- `hop_length` – Hop size between frames
- `window` – Optional synthesis window (Hann by default)

**Returns:** Reconstructed waveform `[batch, T]`.

---

### `torch_window_sumsquare(window: Tensor, n_frames: int, hop_length: int, n_fft: int) -> Tensor`

Computes the window sum-square array used for normalization in overlap-add reconstruction.

**Parameters:**
- `window` – Window function tensor
- `n_frames` – Number of STFT frames
- `hop_length` – Hop size
- `n_fft` – FFT size

**Returns:** Window sum-square tensor `[T]`.

## Example

```python
from puresound.nnet.lobe.stft import create_fourier_kernels, mel_filterbank

real_k, imag_k = create_fourier_kernels(n_fft=512, win_length=512, freq_scale="linear")
mel_fb = mel_filterbank(n_fft=512, n_mels=80, sr=16000, fmin=20.0, fmax=8000.0)
```
