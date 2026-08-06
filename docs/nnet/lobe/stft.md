# puresound.nnet.lobe.stft

繁體中文版本：`stft.zh-TW.md`

Free functions behind [`encoder.ConvSTFT`](encoder.md): Fourier-kernel
generation, overlap-add reconstruction, and Mel-scale conversions. Adapted
from [nnAudio](https://github.com/KinWaiCheuk/nnAudio) and
[librosa](https://librosa.org/)-style Mel utilities.

## `create_fourier_kernels`

```python
create_fourier_kernels(
    n_fft,
    win_length=None,
    freq_bins=None,
    fmin=50,
    fmax=6000,
    sr=44100,
    freq_scale="linear",
) -> Tuple[np.ndarray, np.ndarray, list, list]
```

**Parameters:**
- `n_fft` – window size
- `win_length` – defaults to `n_fft` if `None` (accepted for API symmetry with callers; not otherwise used to change the kernel width)
- `freq_bins` – number of frequency bins; defaults to `n_fft // 2 + 1` if `None`
- `fmin` / `fmax` – frequency range for `"linear"`/`"log"` binning; ignored when `freq_scale="no"`
- `sr` – sample rate, used to convert `fmin`/`fmax` to bin indices
- `freq_scale` – `"linear"` (uniform spacing between `fmin`/`fmax`), `"log"` (log spacing between `fmin`/`fmax`), or `"no"` (uniform spacing from 0 Hz to Nyquist, ignoring `fmin`/`fmax`)

**Returns:** `(wsin, wcos, bins2freq, binslist)` — `wsin`/`wcos` are real
NumPy arrays of shape `(freq_bins, 1, n_fft)`; `bins2freq` maps each bin index
to its center frequency in Hz; `binslist` is the same mapping in normalized
DFT-bin units.

`ConvSTFT` always calls this with explicit `sr`/`fmax` (its own defaults are
`sr=22050, fmax=6000` — see [`encoder.md`](encoder.md)), so the different
bare-function defaults above (`sr=44100`, `fmax=6000`) only matter if you
call `create_fourier_kernels` directly.

## `mel_filterbank`

```python
mel_filterbank(
    sr: int,
    n_fft: int,
    n_banks: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    norm: int = 1,
) -> torch.Tensor
```

> **Parameter order is `(sr, n_fft, n_banks, ...)`, not `(n_fft, n_mels,
> sr, ...)`.** Calling this positionally with the latter order silently
> passes your sample rate where `n_fft` is expected and vice versa — no
> exception is raised (both are plain ints), the filterbank is just wrong.
> Always pass `sr` and `n_fft` by keyword unless you're sure of the order.
> The bank-count parameter is also named `n_banks`, not `n_mels`.

**Parameters:**
- `sr` – sample rate
- `n_fft` – FFT size (linear bin count = `n_fft // 2 + 1`)
- `n_banks` – number of Mel filters (feature dimension of the output)
- `fmin` / `fmax` – Mel frequency range; `fmax` defaults to `sr / 2` if `None`
- `norm` – if `1` (default), Slaney-style area normalization (approximately constant energy per band)

**Returns:** `[n_banks, n_fft // 2 + 1]`. Raises `ValueError` if any filter
ends up empty (`fmax` too low or `n_banks` too high for the given `sr`/`n_fft`).

## `overlap_add`

```python
overlap_add(X: Tensor, stride: int) -> Tensor
```

**Parameters:**
- `X` – framed signal `[batch, n_fft, n_frames]`
- `stride` – hop size between frames

Implemented as `torch.nn.functional.fold` with `kernel_size=(1, n_fft)`. There
is **no `window` parameter** — `overlap_add` only sums overlapping frames, it
does not know about or apply any window.

**Returns:** `[batch, n_fft + stride * (n_frames - 1)]`.

> **This raw output is not yet normalized.** Overlapping windowed frames
> summed via `fold` are scaled by however much the window overlaps at each
> sample — correct reconstruction requires dividing by
> [`torch_window_sumsquare`](#torch_window_sumsquare) afterward wherever that
> sum-square is non-negligible. `ConvSTFT.inverse` does exactly this (see
> [`encoder.md`](encoder.md)):
> ```python
> real = overlap_add(real, stride)
> w_sum = torch_window_sumsquare(window_mask, n_frames, stride, n_fft)
> nonzero = w_sum > 1e-10
> real[:, nonzero] = real[:, nonzero].div(w_sum[nonzero])
> ```

## `torch_window_sumsquare`

```python
torch_window_sumsquare(w, n_frames: int, stride: int, n_fft: int, power=2) -> Tensor
```

**Parameters:**
- `w` – the window function tensor, length `n_fft`
- `n_frames` – number of frames that were overlap-added
- `stride` – hop size
- `n_fft` – FFT/window size
- `power` – exponent applied to the window before summing (`2` for the usual sum-of-squares normalization)

Computes the same overlap-add as `overlap_add`, but on `w**power` repeated
across all `n_frames` instead of on real frame data — i.e. "what would the
overlap-add of this window alone sum to at each sample." **Returns:**
`[batch=1, n_fft + stride * (n_frames - 1)]`.

## `extend_fbins`

```python
extend_fbins(X: Tensor) -> Tensor
```

**Parameters:** `X` – one-sided spectrum `[batch, n_fft//2+1, T, 2]`

Mirrors bins `1..-2` (excluding DC and Nyquist) back onto the top half,
negating the imaginary part of the mirrored bins (odd symmetry) to
reconstruct the full `n_fft`-wide two-sided spectrum needed by the inverse
convolution kernels. **Returns:** `[batch, n_fft, T, 2]`.

## Mel-scale helpers

- **`hz2mel(frequencies)`** / **`mel2hz(mels)`** – HTK-style Hz↔Mel conversion with a linear region below 1000 Hz and a log region above (librosa-compatible formulas).
- **`fft_frequencies(sr=16000, n_fft=512) -> np.ndarray`** – center frequency of each of the `n_fft//2+1` linear FFT bins.
- **`mel_frequencies(n_mels=128, fmin=0.0, fmax=8000)`** – `n_mels` Mel-band center frequencies, uniformly spaced in Mel scale between `fmin`/`fmax`.

`mel_filterbank` uses all three of the above internally.

## Wiring

`ConvSTFT`/`ConvEncDec` ([`encoder.md`](encoder.md)) build on
`create_fourier_kernels`, `extend_fbins`, `overlap_add`, and
`torch_window_sumsquare` for their forward/inverse STFT. `mel_filterbank` is
used directly by `puresound/nnet/features.py`'s `FeatureEncoder`
(`mel_filterbank(sr, n_fft, n_banks)` — called correctly, by keyword-safe
positional order) to build a Mel projection matrix.

## Example

```python
from puresound.nnet.lobe.stft import create_fourier_kernels, mel_filterbank

wsin, wcos, bins2freq, _ = create_fourier_kernels(n_fft=512, sr=16000, fmax=8000)
mel_fb = mel_filterbank(sr=16000, n_fft=512, n_banks=80, fmin=20.0, fmax=8000.0)
```
