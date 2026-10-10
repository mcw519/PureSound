# puresound.nnet.lobe.stft

繁體中文版本：[stft.zh-TW.md](stft.zh-TW.md)

Free functions behind [`encoder.ConvSTFT`](encoder.md) and the Mel front end of
[`FeatureEncoder`](../features.md): Fourier-kernel generation, overlap-add
reconstruction, and Mel-scale conversions. Adapted from
[nnAudio](https://github.com/KinWaiCheuk/nnAudio) and librosa's Mel utilities.

## `create_fourier_kernels`

Builds the sine and cosine kernels a `Conv1d` uses to compute a DFT:
`wcos[k, 0, n] = cos(2π f_k n / n_fft)`, `wsin[k, 0, n] = sin(2π f_k n / n_fft)`,
where `f_k` is bin `k`'s frequency in DFT-bin units.

```python
create_fourier_kernels(
    n_fft,                 # window size
    win_length=None,       # defaults to n_fft; does not change the kernel width
    freq_bins=None,        # defaults to n_fft // 2 + 1
    fmin=50,               # range for "linear" / "log"; ignored for "no"
    fmax=6000,
    sr=44100,              # converts fmin / fmax to bin units
    freq_scale="linear",   # "linear" | "log" | "no"
) -> (wsin, wcos, bins2freq, binslist)
```

- `"linear"` – `freq_bins` bins evenly spaced from `fmin` towards `fmax`.
- `"log"` – log-spaced from `fmin` towards `fmax`.
- `"no"` – the standard DFT bins, 0 Hz to Nyquist; `fmin`/`fmax` ignored.
  Any other value logs a warning and returns uninitialised kernels.

Returns `wsin`/`wcos` as float32 NumPy arrays `(freq_bins, 1, n_fft)`,
`bins2freq` (each bin's frequency in Hz) and `binslist` (the same in DFT-bin
units). `ConvSTFT` passes its own `sr`/`fmin`/`fmax`/`freq_scale` (default
`"no"`), so these defaults matter only for direct calls.

## `mel_filterbank`

```python
mel_filterbank(
    sr: int,
    n_fft: int,
    n_banks: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,   # defaults to sr / 2
    norm: int = 1,                  # 1: Slaney area normalisation
) -> torch.Tensor                   # [n_banks, n_fft // 2 + 1]
```

Triangular Mel filters with centres evenly spaced on the Mel scale between
`fmin` and `fmax`. With `norm=1` each filter is scaled by
`2 / (f_{i+2} - f_i)`, giving approximately constant energy per band. Raises
`ValueError` if a filter comes out empty (`n_banks` too high for the
`sr`/`n_fft`/`fmax`).

The order is `(sr, n_fft, n_banks)`. Both leading arguments are ints, so a
swapped positional call builds a wrong filterbank without an error; pass them by
keyword. `FeatureEncoder` calls `mel_filterbank(sr, n_fft, n_banks)` and
transposes the result to `[n_fft // 2 + 1, n_banks]`.

## `overlap_add`

```python
overlap_add(X: Tensor, stride: int) -> Tensor
```

Sums framed signal `X [batch, n_fft, n_frames]` at hop `stride` with
`torch.nn.functional.fold`; returns `[batch, n_fft + stride * (n_frames - 1)]`.
It applies no window and does not normalise.

## `torch_window_sumsquare`

```python
torch_window_sumsquare(w, n_frames: int, stride: int, n_fft: int, power=2) -> Tensor
```

The overlap-add of `w ** power` repeated over `n_frames` frames:
`[1, 1, 1, n_fft + stride * (n_frames - 1)]`. Dividing the
`overlap_add` output by it (where it is non-negligible) gives the
window-compensated inverse STFT. `ConvSTFT.inverse` does:

```python
real = overlap_add(real, stride)
w_sum = torch_window_sumsquare(window_mask, n_frames, stride, n_fft)
nonzero = w_sum > 1e-10
real[:, nonzero] = real[:, nonzero].div(w_sum[nonzero])
```

## `extend_fbins`

```python
extend_fbins(X: Tensor) -> Tensor   # [batch, n_fft//2 + 1, T, 2] -> [batch, n_fft, T, 2]
```

Rebuilds the two-sided spectrum from a one-sided one: mirrors bins `1 .. -2`
(DC and Nyquist excluded) onto the top half and negates their imaginary part
(Hermitian symmetry of a real signal's spectrum). Used by `ConvSTFT.inverse`.

## Mel-scale helpers

- `hz2mel(frequencies)` / `mel2hz(mels)` – Slaney-style Mel scale (librosa's
  default, `htk=False`): linear below 1000 Hz (`f_sp = 200/3` Hz per Mel),
  logarithmic above.
- `fft_frequencies(sr=16000, n_fft=512)` – centre frequency of each of the
  `n_fft // 2 + 1` FFT bins.
- `mel_frequencies(n_mels=128, fmin=0.0, fmax=8000)` – `n_mels` frequencies
  evenly spaced on the Mel scale between `fmin` and `fmax`.

`mel_filterbank` is built on these three.

## Example

```python
from puresound.nnet.lobe.stft import create_fourier_kernels, mel_filterbank

wsin, wcos, bins2freq, _ = create_fourier_kernels(n_fft=512, sr=16000, freq_scale="no")
mel_fb = mel_filterbank(sr=16000, n_fft=512, n_banks=80, fmin=20.0, fmax=8000.0)
```
