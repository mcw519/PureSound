# puresound.nnet.lobe.encoder

繁體中文版本：[encoder.zh-TW.md](encoder.zh-TW.md)

Waveform encoders and their inverses: a learned filterbank (`FreeEncDec`) and
an STFT implemented as convolutions with explicit Fourier kernels
(`ConvSTFT`, wrapped by `ConvEncDec` and `UnifiedConvEncDec`). Every class
has `forward()` for analysis and `inverse()` for synthesis.

None of them pad the signal: with `L` input samples there are
`T = (L - win) // hop + 1` frames, and `inverse` returns
`(T - 1) * hop + win` samples, so a tail shorter than one hop is dropped.

## Class: `FreeEncDec`

A learned analysis `Conv1d` and synthesis `ConvTranspose1d`, both without bias
and not tied to each other.

```python
FreeEncDec(
    win_length: int = 512,               # kernel size of both convs
    laten_length: int = 512,             # number of learned filters
    hop_length: int = 128,               # stride of both convs
    output_active: Optional[str] = None, # nn class name appended to the encoder, e.g. "ReLU"
)
```

| method | shape |
| --- | --- |
| `forward(x)` | `[N, L]` or `[N, 1, L]` -> `[N, laten_length, T]` |
| `inverse(x)` | `[N, laten_length, T]` -> `[N, L']` |

## Class: `ConvEncDec`

A `ConvSTFT` with its window built from a name, plus optional pre-emphasis.

```python
ConvEncDec(
    fft_length: int = 512,
    win_type: str = "hann",          # "hann" | "hamming" | "blackman"
    win_length: int = 512,           # must equal fft_length, else TypeError
    freq_bins: int = None,           # None -> fft_length // 2 + 1
    hop_length: int = 128,
    freq_scale: str = "no",          # "no" | "linear" | "log", see stft.create_fourier_kernels
    iSTFT: bool = True,              # build the inverse kernels
    fmin: int = 0,                   # used by "linear" / "log" only
    fmax: int = 8000,
    sr: int = 16000,
    preemphasis: Optional[float] = None,  # x[t] - p * x[t-1] before the STFT
    trainable: bool = True,          # Fourier kernels are parameters
)
```

| method | shape |
| --- | --- |
| `forward(x)` | `[N, L]` -> `[N, F, T, 2]` (real, imaginary) |
| `inverse(x)` | `[N, F, T, 2]` -> `[N, L']` |

`inverse` does not undo the pre-emphasis. `trainable` defaults to `True`;
recipes that want a fixed STFT set it to `False`.

## Class: `ConvSTFT`

STFT and inverse STFT as convolutions, adapted from
[nnAudio](https://github.com/KinWaiCheuk/nnAudio).

```python
ConvSTFT(
    window_mask: torch.Tensor,        # window tensor of length n_fft, else TypeError
    n_fft: int = 2048,
    win_length: Optional[int] = None, # None -> n_fft
    freq_bins: Optional[int] = None,  # None -> n_fft // 2 + 1
    hop_length: Optional[int] = None, # None -> win_length // 4
    freq_scale: str = "no",
    iSTFT: bool = False,
    fmin: int = 50,
    fmax: int = 6000,
    sr: int = 22050,
    trainable: bool = False,          # windowed kernels wsin / wcos as parameters
)
```

`forward(x [N, 1, L]) -> [N, F, T, 2]` computes, with window `w` and hop `H`,

```
X[k, t] = sum_n w[n] x[t*H + n] exp(-j 2 pi k n / n_fft)
```

as two strided `Conv1d`s with kernels `w * cos` and `w * sin`; the imaginary
part is negated on output.

`inverse(X [N, F, T, 2], refresh_win: bool = True) -> [N, L']` mirrors the bins
to the full `n_fft` (Hermitian), applies the inverse kernels, multiplies by the
window, divides by `n_fft`, overlap-adds, and divides by the window's
sum-square wherever it exceeds `1e-10`. It needs `iSTFT=True` (else
`NameError`). The sum-square depends only on the frame count and is cached;
`refresh_win=False` reuses it.

## Class: `UnifiedConvEncDec`

One `ConvSTFT` per sample rate, each with a 25 ms window and 10 ms hop
(`n_fft = 0.025 * sr`, `hop = 0.01 * sr`), `freq_scale="no"`, `iSTFT=True`.

```python
UnifiedConvEncDec(win_type: str = "hann", trainable: bool = False)
```

Supported rates: 8000, 16000, 22050, 24000, 32000, 44100, 48000 Hz.
`forward(x [N, L], sr)` returns `[N, F, T, 2]` and `inverse(x, sr)` returns
`[N, L']`. `sr` is an `int` for the whole batch, or a `Tensor[N]` dispatched
row by row and concatenated, so all rows must produce the same shape.

The per-rate encoders are submodules in an `nn.ModuleDict` keyed by the rate as
a string (`encoder["16000"]`), so `.to(device)` moves them, their kernels are in
`state_dict()`, and with `trainable=True` the windowed kernels are in
`parameters()`. No recipe uses this class.

## Use in a recipe

`ConvEncDec` and `FreeEncDec` are exported from `puresound.nnet` and chosen by
`model.encoder.type`:

```yaml
model:
  encoder:
    type: ConvEncDec
    encoder_args:
      fft_length: 512
      win_type: "hann"
      win_length: 512
      hop_length: 160
      fmin: 0
      fmax: 8000
      sr: 16000
      trainable: False
```

## Design notes

- Expressing the STFT as convolutions with explicit kernels makes analysis and
  synthesis ordinary layers: the kernels can be trained, and the graph has no
  `torch.stft` call.
- Dividing by the window sum-square makes the inverse exact for any window and
  hop with enough overlap, instead of only for windows that satisfy the
  constant-overlap-add condition.
