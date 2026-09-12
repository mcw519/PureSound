# puresound.nnet.lobe.encoder

繁體中文版本：`encoder.zh-TW.md`

Audio encoder/decoder modules that transform waveforms to latent feature
representations and back. Every class here exposes exactly `forward()`
(analysis) and `inverse()` (synthesis) — there are no `encode()`/`decode()`
method names anywhere in this module.

## Class: `FreeEncDec`

A fully learnable analysis/synthesis filterbank: `Conv1d` encoder,
`ConvTranspose1d` decoder, no constraint tying the two together. Also known
as a "free" encoder since the filters are learned end-to-end rather than
fixed to a known basis (contrast with `ConvSTFT` below).

> Exported from `puresound.nnet` as `FreeEncDec`.

```python
FreeEncDec(
    win_length: int = 512,
    laten_length: int = 512,
    hop_length: int = 128,
    output_active: Optional[str] = None,
)
```

**Parameters:**
- `win_length` – analysis/synthesis window length (the `Conv1d`/`ConvTranspose1d` kernel size)
- `laten_length` – latent feature dimension (encoder output channels / decoder input channels)
- `hop_length` – stride of both the encoder and decoder
- `output_active` – if given, an `nn.{output_active}()` activation (looked up via `getattr(nn, output_active)`, e.g. `"ReLU"`) is appended after the encoder — there is **no `bias` parameter**; both convs are always built with `bias=False`

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, L]` or `[N, 1, L]`

**Returns:** `[N, laten_length, T]`, `T = (L - win_length) // hop_length + 1`.

### `inverse(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, laten_length, T]`

**Returns:** `[N, L]` (channel dim squeezed after the `ConvTranspose1d`).

---

## Class: `ConvEncDec`

STFT-based encoder/decoder: a trainable convolutional STFT (`ConvSTFT`,
below) wrapped with window construction and optional pre-emphasis.

> Exported from `puresound.nnet` as `ConvEncDec`.

```python
ConvEncDec(
    fft_length: int = 512,
    win_type: str = "hann",
    win_length: int = 512,
    freq_bins: int = None,
    hop_length: int = 128,
    freq_scale: str = "no",
    iSTFT: bool = True,
    fmin: int = 0,
    fmax: int = 8000,
    sr: int = 16000,
    preemphasis: Optional[float] = None,
    trainable: bool = True,
)
```

**Parameters:**
- `fft_length` – FFT size (`n_fft` passed down to `ConvSTFT`)
- `win_type` – `"hann"`, `"hamming"`, or `"blackman"` — anything else raises `NotImplementedError`
- `win_length` – analysis window length
- `freq_bins` – number of frequency bins to keep; `None` means `n_fft // 2 + 1`
- `hop_length` – STFT hop size
- `freq_scale` – frequency-bin spacing, one of **`"linear"`, `"log"`, `"no"`** (not just 2 values) — see [`stft.create_fourier_kernels`](stft.md); default here is `"no"` (uniform bins from 0 Hz to Nyquist, `fmin`/`fmax` ignored), which differs from `create_fourier_kernels`'s own bare default of `"linear"`
- `iSTFT` – if `True`, also builds the inverse kernels needed by `inverse()`
- `fmin` / `fmax` – only used when `freq_scale` is `"linear"` or `"log"`
- `sr` – sample rate, used for `"linear"`/`"log"` bin-frequency mapping
- `preemphasis` – if set, applies `x[t] - preemphasis * x[t-1]` before the STFT (first sample left unchanged, via zero-padding)
- `trainable` – if `True`, the underlying STFT kernels (`wsin`/`wcos`) are learnable parameters instead of fixed buffers

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, L]`

**Returns:** `[N, C, T, 2]` (real/imag stacked on the last axis; `C = freq_bins`).

### `inverse(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, C, T, 2]`

**Returns:** `[N, L]`.

---

## Class: `ConvSTFT`

Trainable STFT/iSTFT implemented as `Conv1d`/`Conv2d` with sinusoidal kernels
(no autograd through `torch.stft`), adapted from
[nnAudio](https://github.com/KinWaiCheuk/nnAudio). This is what `ConvEncDec`
builds internally; it can also be used standalone.

```python
ConvSTFT(
    window_mask: torch.Tensor,
    n_fft: int = 2048,
    win_length: Optional[int] = None,
    freq_bins: Optional[int] = None,
    hop_length: Optional[int] = None,
    freq_scale: str = "no",
    iSTFT: bool = False,
    fmin: int = 50,
    fmax: int = 6000,
    sr: int = 22050,
    trainable: bool = False,
)
```

**Parameters:**
- `window_mask` – **required**, a precomputed 1D window tensor of length `n_fft` (e.g. `torch.hann_window(win_length)`) — this is a tensor, not a `window: str` name; raises `TypeError` if `len(window_mask) != n_fft`
- `n_fft` – FFT size
- `win_length` – defaults to `n_fft` if `None`
- `freq_bins` – defaults to `n_fft // 2 + 1` if `None`
- `hop_length` – defaults to `win_length // 4` if `None`
- `freq_scale`, `fmin`, `fmax`, `sr` – forwarded to [`stft.create_fourier_kernels`](stft.md)
- `iSTFT` – if `True`, also registers the mirrored inverse kernels (`kernel_sin_inv`, `kernel_cos_inv`) needed by `inverse()`
- `trainable` – if `True`, `wsin`/`wcos` (the window-multiplied kernels) are `nn.Parameter`; otherwise they're registered buffers

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, channel, L]`

Runs two `Conv1d`s (`wsin`, `wcos`) at stride `hop_length`, truncates to
`freq_bins`. **Returns:** `[N, C, T, 2]` — note the imaginary part is
negated on the way out (`torch.stack((spec_real, -spec_imag), -1)`).

### `inverse(X: Tensor, refresh_win: bool = True) -> Tensor`

Requires `iSTFT=True` at construction (else raises `NameError`) and
`X.dim() == 4`. Mirrors bins back out to full `n_fft` width
([`stft.extend_fbins`](stft.md)), runs the inverse `Conv2d`s, reconstructs
with [`stft.overlap_add`](stft.md), then **normalizes** by dividing the raw
overlap-add output by [`stft.torch_window_sumsquare`](stft.md) at every
position where that sum-square is non-negligible (`> 1e-10`) — this
normalization step is not optional; skipping it leaves the reconstruction
scaled by the window's overlap envelope. The sum-square is cached
(`self.w_sum`) and only recomputed when `refresh_win=True` or on first call,
since it depends only on the number of frames, not on `X`'s values.

---

## Class: `UnifiedConvEncDec`

Handles **any** input sample rate by keeping one `ConvSTFT` per supported
rate, all built to the same 25 ms window / 10 ms hop:

```python
UnifiedConvEncDec(win_type: str = "hann", trainable: bool = False)
```

Supported sample rates (fixed table in `get_stft_parms`): `8000, 16000,
22050, 24000, 32000, 44100, 48000` Hz, each with its own `n_fft`/`hop_length`
pair matching 25 ms/10 ms at that rate, `freq_scale="no"`, `iSTFT=True`.

### `forward(x: Tensor, sr: Union[Tensor, int]) -> Tensor`

**Parameters:**
- `x` – `[N, L]`
- `sr` – a plain `int` (whole batch shares one sample rate → dispatches to a single `ConvSTFT`), or a per-sample `Tensor[N]` (loops sample-by-sample, dispatching each to its own rate's encoder and concatenating — a mixed-sample-rate batch is supported, just not vectorized)

**Returns:** `[N, C, T, 2]`.

### `inverse(x: Tensor, sr: Union[Tensor, int]) -> Tensor`

Symmetric to `forward`. **Returns:** `[N, L]`.

No caller in the repository's recipes today — exercised directly by
`test/test_lobe.py::test_unified_stft_encoder` across all 7 rates; a library
piece for multi-sample-rate deployments.

## Wiring

`ConvEncDec` and `FreeEncDec` are re-exported as `puresound.nnet.ConvEncDec` /
`puresound.nnet.FreeEncDec` and are the two `encoder.type` choices recipe
configs pick between (e.g. `egs/default_config.yaml`'s
`encoder: {type: ConvEncDec, encoder_args: {...}}`).

## Example

```python
from puresound.nnet.lobe.encoder import FreeEncDec, ConvEncDec

# Learnable filterbank encoder
enc = FreeEncDec(win_length=16, hop_length=8, laten_length=512)
feat = enc(wav)
wav_out = enc.inverse(feat)

# STFT encoder
enc_stft = ConvEncDec(fft_length=512, hop_length=128, win_length=512, freq_scale="no")
spec = enc_stft(wav)          # [N, C, T, 2]
wav_out = enc_stft.inverse(spec)
```
