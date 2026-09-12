# puresound.audio.impulse_response

繁體中文版本：[`impulse_response.zh-TW.md`](impulse_response.zh-TW.md)

Room Impulse Response (RIR) convolution and random IIR filtering utilities.

> **Note:** the file itself, `impulse_response.py`, is spelled correctly. The
> typo lives one level down: `wav_apply_rir`'s RIR argument is named
> `impaulse`, preserved for backwards compatibility.

## Functions

### `compute_drr_db(rir: Tensor, sample_rate: int, direct_window_ms: float = 2.5) -> float`

Thin re-export of `puresound.audio.rir.metrics.compute_drr_db`. Direct path =
energy in `[peak, peak + direct_window_ms]`; everything after that window is
the reverberant tail. Returns `+inf` when the tail carries no energy (e.g. an
anechoic or already-trimmed RIR).

---

### `wav_apply_rir(wav: Tensor, impaulse: Tensor, sample_rate: int, rir_mode: str = "full") -> Tensor`

Convolves a waveform with a Room Impulse Response to simulate reverberation.

**Parameters:**
- `wav` – Clean speech waveform tensor `[channels, T]` (2-D; a bare `[T]`
  tensor raises on the internal `wav_ch, _ = wav.shape` unpack)
- `impaulse` – Room impulse response tensor `[channels, T_rir]` (also 2-D)
- `sample_rate` – Sample rate in Hz (sizes the `"direct"`/`"early"` windows)
- `rir_mode` – Reverberation mode:
  - `"full"` – Full convolution; output retains all late reflections
  - `"direct"` – Truncates the RIR to `[peak, peak + 6 ms]`
  - `"early"` – Truncates the RIR to `[peak, peak + 50 ms]`

**Returns:** Reverberant waveform, trimmed to `wav`'s original length and
time-aligned so the output starts at the RIR's peak sample (the direct
arrival), not at the start of the raw convolution.

If `impaulse` has one channel, every `wav` channel is convolved with it
independently ([N, T] in, [N, T] out). If `impaulse` has multiple channels,
`wav` must be single-channel and each RIR channel produces one output channel
([1, T] in, [C_rir, T] out) — the mic-array case.

**Notes:**
- Uses FFT-based convolution (`fftconvolve`).
- In `"direct"` and `"early"` modes the RIR is windowed *before* convolution,
  then peak-normalized. Because the window always starts at the peak, all
  three `rir_mode`s apply the same scale factor to a given RIR — so an
  `"early"`-mode clean target and a `"full"`-mode noisy mixture built from the
  same RIR share the same direct-path level; only late-reverb energy differs.
- **Side effect:** bank RIRs carry a physical `1/r` distance gain on disk with
  inter-channel level ratios intact, but this per-channel peak normalization
  discards it — a mixture does not inherit a distance-dependent level; only
  DRR, decay shape, and spectral tilt still distinguish near from far.
  Recipes' SIR handling (`mix_mode: physical`, hard-SIR ranges) assumes this;
  changing the normalization changes those assumptions with it.

---

### `rand_add_2nd_filter_response(wav: Tensor, a: Optional[Tensor] = None, b: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]`

Applies a random second-order IIR (biquad) filter to simulate microphone or
channel frequency coloration [1].

**Parameters:**
- `wav` – Input waveform tensor
- `a`, `b` – Optional pre-sampled denominator/numerator coefficients, each
  `[1, x1, x2]`. When omitted, both `x1, x2` pairs are drawn i.i.d. uniform
  in `[-3/8, 3/8]`.

**Returns:** `(filtered_wav, a, b)` — the sampled coefficients come back so a
caller can re-apply the exact same random filter to a second, related signal
(e.g. color a mixture and its clean target identically).

**Notes:**
- There is no `sample_rate` parameter and no frequency/Q/gain design step:
  coefficients are drawn directly in the biquad's `a`/`b` coefficient space,
  not from a highpass/lowpass/peaking/shelf parameterization.

Reference: [1] *A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band
Speech Enhancement*.

## Example

```python
from puresound.audio.impulse_response import wav_apply_rir, rand_add_2nd_filter_response
from puresound.audio.io import AudioIO

rir, _ = AudioIO.open("rir.wav")
clean, sr = AudioIO.open("clean.wav", target_sr=16000)

reverberant = wav_apply_rir(clean, rir, sr, rir_mode="early")
colored, a, b = rand_add_2nd_filter_response(clean)
```
