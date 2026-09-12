# puresound.audio.dsp

繁體中文版本：[`dsp.zh-TW.md`](dsp.zh-TW.md)

Resampling plus RBJ Audio-EQ-Cookbook biquad design/application, and
`ParametricEQ`, a fixed (non-trainable) chain built on top of them. The
trainable counterpart is `puresound.nnet.lobe.dsp.FrequencyEQLayer`, which
calls the same `get_biquad_params` with the same filter-type strings to seed
a frequency-domain EQ weight that then gets fine-tuned end to end.

## Functions

### `wav_resampling(wav: Tensor, origin_sr: int, target_sr: int, backend: str = "sox", torch_backend_params: Optional[Dict] = None)`

Resamples `wav: [..., L]` from `origin_sr` to `target_sr`. Note the
parameter is `origin_sr`, not `orig_sr`.

- `backend="sox"` (default): a sox `rate` effect via
  `torchaudio.sox_effects`. **Silently reuses the `torchaudio` code path
  below** if the installed torchaudio has no `sox_effects` attribute at all
  (some wheel builds don't bundle it) — there is no error, no warning.
- `backend="torchaudio"`: `torchaudio.transforms.Resample` with a
  **randomized** anti-aliasing filter — `lowpass_filter_width` from
  `{6, 16, 32, 64, 128}`, `rolloff` from `Uniform(0.8, 0.99)`,
  `resampling_method` from `{"sinc_interp_hann", "sinc_interp_kaiser"}` —
  unless `torch_backend_params` supplies `{"lp_width", "rolloff", "window"}`
  explicitly, in which case those exact values are reused instead of
  re-randomizing. This is what lets a down+up round trip (see
  [augmentation.md](augmentation.md)'s `apply_src_effect`) behave like one
  coherent low-quality resampler rather than two independently-random ones.
- `origin_sr == target_sr` short-circuits to a no-op, still returning the
  backend-appropriate tuple shape below.

**Return arity is keyed off the literal `backend` string you passed, not off
which implementation actually ran:**

- `backend="sox"` → always `(wav, target_sr)`, a 2-tuple — *even* in the
  silent-fallback case above, where the randomized torchaudio filter params
  that were actually used are computed internally but **not** returned to
  you (they're simply discarded).
- `backend="torchaudio"` → always `(wav, target_sr, torch_backend_params)`,
  a 3-tuple, so a caller can thread `torch_backend_params` into a paired
  second call.

```python
wav_16k, sr = wav_resampling(wav, origin_sr=48000, target_sr=16000, backend="sox")

down, sr_d, params = wav_resampling(wav, origin_sr=16000, target_sr=8000, backend="torchaudio")
up,   sr_u, _      = wav_resampling(down, origin_sr=8000, target_sr=16000,
                                     backend="torchaudio", torch_backend_params=params)
```

---

### `get_biquad_params(gain_dB: float, cutoff_freq: float, q_factor: float, sample_rate: float, filter_type: str)`

RBJ Audio-EQ-Cookbook biquad design. **All 5 parameters are required — none
have defaults**, and `filter_type` is the *last* positional argument.

`filter_type` must be one of `"high_shelf"`, `"low_shelf"`, `"peaking"`,
`"lpf"`, `"hpf"`, `"bpf"`, `"notch"` — underscored shelf names, 3-letter
pass-filter abbreviations (not `"highpass"`/`"lowpass"`). `gain_dB` only
affects `high_shelf`/`low_shelf`/`peaking` — the `lpf`/`hpf`/`bpf`/`notch`
formulas never reference the gain term (`A = 10**(gain_dB/40)` is computed
unconditionally but simply unused for those four), so any value works for
them; pass `0.0` by convention.

**Returns `(b, a)`, two length-3 numpy arrays** — not a 5-tuple
`(b0, b1, b2, a1, a2)`. Both are already normalized by `a0`, so `a[0] == 1.0`
and there's no separate `a0` to carry around.

```python
b, a = get_biquad_params(gain_dB=0, cutoff_freq=100, q_factor=0.707,
                          sample_rate=16000, filter_type="hpf")
```

`puresound.nnet.lobe.dsp.FrequencyEQLayer` calls this once per band
(`low_shelf`, `peaking` × N, `high_shelf`) at construction time to seed its
learnable frequency-domain EQ weight — the same shelf/peaking vocabulary as
`ParametricEQ` below, just made trainable.

---

### `wav_apply_biquad_filter(wav: Tensor, b_coeff: np.ndarray, a_coeff: np.ndarray)`

Applies the filter with `scipy.signal.lfilter`, one channel at a time.
Round-trips through numpy internally (clones `wav`, converts to numpy,
unsqueezes a 1-D input to `[1, L]` first) and returns a fresh
`torch.Tensor` — **not differentiable**, gradients do not flow through this
call.

## Class: `ParametricEQ`

A **fixed** (non-trainable, not an `nn.Module`) series of biquads: one low
shelf, N peaking bands, one high shelf, applied in that order.

### Constructor

```python
ParametricEQ(
    sample_rate: float,
    eq_band_gain: Tuple[float, ...],
    eq_band_cutoff: Tuple[float, ...],
    eq_band_q_factor: Tuple[float, ...],
    low_shelf_gain_dB: float = 0.0,
    low_shelf_cutoff_freq: float = 80,
    low_shelf_q_factor: float = 0.707,
    high_shelf_gain_dB: float = 0.0,
    high_shelf_cutoff_freq: float = 1000,
    high_shelf_q_factor: float = 0.707,
    dtype = np.float32,
)
```

`eq_band_gain`/`eq_band_cutoff`/`eq_band_q_factor` must be equal-length —
one `"peaking"` band per index (`assert`ed in the constructor). Total filter
count applied per `forward` call is `len(eq_band_gain) + 2` (the two
shelves included).

### Methods

#### `forward(wav: Tensor) -> Tensor`

Applies every biquad in sequence via `wav_apply_biquad_filter`.

#### `plot_eq(savefig: Optional[str] = None)`

Plots `|H(f)|` for the combined chain — the product, across all stages, of
each stage's `rfft(b) / rfft(a)`. **Requires `sample_rate` to be exactly
`16000` or `32000`** — any other value raises a bare `ValueError` (the plot
FFT size, 512 vs 1024, is hardcoded per rate rather than derived from it).
Draws with `matplotlib.pyplot` directly (no figure is created/returned; call
this once per plot you want, and `savefig=...` if you don't want it to try
to display interactively).

## Example

```python
from puresound.audio.dsp import ParametricEQ, wav_resampling

wav_16k, sr = wav_resampling(wav, origin_sr=48000, target_sr=16000, backend="sox")

eq = ParametricEQ(
    sample_rate=16000,
    eq_band_gain=(3.0,),
    eq_band_cutoff=(1000.0,),
    eq_band_q_factor=(1.0,),
    low_shelf_cutoff_freq=80,
    high_shelf_cutoff_freq=7800,
)
wav_eq = eq.forward(wav_16k)
```
