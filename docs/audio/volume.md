# puresound.audio.volume

繁體中文版本：[`volume.zh-TW.md`](volume.zh-TW.md)

Amplitude-domain utilities: RMS measurement, normalization/rescaling, and
three distortion/fade effects used throughout `augmentation.py` and the
`task/*.py` dataset pipelines. All operate on `wav: [..., L]` (time last)
and are shape-preserving.

## Functions

### `calculate_rms(wav: Tensor, to_log: bool = False) -> Tensor`

`sqrt(mean(wav**2, dim=-1))`, reducing only the last axis — for
`wav: [C, L]` this returns `[C]`, **a tensor, not a Python scalar**, unless
`wav` is 1-D. `to_log=True` returns `20 * log10(rms)` (dBFS-style) instead
of the linear value.

---

### `normalize_waveform(wav: Tensor, amp_type: str = "avg") -> Tensor`

Divides `wav` by one of three per-signal amplitude measures (`+ 1e-14` to
avoid division by zero):

| `amp_type` | denominator | reduction |
|---|---|---|
| `"avg"` (default) | `mean(\|wav\|, dim=-1)` | `keepdim=True` |
| `"peak"` | `max(\|wav\|, dim=-1)` | `keepdim=True` |
| `"rms"` | `sqrt(mean(wav**2, dim=-1))` | **`keepdim=False`** |

> **Shape gotcha, verified**: because the `"rms"` branch reduces without
> `keepdim`, it only broadcasts correctly back against `wav` when `wav` has
> a single leading dim of size 1 (the common `[1, L]` mono case) — a
> genuinely multi-channel `wav: [C, L]` with `C > 1` raises a
> `RuntimeError` on `wav / den` (confirmed:
> `normalize_waveform(torch.randn(2, 16000), amp_type="rms")` fails; the
> same call with `amp_type="avg"`/`"peak"` succeeds, since their `keepdim=True`
> denominator is `[C, 1]` and broadcasts against any `C`). Every real corpus
> in this repo is mono, so this hasn't bitten anyone in practice, but don't
> feed genuine multi-channel audio through `amp_type="rms"`.

---

### `rescale_waveform(wav: Tensor, target_lvl: float, amp_type: str = "avg", scale: str = "linear") -> Tensor`

**Two steps, not one**: first `normalize_waveform(wav, amp_type)` (so the
signal sits at unit `amp_type` level), *then* multiplies by `target_lvl`
(converted from dB first if `scale="db"`: `target_lvl = 10**(target_lvl/20)`).
`amp_type` picks *which* amplitude measure `target_lvl` is relative to;
`scale` picks the units `target_lvl` is expressed in. Since it calls
`normalize_waveform` internally, the same `amp_type="rms"` + multi-channel
shape gotcha above applies here too.

The common call in this repo is RMS-referenced dB leveling (what
[`AudioIO.open`](io.md)'s `target_lvl` argument does):

```python
wav_m28dBFS = rescale_waveform(wav, target_lvl=-28.0, amp_type="rms", scale="dB")
```

`scale` defaults to `"linear"`, **not** `"dB"` — pass `scale="dB"` explicitly
for the dBFS use case above.

---

### `rand_gain_distortion(wav: Tensor, sample_rate: int = 16000, start_time: Optional[float] = None, duration: Optional[float] = None, return_info: bool = False) -> Tensor | Tuple[Tensor, Tuple]`

**Not a whole-signal gain scale** — multiplies a random *contiguous segment*
of `wav` by a random gain and clips the result to `[-1, 1]`. This is what
[`augmentation.apply_gain_distortion`](augmentation.md) exposes; there is no
min/max-gain range argument, everything is randomized internally:

- `distortion_gain = 4 ** N(0, 1)` — log-normal-ish, centered at unity gain,
  symmetric in log-space around `×4`/`÷4` per standard deviation.
- If `start_time`/`duration` (seconds) aren't given, the segment's start and
  length are each drawn uniformly over the whole remaining signal
  (`randint(0, len(wav))`, then `randint(0, len(wav) - start)`) — fully
  random placement.
- If given, both are "dithered" rather than used exactly: actual start/
  duration in samples is `int((value + Uniform(0, 1) * value) * sample_rate)`
  — so `start_time=0.1` can land anywhere in `[0.1s, 0.2s]`, i.e. a *lower
  bound* stretched by up to 2×, not an exact placement (verified: `start_time=0.1,
  duration=0.2` at `sample_rate=16000` produced an actual start of 1827
  samples and duration of 5366 samples, both inside their respective
  `[1×, 2×]` windows).

`return_info=True` returns `(distorted_wav, (start_sample, duration_samples,
gain))` so a caller can log or reapply the identical distortion.

---

### `wav_fade_in(wav: Tensor, sr: int, fade_len_s: int, fade_begin_s: int = 0, fade_shape: str = "linear") -> Tensor`
### `wav_fade_out(wav: Tensor, sr: int, fade_len_s: int, fade_begin_s: int = 0, fade_shape: str = "linear") -> Tensor`

Ramps amplitude from (fade-in) or to (fade-out) 0 over `fade_len_s` seconds,
starting at `fade_begin_s`. `fade_shape` (for fade-in; fade-out mirrors each
curve, `1 - fade`):

| shape | curve, `t = linspace(0, 1, fade_len_s * sr)` |
|---|---|
| `"linear"` | `t` |
| `"exponential"` | `2**(t - 1) * t` |
| `"logarithmic"` | `log10(0.1 + t) + 1` |

The ramp is concatenated with a block of ones before/after the fade window
and clamped to `[0, 1]`, then multiplied into `wav` — everything outside
`[fade_begin_s, fade_begin_s + fade_len_s]` is left untouched (multiplied by
exactly 1). Despite the `_s`-suffixed names being typed `int` in the
signature, they're used directly as seconds (`int(sr * fade_len_s)`), so
fractional values (e.g. `fade_len_s=0.01`) work fine in practice.

---

### `wav_clipping(wav: Tensor, min_quantile: float = 0.0, max_quantile: float = 0.9) -> Tensor`

Hard-clips `wav` at its own **empirical quantiles**, recomputed per call —
not a fixed dB threshold. `torch.quantile(wav, [min_quantile, max_quantile])`
gives `(min_, max_)`, then `torch.clip(wav, min_, max_)`.

> The default bounds are **asymmetric**: `min_quantile=0.0` is the signal's
> own minimum (a no-op floor — nothing gets clipped off the bottom by
> default), while `max_quantile=0.9` clips the top 10%. Verified: with the
> defaults, `clipped.min() == wav.min()` but `clipped.max() < wav.max()`.
> For symmetric clipping, pass e.g. `min_quantile=0.01, max_quantile=0.99`
> explicitly.

## Example

```python
from puresound.audio.volume import rescale_waveform, wav_fade_in, wav_clipping

wav_m28 = rescale_waveform(wav, target_lvl=-28.0, amp_type="rms", scale="dB")
wav_faded = wav_fade_in(wav_m28, sr=16000, fade_len_s=0.01, fade_shape="linear")
wav_clipped = wav_clipping(wav_faded, min_quantile=0.05, max_quantile=0.95)
```
