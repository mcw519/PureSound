# puresound.audio.volume

繁體中文版本：[volume.zh-TW.md](volume.zh-TW.md)

Amplitude-domain utilities: RMS measurement, normalisation to a target level,
and three level effects (segment gain, fades, quantile clipping). All take
`wav: [..., L]` and reduce or act along the last axis only; outputs keep the
input shape.

## `calculate_rms(wav, to_log=False) -> Tensor`

`rms = sqrt(mean(wav², dim=-1))`, one value per leading index (`[C]` for
`[C, L]`, a 0-d tensor for `[L]`). `to_log=True` returns `20·log10(rms)`, the
level in dBFS. Silence gives `-inf` in the log form.

## `normalize_waveform(wav, amp_type="avg") -> Tensor`

Divides by one per-signal amplitude measure plus `1e-14`, with `keepdim=True`
so it broadcasts for any number of channels:

| `amp_type` | denominator |
| --- | --- |
| `"avg"` | `mean(abs(wav))` |
| `"peak"` | `max(abs(wav))` |
| `"rms"` | `sqrt(mean(wav²))` |

## `rescale_waveform(wav, target_lvl, amp_type="avg", scale="linear") -> Tensor`

`normalize_waveform(wav, amp_type) · g`, with `g = target_lvl` for
`scale="linear"` or `g = 10^(target_lvl/20)` for `scale="dB"` (case-insensitive).
The default scale is linear, so the dBFS levelling used across the repo is
spelled out explicitly:

```python
rescale_waveform(wav, target_lvl=-28.0, amp_type="rms", scale="dB")   # RMS at -28 dBFS
```

This is what [`AudioIO.open(..., target_lvl=...)`](io.md) calls.

## `rand_gain_distortion(wav, sample_rate=16000, start_time=None, duration=None, return_info=False)`

Multiplies one contiguous segment by a random gain and hard-clips the whole
signal to [-1, 1]; it models a momentary gain jump followed by converter
saturation, not a whole-signal level change.

- Gain: `4^z`, `z ~ N(0, 1)` from Python's `random`, i.e. log-normal around
  unity with a standard deviation of `20·log10 4 ≈ 12 dB`.
- Segment start, in samples: `randint(0, L)` when `start_time` is `None`,
  otherwise `int(start_time·(1 + u)·sample_rate)` with `u ~ U(0, 1)`, so the
  start lands in `[start_time, 2·start_time)` seconds.
- Segment length: `randint(0, L − start)` when `duration` is `None`, otherwise
  `int(duration·(1 + u)·sample_rate)`, dithered the same way.

`return_info=True` returns `(wav, (start_sample, duration_samples, gain))` so
the same distortion can be logged or reapplied. Exposed to recipes as
`AudioEffectAugmentor.apply_gain_distortion` (see [augmentation.md](augmentation.md)).

## `wav_fade_in(wav, sr, fade_len_s, fade_begin_s=0, fade_shape="linear")` / `wav_fade_out(...)`

Multiplies `wav` by an envelope that is 1 everywhere except on the window
`[fade_begin_s, fade_begin_s + fade_len_s)` seconds, where it follows one of
three curves of `t = linspace(0, 1, int(sr·fade_len_s))`, clamped to [0, 1]:

| `fade_shape` | fade-in | fade-out |
| --- | --- | --- |
| `"linear"` | `t` | `1 − t` |
| `"exponential"` | `2^(t−1)·t` | `2^(−t)·(1 − t)` |
| `"logarithmic"` | `log10(0.1 + t) + 1` | `log10(1.1 − t) + 1` |

Each fade-out curve is its fade-in curve reversed in time. The arguments are
seconds and may be fractional despite the `int` annotation. The window must fit
inside the signal (`fade_begin_s + fade_len_s ≤ L / sr`); otherwise the envelope
length check fails.

## `wav_clipping(wav, min_quantile=0.0, max_quantile=0.9) -> Tensor`

Hard-clips at the signal's own empirical quantiles:
`torch.clip(wav, quantile(wav, min_quantile), quantile(wav, max_quantile))`,
recomputed on every call. The clipping level is therefore relative to the
signal, not a fixed dBFS threshold. The defaults are asymmetric: `0.0` is the
minimum sample (nothing is clipped from below) while `0.9` clips the top 10 %
of samples. Pass mono `[1, L]` or `[L]`; per-channel bounds of a `[C, L]` input
with `C > 1` do not broadcast.

`clipping_thresholds(wav, min_quantile, max_quantile)` returns the two
levels `(lo, hi)` without clipping, so a second signal can be clipped at the
first one's levels; `DeviceChain` does this for the target (see
[level and dynamics](../augmentation/level_dynamics.md)).

## Example

```python
from puresound.audio.volume import rescale_waveform, wav_clipping, wav_fade_in

wav = rescale_waveform(wav, target_lvl=-28.0, amp_type="rms", scale="dB")
wav = wav_fade_in(wav, sr=16000, fade_len_s=0.01)
wav = wav_clipping(wav, min_quantile=0.01, max_quantile=0.99)   # symmetric
```
