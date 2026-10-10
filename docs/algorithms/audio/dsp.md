# puresound.audio.dsp

繁體中文版本：[dsp.zh-TW.md](dsp.zh-TW.md)

Signal-processing primitives shared by the synthesis pipeline: a linearity
guard for backends that clip, sample-rate conversion, RBJ biquad design, a fixed
parametric EQ, and a compressor gain curve. All waveforms are `[..., L]`, time
last.

## `apply_linear(fn, wav, *, headroom=0.5, max_escalations=8)`

Runs a *linear* backend so that its built-in clipper can never fire.

The torchaudio filter backends saturate at digital full scale
(`FULL_SCALE = 1.0`): `torchaudio.functional.lfilter` clamps to [-1, 1] by
default, and every `*_biquad` is built on it without a way to turn that off. A
stage that models a linear operator (a microphone
response, a rumble filter, a gain) would silently become a waveshaper on a hot
signal. That matters most for mixture/target pairs: both go through the same
stage with the same parameters so their level relationship survives, and a
fixed ceiling squashes the louder mixture while the quieter target passes
untouched.

For a linear operator `H` and any `a > 0`, `H(x) = H(a·x) / a`. So:

1. `scale = headroom / peak(wav)`; call `fn(wav · scale)`.
2. If any output sample reaches `|y| ≥ 1 − 1e-6`, divide `scale` by 8 and retry.
3. Return `fn(wav · scale) / scale`; after `max_escalations` failed attempts
   raise `RuntimeError` (the operator has more gain than any linear stage in the
   chain should).

Silent or non-finite input is passed to `fn` unchanged. `fn` must not consume
randomness: a retry calls it again, which would shift the seeded RNG stream the
datasets rely on. Draw random parameters outside and close over them. Where a
backend can turn its ceiling off, do that instead (`lfilter(..., clamp=False)`).

**Used by:** `AudioEffectAugmentor.apply_hpf` (the device chain's rumble
filter) and `apply_media_coloring`.

## `wav_resampling(wav, origin_sr, target_sr, backend="sox", torch_backend_params=None)`

Sample-rate conversion with two deliberately different behaviours.

| `backend` | what runs | returns |
| --- | --- | --- |
| `"sox"` | `torchaudio.functional.resample` with a fixed Kaiser-windowed sinc (`lowpass_filter_width=64`, `rolloff≈0.9476`, `beta≈14.77`, the "kaiser_best" grade). Deterministic and near-transparent; sox itself is not called. | `(wav, target_sr)` |
| `"torchaudio"` | `torchaudio.transforms.Resample` with a **random** filter: `lowpass_filter_width ∈ {6, 16, 32, 64, 128}`, `rolloff ~ U(0.8, 0.99)`, `resampling_method ∈ {sinc_interp_hann, sinc_interp_kaiser}`, drawn from Python's `random` unless `torch_backend_params = {"lp_width", "rolloff", "window"}` fixes them. | `(wav, target_sr, params)` |

`"sox"` is the neutral converter: corpus loading, noise loading and RIR
resampling use it, and it must never draw randomness, or every file load would
become an uncontrolled augmentation. `"torchaudio"` is an augmentation that
models a cheap resampler; its returned `params` can be passed to the second leg
of a down-and-up round trip so both legs use one filter (see
`AudioEffectAugmentor.apply_src_effect` in [augmentation.md](augmentation.md)).

When `origin_sr == target_sr` nothing is computed; the return shape still
follows the backend (`(wav, sr)` or `(wav, sr, torch_backend_params or {})`).

The `"sox"` backend also returns a sample-less `wav` (`L == 0`) unchanged, at
`target_sr`; the polyphase kernel cannot process it, and callers treat an empty
waveform as unusable audio.

## `get_biquad_params(gain_dB, cutoff_freq, q_factor, sample_rate, filter_type) -> (b, a)`

RBJ Audio-EQ-Cookbook biquad design (R. Bristow-Johnson). With
`A = 10^(gain_dB/40)`, `ω0 = 2π·cutoff_freq/sample_rate` and
`α = sin ω0 / (2·q_factor)`, `filter_type` selects one of the cookbook
formulas:

| `filter_type` | filter | uses `gain_dB` |
| --- | --- | --- |
| `"low_shelf"`, `"high_shelf"` | shelving | yes |
| `"peaking"` | peaking EQ | yes |
| `"lpf"`, `"hpf"`, `"bpf"`, `"notch"` | low/high/band-pass (constant 0 dB peak gain), notch | no |

All five arguments are required. The result is two length-3 NumPy arrays
already divided by `a0`, so `a[0] == 1`. `puresound.nnet.lobe.dsp.FrequencyEQLayer`
calls it to seed its trainable EQ (see [lobe/dsp](../models/lobe/dsp.md)).

## `wav_apply_biquad_filter(wav, b_coeff, a_coeff) -> Tensor`

`scipy.signal.lfilter` applied channel by channel to a NumPy copy; a 1-D input
comes back as `[1, L]`. It does not clip (SciPy has no clamp), runs on CPU, and
is not differentiable.

## `ParametricEQ`

A fixed (not trainable, not an `nn.Module`) cascade: one low shelf, `N` peaking
bands, one high shelf, applied in that order by `forward(wav)`.

```python
ParametricEQ(sample_rate, eq_band_gain, eq_band_cutoff, eq_band_q_factor,
             low_shelf_gain_dB=0.0, low_shelf_cutoff_freq=80, low_shelf_q_factor=0.707,
             high_shelf_gain_dB=0.0, high_shelf_cutoff_freq=1000, high_shelf_q_factor=0.707,
             dtype=np.float32)
```

The three `eq_band_*` tuples must have equal length (one peaking band each).
`plot_eq(savefig=None)` draws `|H(f)| = Π_k |B_k(f)/A_k(f)|` with matplotlib; it
supports only `sample_rate` 16000 (512-point FFT) or 32000 (1024-point) and
raises `ValueError` otherwise.

## `compressor_gain(wav, sample_rate, *, threshold_db, ratio, attack_ms=5.0, release_ms=120.0, makeup=True) -> Tensor[1, T]`

The gain curve of a feed-forward compressor, **returned rather than applied**.

1. Detector: `|x|` smoothed by two one-pole filters with
   `a = 1 − exp(−1000 / (τ_ms · fs))` for the attack and release times, and
   `env = max(fast, slow)`. On an onset the fast pole dominates (the envelope
   rises at the attack rate); in a decay the slow pole dominates.
2. Static curve: `g_dB = −max(20·log10 env − threshold_db, 0) · (1 − 1/ratio)`.
3. Makeup: with `makeup=True` the linear gain is divided by its mean, so
   compression is not also a level change.

`ratio` must be ≥ 1 (1 = no compression); attack and release must be positive.
The input is flattened, so pass a mono `[1, T]` signal.

A compressor is a time-varying gain, and a gain distributes over a sum:
`g·(near + far) = g·near + g·far`. The caller derives the curve from the mixture
and multiplies it into both the mixture and the target, and the target stays the
near component of the compressed mixture. That is why it is returned, and why
the `|x|^p` waveshaper in `apply_media_coloring` is not used for this. The
max-of-two-poles detector has the shape of the usual switched-coefficient loop
and costs two vectorised filters instead of a per-sample Python loop.

**Used by:** `DeviceChain` compressor stage (see
[device chain](../augmentation/device_chain.md)).

## Example

```python
from puresound.audio.dsp import ParametricEQ, apply_linear, wav_resampling
import torchaudio

wav_16k, sr = wav_resampling(wav, origin_sr=48000, target_sr=16000)          # neutral

down, _, params = wav_resampling(wav_16k, 16000, 8000, backend="torchaudio")  # augmentation
up, _, _ = wav_resampling(down, 8000, 16000, backend="torchaudio", torch_backend_params=params)

hpf = apply_linear(lambda w: torchaudio.functional.highpass_biquad(w, 16000, 100.0), wav_16k)

eq = ParametricEQ(16000, eq_band_gain=(3.0,), eq_band_cutoff=(1000.0,),
                  eq_band_q_factor=(1.0,), high_shelf_cutoff_freq=7800)
wav_eq = eq.forward(wav_16k)
```
