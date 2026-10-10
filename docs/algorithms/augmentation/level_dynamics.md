# Level and dynamics

繁體中文版本：[level_dynamics.zh-TW.md](level_dynamics.zh-TW.md)

Everything in synthesis that concerns loudness: the level measures, SNR and SIR
mixing, the paired peak guards, gain and clipping distortion, fades, and the
dynamic-range compressor. The functions live in `puresound/audio/volume.py`,
`puresound/audio/noise.py` and `puresound/audio/dsp.py`; the device-chain stages
that call them are in [Device chain](device_chain.md).

## 1. Level measures

`volume.py` provides three amplitude measures over the last axis:

```
rms(x)  = sqrt(mean(x²))      energy level
avg(x)  = mean(|x|)           mean absolute amplitude
peak(x) = max(|x|)            peak
```

- **RMS** is tied to energy and underlies every SNR and SIR computation.
- **avg** is a first-moment statistic, less sensitive to isolated large peaks.
- **peak** only has a physical meaning in the digital domain, where it is the
  distance to full scale. The peak guards (§3) use it.

`peak/rms` is the crest factor, a measure of how dynamic a signal is; speech
typically sits around 12–18 dB, and compression lowers it (§7).

`normalize_waveform(wav, amp_type)` divides by one of the three (`amp_type` in
`rms`, `avg`, `peak`, with `1e-14` added to the denominator);
`rescale_waveform(wav, target_lvl, amp_type, scale)` normalises and then
multiplies by `target_lvl` (linear, or dB converted as `10^(L/20)`).

**Level at load time.** When the recipe sets `dataset.gain_normalized_to` (in dB,
for example `-28`), every utterance is loaded with
`AudioIO.open(target_lvl=...)`, which rescales it to that RMS level. All sources
then enter synthesis at one level, which removes the source's own loudness as a
cue ([Distance cues](distance_cues.md)). With the field left empty, utterances
keep their file level.

## 2. SNR and SIR mixing

Speech-to-noise (SNR) and foreground-to-interferer (SIR) mixing share one
function, `noise.add_bg_noise(wav, noise, snr_list)`, and one derivation.

### 2.1 Scale factor

For speech `s`, noise `n` and a target ratio in dB, find `α` with

```
20·log10( rms(s) / rms(α·n) ) = SNR_dB
α = rms(s) / ( rms(n) · 10^(SNR_dB/20) )
```

The noise is RMS-normalised first (`rms(n) = 1`), so `α = rms(s) / 10^(SNR_dB/20)`
and `y = s + α·n`. The function returns one mixture and one scaled noise per
entry of `snr_list`.

### 2.2 Preparing the noise

- Multichannel noise keeps its first channel.
- Several noise clips are each RMS-normalised, concatenated, and normalised
  again. The second pass is needed because two unit-RMS clips of unequal length
  do not concatenate to unit RMS.
- Noise longer than the speech is cropped at a random offset; shorter noise is
  tiled and cropped.

When recorded noise fires, `NoiseStage` passes two clips ("dynamic" noise) with
probability `prob / 4` and one clip otherwise
([Scene construction](scene_construction.md)).

### 2.3 What the ratio means

- **RMS is over the whole row, silence included.** If speech fills 2 s of a 6 s
  row, `rms(s)` is about `10·log10(3) ≈ 4.8 dB` below the level while speaking,
  so the SNR during speech is about 4.8 dB higher than the nominal value. The
  relation between nominal SNR and difficulty therefore depends on speech
  density. Switching to active-speech RMS would move the effective SNR of every
  recipe, which is a change of distribution
  ([Engineering contract](engineering_contract.md)).
- **SIR is defined after overlap gating.** Gating silences parts of each
  interferer before mixing, so the SIR refers to the energy the interferer
  actually produced; the more it is gated, the louder it is while it talks, for
  the same nominal SIR.
- **White-noise SNR is relative to the current mixture.** White noise is added
  after recorded noise, so its reference is the mixture including that noise.

### 2.4 White noise

`noise.add_bg_white_noise(wav, snr_list)`:

```
σ = rms(s) / 10^(SNR_dB/20),   n[k] ~ N(0, σ²)
```

The RMS of zero-mean Gaussian noise equals its standard deviation, so no further
normalisation is needed.

## 3. Paired peak guards

Two places divide mixture and target by one shared peak, a linear rescale that
keeps their level relationship and superposition:

- `DynamicBaseDataset.avoid_audio_clipping(wav_list)` runs in `ns.py` after the
  foreground/interferer mix and echo, before speed perturbation: if the largest
  peak over the list exceeds 1, every signal is divided by it.
- `DeviceChain._analogue_to_digital` does the same at the converter
  ([Device chain](device_chain.md)).

Neither clips; deliberate overload is only the clipping branch of the volume
stage.

## 4. Gain distortion

`volume.rand_gain_distortion(wav, sample_rate, start_time=None, duration=None)`
models an AGC jump or a level accident: a random interval is multiplied by one
gain and the result clipped.

```
g = 4^z,  z ~ N(0, 1)
y = clip(x · mask, −1, 1),   mask = g on the interval, 1 elsewhere
```

`g_dB = 20·log10(4)·z ≈ 12.04·z`, so the gain is normal in dB with median 0 dB,
68 % within ±12 dB and 95 % within ±24 dB; a distribution symmetric in dB is the
one that makes louder and quieter equally likely. Without explicit
`start_time`/`duration` the interval is uniform over the clip. The clip is part
of the model: a segment pushed too far does clip. This is a different thing from
the backend saturation `apply_linear` removes
([Spectral and channel effects](spectral_channel.md)).

It is reachable as `AudioEffectAugmentor.apply_gain_distortion`; no task
pipeline calls it. The device chain's gain is the volume stage's
`g ~ U(perturbed_range)` instead.

## 5. Clipping

`volume.wav_clipping(wav, min_quantile=0.0, max_quantile=0.9)` clips at sample
quantiles of the input:

```
lo = Q_min_quantile(x),  hi = Q_max_quantile(x),  y = clip(x, lo, hi)
```

A fixed absolute threshold would do nothing to a quiet row and ruin a loud one.
A quantile threshold maps the parameter onto the *amount* of distortion:
`max_quantile = 0.9` clips the top 10 % of samples at any level. The approach
follows the URGENT challenge simulation. The function expects mono input
(`[L]` or `[1, L]`).

In the device chain the volume stage draws `min_q ~ U(clipping_range.min)` and
`max_q ~ U(clipping_range.max)` and clips the mixture at its quantiles; by
default the target is clipped at the mixture's thresholds, and
`target_clipping: own_quantile` clips it at its own quantiles instead
([Device chain](device_chain.md)).

## 6. Fades

`volume.wav_fade_in` and `volume.wav_fade_out(wav, sr, fade_len_s,
fade_begin_s=0, fade_shape="linear")` multiply by a gain envelope. With `f`
rising linearly from 0 to 1 over `fade_len_s · sr` samples:

| Shape | Fade in | Fade out |
|---|---|---|
| `linear` | `f` | `1 − f` |
| `exponential` | `2^(f−1) · f` | `2^(−f) · (1 − f)` |
| `logarithmic` | `log10(0.1 + f) + 1` | `log10(1.1 − f) + 1` |

The envelope is padded with ones before and after and clamped to [0, 1];
`fade_begin_s + fade_len_s` must fit in the signal. Logarithmic rises fast at the
start (close to linear in dB), exponential slowly. These are library utilities;
no task pipeline calls them.

## 7. Dynamic-range compressor

`dsp.compressor_gain(wav, sample_rate, *, threshold_db, ratio, attack_ms=5.0,
release_ms=120.0, makeup=True)` models "the recording went through a
compressor", as broadcast, conferencing and publication chains do. It returns the
gain curve, `[1, T]`, not the compressed signal.

### 7.1 Envelope detector

An asymmetric one-pole peak follower on `|x|`:

```
a_att = 1 − exp(−1000 / (attack_ms  · fs))
a_rel = 1 − exp(−1000 / (release_ms · fs))
one_pole(x, a):  y[n] = a·x[n] + (1 − a)·y[n−1]
env[n] = max( one_pole(|x|, a_att), one_pole(|x|, a_rel) )
```

The step response of the one-pole filter is `1 − (1−a)^n`; setting
`(1−a)^(fs·τ) = e^−1` gives `a = 1 − exp(−1/(fs·τ))`, and the 1000 converts
milliseconds.

A textbook detector switches per sample between the attack and release
coefficients. The pointwise maximum of a fast and a slow pole has the same shape:
on an onset the fast pole is higher, so the envelope rises at the attack rate;
during a decay the slow pole is higher, so it falls at the release rate. It is
not the same curve, but it is two vectorised `lfilter(..., clamp=False)` calls
instead of a per-sample Python loop in the data loader. The properties the
augmentation needs (fast onset response, no pumping on decay) are pinned by
tests rather than by copying one hardware design.

### 7.2 Gain

```
E_dB[n] = 20·log10(env[n])
over[n] = max(0, E_dB[n] − T)
g_dB[n] = −over[n] · (1 − 1/R)
g[n]    = 10^(g_dB[n]/20)
```

An input `over` dB above threshold should leave the compressor `over/R` dB above
it, so the reduction is `over·(1 − 1/R)`. `R = 1` is no compression; `R → ∞` is a
limiter. `ratio < 1` (an expander) and non-positive attack or release raise
`ValueError`.

### 7.3 Makeup

With `makeup=True`, `g ← g / mean(g)`. A compressor's mean gain is below 1, so
without this, compression and "turn the row down" would be the same
augmentation and the model could learn either. After normalisation the only
observable effect is the change in envelope shape. `mean(g) = 1` does not bound
`max(g)`, so quiet passages can be raised and the peak can exceed the input's;
the converter stage downstream handles anything above full scale.

### 7.4 Why the target may follow a compressor

A compressor is a time-varying gain, and a gain distributes over a sum:

```
g[n]·(near[n] + far[n]) = g[n]·near[n] + g[n]·far[n]
```

With the same `g` on mixture and target, the target is still exactly the near
component of the compressed mixture. The `|x|^p` waveshaper of media coloring
([Spectral and channel effects](spectral_channel.md)) has no such
decomposition, so it can only model a device in the room before mixing, never a
compressed recording.

The curve is computed from the **mixture**, which is what a real compressor
after the microphone sees; a curve from the target alone would be an effect no
device performs.

The stage exists because flattening the envelope changes the direct-to-reverberant
ratio the model reads off a recording, so a compressed distant talker can look
like a near one, whereas absolute level does not move that estimate. A plain gain
stage cannot stand in for it.

## Configuration

| Knob | Schema | Where it applies |
|---|---|---|
| `dataset.gain_normalized_to` | `DatasetConfig` | `AudioIO.open(target_lvl=...)` at load |
| `augmentation_volume` | `VolumeAugmentation`: `perturbed_range` (linear gain), `clipping_prob`, `clipping_range.min`, `clipping_range.max` (quantile ranges), `target_clipping` | `DeviceChain._volume`: one draw picks gain or clipping, never both |
| `augmentation_compressor` | `CompressorAugmentation`: `threshold_db_range` (dB re amplitude 1), `ratio_range`, `attack_ms_range`, `release_ms_range` (ms) | `DeviceChain._compressor` |
| `augmentation_noise` | `NoiseAugmentation` (`snr_range`, `snr_bands`, `prob_white_noise`, `white_noise_snr_range`, ...) | `NoiseStage` ([Scene construction](scene_construction.md)) |
| `augmentation_speech.snr_range` | `SpeechAugmentation` | the foreground/interferer SIR draw |

`CompressorAugmentation` requires the lower bound of `ratio_range` to be ≥ 1 and
of `attack_ms_range` to be > 0.

```yaml
augmentation_compressor:
  used: True
  prob: 0.35
  threshold_db_range: [-34.0, -22.0]
  ratio_range: [1.5, 6.0]
  attack_ms_range: [2.0, 15.0]
  release_ms_range: [60.0, 300.0]
```

## Notes

- The compressor sits after the preamp and before the converter, inside the
  linear group, because the target follows it.
- The compressor curve and the signals are multiplied over their shortest common
  length; any uncovered tail samples keep their values.
- Changing the RMS convention of §2.3 changes the synthesis distribution.
