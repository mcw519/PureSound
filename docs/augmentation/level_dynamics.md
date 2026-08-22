# Level and dynamics

繁體中文版本：[`level_dynamics.zh-TW.md`](level_dynamics.zh-TW.md)

This chapter covers everything in the pipeline concerned with loudness: the
three level metrics, the derivation of SNR/SIR mixing, gain distortion,
clipping, fades, and the one time-varying dynamics technique — the dynamic
range compressor.

## Algorithm

### 1. Three level metrics

`puresound/audio/volume.py` provides three amplitude metrics with different
uses:

```
rms(x)  = sqrt( mean(x²) )      # energy level
avg(x)  = mean(|x|)             # mean absolute amplitude
peak(x) = max(|x|)              # peak
```

**How to choose**:

* **RMS** relates directly to signal energy and underlies every SNR/SIR
  computation. For non-stationary signals such as speech it reflects the mean
  energy over the whole segment.
* **Mean absolute amplitude** (`avg`) is `AudioIO`'s default normalisation
  basis. Being a first-moment rather than second-moment statistic, it is less
  sensitive to large peaks and therefore more stable on signals containing
  sporadic large impulses.
* **Peak** has a well-defined meaning only in the digital domain, where it
  corresponds to full scale and clipping risk. It is the only metric used at
  the A/D boundary ([ch6](device_chain.md)).

The ratios between them carry information of their own: `peak/rms` is the crest
factor, describing how dynamic a signal is. Speech typically has a crest factor
of 12–18 dB, which drops after compression — one of the mechanisms by which the
compressor of §6 affects near/far readout.

`normalize_waveform` divides by one of the three (with `1e-14` added to avoid
division by zero); `rescale_waveform` normalises first and then multiplies by
the target level (linear, or dB converted as `10^(L/20)`).

**Level flattening at load time**: `AudioIO.open` rescales every loaded
utterance to `target_lvl` (the recipe's `audio_gain_normalized_to`, in dB RMS).
All sources therefore enter the synthesis flow at a uniform level, which is the
first of the two causes behind "the level cue is removed" in
[ch3](distance_cues.md) §1.1.

### 2. Deriving SNR / SIR mixing

SIR (foreground talker against interfering talkers) and SNR (speech against
noise) share one implementation
(`puresound/audio/noise.py::add_bg_noise`) and identical mathematics.

#### 2.1 Statement and solution

Given speech `s`, noise `n`, and a target SNR in dB, find the scale factor `α`
such that

```
20 · log10( rms(s) / rms(α·n) ) = SNR_dB
```

Since `rms(α·n) = α · rms(n)`:

```
rms(s) / (α · rms(n)) = 10^(SNR_dB/20)
α = rms(s) / ( rms(n) · 10^(SNR_dB/20) )
```

The implementation RMS-normalises the noise first (`rms(n) = 1`), reducing this
to

```
α = rms(s) / 10^(SNR_dB/20)
y = s + α · n
```

#### 2.2 Preprocessing details

The order of operations for multiple noise clips is worth noting: each clip is
RMS-normalised **individually**, they are concatenated, and then the
concatenation is RMS-normalised **again**. The second normalisation is
necessary — two unit-RMS signals concatenated only have unit RMS overall when
they are of equal length, so unequal lengths require renormalisation to
guarantee `rms(n) = 1`.

Length matching: noise longer than the speech gets a randomly positioned crop
(increasing the variety extracted from one clip); noise shorter than the speech
is tiled cyclically to sufficient length and then cropped.

#### 2.3 Three semantic details

**(a) RMS is computed over the whole row, silence included.** This determines
what SNR actually means. If speech occupies only 2 of a 6-second row, `rms(s)`
is pulled down by roughly `10·log10(3) ≈ 4.8 dB`, so the "SNR while speaking"
is about 4.8 dB higher than the nominal value.

This is not a bug, but it makes the relationship between nominal SNR and
perceptual difficulty depend on speech density. Changing it to an
active-speech RMS (computed only on speech frames) would shift the effective
SNR distribution of every recipe, which amounts to redefining corpus
difficulty — a distribution break
([ch8](engineering_contract.md)).

**(b) SIR is defined on the post-gating signal.** Overlap gating
([ch7](scene_construction.md)) runs before mixing, so large stretches of the
interferers are already silenced. `add_bg_noise` receives the gated signal, so
SIR is defined against "the total energy the interferer actually produced"
rather than against the original utterance. The stronger the gating, the louder
the interferer's instantaneous level while speaking, for the same nominal SIR.

**(c) The white-noise path's SNR is relative to the current mixture.** White
noise is added after recorded noise, so its reference `rms(s)` is "the mixture
including recorded noise", not the original speech.

#### 2.4 White noise generation

```
σ = rms(s) / 10^(SNR_dB/20)
n[k] ~ N(0, σ²)
```

The RMS of a Gaussian distribution equals its standard deviation, so setting σ
to the target RMS directly is sufficient; no further normalisation is needed.

### 3. Gain distortion (`rand_gain_distortion`)

Models an AGC jump or a recording-level accident: a randomly positioned,
randomly long interval is multiplied by a gain.

```
g = 4^z,    z ~ N(0, 1)
y = clip(x · mask, −1, 1)
```

#### 3.1 The design of the gain distribution

`4^z` is normally distributed in the dB domain:

```
g_dB = 20·log10(4^z) = z · 20·log10(4) ≈ z · 12.04
```

so `g_dB ~ N(0, 12²)`. The meaning of this parameterisation: the median is
0 dB (unchanged), roughly 68% of rows lie within ±12 dB and 95% within ±24 dB.
Using a normal distribution in the dB domain is the convention for gain
perturbation — perception of loudness is approximately logarithmic, so only a
distribution symmetric in dB represents "equally likely to get louder or
quieter".

#### 3.2 The clip here is deliberate

`clip(±1)` inside this function is **part of the distortion model**: a segment
whose gain was pushed too far genuinely clips. This differs in kind from the
unintentional saturation that `apply_linear`
([ch5](spectral_channel.md)) protects against — that is a linear operator being
contaminated by a backend's implicit limiter, a defect; this is a modelled
phenomenon.

### 4. Clipping (`wav_clipping`)

Quantile-based hard clipping:

```
lo = Q_{min_quantile}(x),   hi = Q_{max_quantile}(x)
y = clip(x, lo, hi)
```

#### 4.1 Why quantiles rather than absolute thresholds

The effect of an absolute threshold (say, always clipping at ±0.8) depends on
the signal's level: a quiet signal peaking at 0.3 is untouched, while one
peaking at 1.0 is heavily clipped. The same parameter produces wildly different
amounts of distortion on different rows.

A quantile threshold adapts to each row's amplitude distribution:
`max_quantile = 0.9` means "clip the largest 10% of samples" regardless of how
loud the row is overall. The parameter therefore maps directly onto the
*amount* of distortion rather than the absolute position of the threshold. The
approach is taken from the URGENT challenge data simulation flow.

#### 4.2 How the target is clipped

In the device chain the target is not clipped at its own quantiles but at the
**threshold values the mixture actually produced** (the absolute `lo` and `hi`).
The reason is physical: analogue overload happens at the microphone output and
the threshold is an absolute voltage set by the circuit, not a per-signal
statistic. Mixture and target went through the same overload, so the threshold
is the same for both. Details in [ch6](device_chain.md).

### 5. Fades (`wav_fade_in` / `wav_fade_out`)

Implemented as gain envelopes. With `f` rising linearly from 0 to 1 over
`fade_len_s · sr` samples:

| Shape | Fade-in gain | Fade-out gain |
|---|---|---|
| linear | `f` | `1 − f` |
| exponential | `2^(f−1) · f` | `2^(−f) · (1 − f)` |
| logarithmic | `log10(0.1 + f) + 1` | `log10(1.1 − f) + 1` |

The envelope is padded with ones before and after and clamped to [0, 1].

The shapes differ in how energy is distributed over time: linear is uniform in
amplitude; logarithmic rises quickly at the start (approximately linear in dB,
which sounds more even); exponential rises slowly at the start, suiting a
gradual fade-in.

### 6. Dynamic range compressor (`compressor_gain`)

Models "this recording went through a compressor" — routine in broadcast,
conferencing, and publication chains.

#### 6.1 It returns a gain curve, not a compressed signal

The function returns a `[1, T]` gain curve for the caller to apply. This
interface exists so that the same curve can be applied to both mixture and
target; the reason is in §6.5.

#### 6.2 Envelope detector

A compressor must first track the signal's current loudness. The
implementation uses an asymmetric one-pole peak follower:

```
a_att = 1 − exp(−1000 / (attack_ms  · fs))
a_rel = 1 − exp(−1000 / (release_ms · fs))

one_pole(x, a):  y[n] = a·x[n] + (1 − a)·y[n−1]

env[n] = max( one_pole(|x|, a_att), one_pole(|x|, a_rel) )
```

**Deriving the time constants.** The step response of
`y[n] = a·x[n] + (1−a)·y[n−1]` is `y[n] = 1 − (1−a)^n`, and its equivalent time
constant τ satisfies `(1−a)^{fs·τ} = e^{−1}`, giving
`a = 1 − exp(−1/(fs·τ))`. Converting τ from seconds to milliseconds yields the
factor of 1000 above.

**Why take the maximum of two poles.** A standard compressor detector decides
per sample: use the fast attack coefficient when the signal exceeds the current
envelope, the slow release coefficient when it falls below. The implementation
instead takes the pointwise maximum of a fast and a slow pole, which is
equivalent in shape:

* On an onset the fast pole rises faster than the slow one, so `max` selects it
  and the envelope rises at the attack rate.
* During a decay the slow pole falls more slowly than the fast one, so `max`
  selects it and the envelope falls at the release rate.

The two are not the same curve (correlation about 0.93, not 1.0). The
approximation is chosen for performance: the per-sample switching loop cannot
be vectorised and takes about 2.4 s for six seconds of audio, accumulating to
hundreds of minutes per epoch inside the dataloader; `max(fast, slow)` uses two
`lfilter` calls and takes about 12 ms.

The exact detector formulation is a design degree of freedom (real hardware
differs unit to unit). The properties the augmentation genuinely needs — fast
response to onsets, no pumping during decay — are asserted by tests rather than
obtained by copying one particular implementation.

#### 6.3 Gain computation

The standard above-threshold compression curve, applied in the dB domain:

```
E_dB[n] = 20 · log10(env[n])
over[n] = max(0, E_dB[n] − T)              # amount above threshold
g_dB[n] = −over[n] · (1 − 1/R)
g[n]    = 10^(g_dB[n]/20)
```

**Meaning of the ratio R.** When the input exceeds the threshold by `over` dB,
the output should exceed it by only `over/R` dB, so the required gain reduction
is `over − over/R = over·(1 − 1/R)`. Limiting behaviour: `R = 1` gives
`g_dB = 0` (no compression); `R → ∞` gives `g_dB = −over` (a limiter, output
pinned at the threshold).

#### 6.4 Makeup normalisation

```
g ← g / mean(g)
```

This normalises the mean of the gain curve to 1. **Without it, compression and
"turning the whole thing down" are indistinguishable in the data**: a
compressor's mean gain is necessarily below 1, so the model could learn merely
"this row is quieter" rather than "this row's dynamics were compressed". After
normalisation the only observable effect of compression is the change in
envelope shape.

Note that normalisation guarantees `mean(g) = 1` but not `max(g) ≤ 1` — quiet
passages can be pushed up to gains above 1, so a compressed signal's peak may
exceed the original's. The downstream A/D boundary
([ch6](device_chain.md)) handles anything above full scale.

#### 6.5 Why a compressor may be applied to the target and a waveshaper may not

This is the most important design principle in this chapter, and it dictates
that "the recording was compressed" and "there is a television in the room"
must be modelled by two different mechanisms.

A compressor's output is the pointwise product of a time-varying gain with the
signal, and multiplication distributes over addition:

```
g[n] · (near[n] + far[n]) = g[n]·near[n] + g[n]·far[n]
```

So after applying the same `g` to both mixture and target, the target is still
exactly the near-field component of the compressed mixture. Superposition
survives and the training pair's relationship is unchanged.

Contrast the `|x|^p` waveshaper in `apply_media_coloring`
([ch5](spectral_channel.md) §6):

```
(near + far)^p ≠ near^p + far^p
```

A waveshaper admits no per-source decomposition. It is the correct model for
"that television in the room" — the distortion happens before mixing, acting on
a single source — but the wrong model for "the whole recording was compressed",
because a target following it could only be independently distorted into
something that is no longer a component of the mixture.

#### 6.6 The curve is derived from the mixture

`compressor_gain` takes the **mixture** as input. In a real chain the
compressor sits after the microphone and the mixture is what it sees. Deriving
the curve from the target instead would produce an effect no real device
performs (a compressor cannot "see" only the near-field talker).

The motivation for this stage: publication and conferencing chains compress
routinely, and envelope structure is one component of near/far readout (through
the crest factor, §1), so covering "a compressed world" in the training
distribution is the data-layer way to address it. Sources for the measurements
behind that motivation are cited in the `CompressorAugmentation` docstring and
in `egs/voice_isolate/benchmarks/`.

## Engineering

### Config mapping

| Knob | Schema | Where it lands |
|---|---|---|
| `augmentation_volume` | `VolumeAugmentation` (`perturbed_range`, `clipping_prob`, `clipping_range.min/max`) | `DeviceChain._volume`: one draw picks the gain or clipping branch |
| `augmentation_compressor` | `CompressorAugmentation` | `DeviceChain._compressor` |
| `augmentation_noise.snr_range` etc. | `NoiseAugmentation` | `NoiseStage` ([ch7](scene_construction.md)) |
| `augmentation_speech.snr_range` | `SpeechAugmentation` | Hard-SIR draw ([ch7](scene_construction.md)) |

Schema constraints on `CompressorAugmentation`: the lower bound of
`ratio_range` must be ≥ 1.0 (1.0 means no compression; below 1 is an expander,
not what this stage models), and the lower bound of `attack_ms_range` must
be > 0 (0 divides by zero in the time-constant formula).

### Ordering

The compressor's position in the device chain is "after the preamp, before the
converter", matching a real chain. It has to stay inside the linear group
([ch6](device_chain.md)) because the target follows it.

Each time `_volume` fires it takes either the gain or the clipping branch
(decided by `clipping_prob`), never both.

### Pitfalls

* The compressor curve and the signals are aligned by
  `min(gain, noisy, target)` before multiplying. The SRC stage can leave a 1–2
  sample length difference; samples past the end keep their original values,
  which is expected.
* `rand_gain_distortion` currently has only an augmentor API (used by TSE). The
  level path for NS and voice isolation is the device chain's `_volume` stage.
  Their gain distributions differ (`N(0, 12dB²)` versus
  `U(perturbed_range)`), so do not conflate them.
* The RMS semantics of §2.3(a): changing them causes a distribution break.
  Confirm you are willing to pay for retrained controls before doing so.
