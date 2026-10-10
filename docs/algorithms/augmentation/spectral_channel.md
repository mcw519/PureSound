# Spectral and channel effects

繁體中文版本：[spectral_channel.zh-TW.md](spectral_channel.zh-TW.md)

Microphones have a frequency response, devices have rumble filters, and
transmission paths pass through resamplers. This page covers the linear filters
and resamplers used in synthesis, speed and pitch perturbation, media coloring
(one deliberate nonlinearity), and `apply_linear`, which keeps the linear stages
linear. The code is in `puresound/audio/dsp.py`,
`puresound/audio/impulse_response.py` and `puresound/audio/augmentation.py`; the
device-chain stages that call it are described in [Device chain](device_chain.md).

## 1. Biquads

### 1.1 Transfer function

A biquad is a second-order IIR filter:

```
H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (a0 + a1·z⁻¹ + a2·z⁻²)
y[n] = b0·x[n] + b1·x[n−1] + b2·x[n−2] − a1·y[n−1] − a2·y[n−2]     (a0 = 1)
```

Second order is the smallest filter that gives one resonance or one transition
band: a complex-conjugate pole pair sets frequency and Q, the two zeros shape the
stopband. Higher-order responses are cascades of biquads.

### 1.2 RBJ design

`dsp.get_biquad_params(gain_dB, cutoff_freq, q_factor, sample_rate, filter_type)`
implements the RBJ Audio EQ Cookbook and returns `(b, a)` as length-3 NumPy
arrays normalised by `a0`. `filter_type` is one of `high_shelf`, `low_shelf`,
`peaking`, `lpf`, `hpf`, `bpf`, `notch`. The intermediate variables are

```
A  = 10^(gain_dB / 40)       square root of the amplitude gain
w0 = 2π · fc / fs            normalised angular frequency, rad/sample
α  = sin(w0) / (2·Q)         bandwidth parameter
```

`A` uses 40 rather than 20 because it enters the shelf and peaking formulas in
both numerator and denominator (for example `1 + α·A` over `1 + α/A`), so the
response gain is `A²`. `gain_dB` is ignored by `lpf`, `hpf`, `bpf` and `notch`.

`Q = f0 / Δf` (centre frequency over −3 dB bandwidth). `Q = 1/√2 ≈ 0.707` is the
Butterworth value, maximally flat with no overshoot; `Q = 0.5` is critically
damped; `Q > 1` peaks near the cutoff.

Three representative coefficient sets:

```
LPF:      b = [(1 − cos w0)/2,  1 − cos w0,  (1 − cos w0)/2]      a = [1 + α, −2·cos w0, 1 − α]
HPF:      b = [(1 + cos w0)/2, −(1 + cos w0), (1 + cos w0)/2]     a = [1 + α, −2·cos w0, 1 − α]
peaking:  b = [1 + α·A, −2·cos w0, 1 − α·A]                       a = [1 + α/A, −2·cos w0, 1 − α/A]
```

LPF and HPF share their poles and differ only in where the zeros sit (Nyquist
versus DC). In the peaking filter `gain_dB → −gain_dB` swaps numerator and
denominator, so a cut exactly inverts a boost.

The cookbook formulas come from analogue prototypes through the bilinear
transform, `Ω = 2·tan(ω/2)`. They pre-warp at `w0`, so the cutoff lands where
specified, but bandwidth and gain deviate from the analogue design near Nyquist.
In the speech band at 16 kHz the deviation is negligible.

### 1.3 `ParametricEQ`

`dsp.ParametricEQ` cascades a low shelf, N peaking bands and a high shelf,
`H(z) = Π H_i(z)`, applied stage by stage with `scipy.signal.lfilter`. In a
cascade the dB responses add, which is how an equaliser is expected to behave;
a parallel sum would let the band phases interfere at the crossovers. It is a
fixed, non-differentiable utility and no synthesis stage uses it. The trainable
`puresound.nnet.lobe.dsp.FrequencyEQLayer` seeds its weights from the same
`get_biquad_params` shelf and peaking designs. `ParametricEQ.plot_eq` supports
only 16 kHz and 32 kHz.

## 2. Random second-order response (transducer)

`impulse_response.rand_add_2nd_filter_response(wav, a=None, b=None)` models the
fact that every microphone has its own frequency response, following the
augmentation in *A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band
Speech Enhancement* (Valin, RNNoise):

```
r0..r3 ~ U(−3/8, 3/8)
a = [1, r0, r1]      poles
b = [1, r2, r3]      zeros
```

It returns `(wav, a, b)`; passing `a` and `b` reuses a response, which is how
the target follows the mixture.

**Stability without a check.** A second-order denominator `1 + a1·z⁻¹ + a2·z⁻²`
has its poles inside the unit circle when `|a2| < 1` and `|a1| < 1 + a2`. With
`a1, a2 ∈ [−3/8, 3/8]` the first holds trivially and the second because
`1 + a2 ≥ 5/8 > 3/8`. Every draw is stable, so no rejection sampling is needed.

**Gain range.** `|B| ≤ 1.75` and `|A| ≥ 0.25` on the unit circle, so the
response is bounded by `20·log10(7) ≈ +16.9 dB` (reached at DC at the corners of
the range). The median peak over the distribution is about +3.5 dB. The upper
tail is why `apply_linear` (§7) has an escalation loop.

**`clamp=False`.** The filter runs as
`torchaudio.functional.lfilter(..., clamp=False)`. `lfilter` clamps its output to
[−1, 1] by default, which on a hot mixture is a waveshaper that hits the mixture
but not the quieter target. Where a backend exposes the switch, turning it off
is exact and free.

## 3. High-pass (rumble filter)

`AudioEffectAugmentor.apply_hpf(wav, sr, cutoff_freq, q_factor)` runs
`torchaudio.functional.highpass_biquad` (an RBJ high-pass) through
`apply_linear`, because the biquad helpers do not expose `clamp`. The device
chain draws the cutoff from a weighted discrete list, since real devices use a
few design cutoffs (for example 100, 200, 300 Hz), and `Q ~ N(0.707, 0.1²)`
clipped to [0.3, 1.3]: close to Butterworth, never exactly.

## 4. Resampling

### 4.1 What it models

"The signal passed through a lower sample rate somewhere in the chain": a
codec's internal rate, Bluetooth, a driver. The stage is a round trip,
`fs → src_sr → fs`. The anti-aliasing filter removes content above `src_sr/2`,
which the upsampling cannot restore, and leaves the filter's own signature:
transition-band shape, passband ripple, residual images.

### 4.2 Two backends

`dsp.wav_resampling(wav, origin_sr, target_sr, backend, torch_backend_params=None)`:

| `backend` | Filter | Returns |
|---|---|---|
| `"sox"` | fixed, near-transparent windowed sinc: `torchaudio.functional.resample` with `sinc_interp_kaiser`, `lowpass_filter_width=64`, `rolloff≈0.9476`, `beta≈14.77` (the "kaiser best" setting). Deterministic. The name refers to the sox `rate` effect it stands in for; sox itself is not called. | `(wav, target_sr)` |
| `"torchaudio"` | randomised windowed sinc: `lowpass_filter_width ∈ {6, 16, 32, 64, 128}`, `rolloff ~ U(0.8, 0.99)`, window ∈ {`sinc_interp_hann`, `sinc_interp_kaiser`}, unless `torch_backend_params` supplies `lp_width`, `rolloff`, `window` | `(wav, target_sr, params)` |

`origin_sr == target_sr` returns the input unchanged in the same tuple shape.

- `lowpass_filter_width` is the sinc truncation length in zero crossings. Short
  kernels mean a wide transition band, weak stopband and more image leakage; the
  range spans cheap to high-quality implementations.
- `rolloff` is the anti-aliasing cutoff as a fraction of Nyquist: 0.8 cuts early
  (little aliasing, more bandwidth lost), 0.99 keeps nearly the full band.
- The window sets the sidelobe/main-lobe trade-off of the truncation.

The fixed backend is the neutral rate converter: corpus loading
(`AudioIO.open(resample_to=...)`), noise and RIR resampling all use it. The
randomised backend is an augmentation. The device chain picks between the two
50/50, because fixing one filter would bake one implementation's signature into
the training data.

### 4.3 The target gets the same filter

The randomised backend returns the three parameters it drew, and the device
chain passes them back for the up leg and for both legs on the target. The pair
must see the *same filter*, not merely the same rates; otherwise mixture and
target carry different band limits and images, and their difference is no
longer what the model should remove.

## 5. Speed and pitch

The `sox_*` methods on `AudioEffectAugmentor` are named after the sox effect
each one reproduces; none of them calls sox.

- **`sox_speed_perturbed(wav, speed, sr)`** reinterprets the clip at
  `sr · speed` and resamples to `sr` (`torchaudio.functional.resample`, default
  filter). Duration scales by `1/speed` and pitch by `speed`, together:
  `speed = 1.1` is 10 % faster and `12·log2(1.1) ≈ 1.65` semitones higher. This is
  the real covariation of speaking rate and pitch, not time stretching.
- **`sox_pitch_perturbed(wav, shift_ratio, sr)`** shifts pitch by `shift_ratio`
  cents at constant duration (`torchaudio.functional.pitch_shift` with 1200 bins
  per octave, a phase vocoder). No task pipeline calls it.
- **`sox_volume_perturbed(wav, vol_ratio, sr)`** is a plain multiply, so it
  cannot clip.

Noise suppression and voice isolation (`ContinuousSpeedAugmentation`) draw the
speed from the grid `arange(lo, hi + 0.025, 0.05)` over `speed_range`; the half
step keeps `hi` on the grid, which a bare `arange(lo, hi, 0.05)` would drop. The
same factor is applied to mixture and target, after mixing and before the
whole-mix RIR, and the factor is stored in the row plan (`speed_factor`) so
per-frame labels can be mapped onto the new time grid. The frozen
target-speaker-extraction task uses the bare `arange`. Speaker embedding uses
`DiscreteSpeedAugmentation`, a list `speed_change` plus `treat_as_new_speaker`.

## 6. Media coloring

`AudioEffectAugmentor.apply_media_coloring(wav, sr, hp_cutoff, lp_cutoff,
compress_power)` models speech played through a television or loudspeaker in the
room. It is applied to interferers flagged as media, before their RIR, because
the device plays the sound and the room then carries it
([Scene construction](scene_construction.md)).

```
1. y = LP(HP(x))                          HP and LP biquads, Q = 0.707, via apply_linear
2. y = sign(y) · (|y| / P)^p · P,  P = max|y|      optional, p ∈ (0, 1)
3. y ← y · rms(x) / rms(y)
```

- **Band limiting.** Small loudspeakers lose bass to driver size and enclosure
  volume and treble to diaphragm mass and crossover design. `hp_cutoff_range`
  and `lp_cutoff_range` cover television and laptop speakers.
- **Power-law waveshaping.** On a peak-normalised signal, `p < 1` lifts small
  values (`0.1^0.7 ≈ 0.2`) and keeps the peak at 1, which compresses the dynamic
  range the way broadcast content is compressed. A static curve is enough because
  it shapes an interfering source, not anything that is scored.
- **RMS restoration.** Without it the coloring would also be a level change,
  which the SIR scaling downstream would absorb, entangling "colored" with "what
  SIR". Restoring RMS confines the effect to spectrum and dynamics.
- **Why `apply_linear` around the filters.** Unwrapped, the biquads would clip at
  full scale and the RMS restoration would scale the damage back to the input
  level, hiding it.

The waveshaper has no per-source decomposition: `(near + far)^p ≠ near^p + far^p`.
It can only act on one source before mixing. "The whole recording was
compressed" needs a time-varying gain instead, the compressor in
[Level and dynamics](level_dynamics.md), because a gain distributes over a sum.

## 7. `apply_linear`

### 7.1 The problem

Every linear stage relies on one property: applied to mixture and target with the
same parameters, it leaves the mixture equal to the sum of its sources at the
drawn SIR. Backend defaults break it. `torchaudio.functional.lfilter` clamps its
output to [−1, 1] unless called with `clamp=False`, and every torchaudio
`*_biquad` helper is built on `lfilter` without exposing the flag. On a hot
signal a stage modelling a linear device then saturates. Because the mixture is
the louder signal of the pair, it is the one squashed while the target passes
untouched, and the pair no longer has the level relationship the recipe asked
for.

### 7.2 The method

Linearity itself: for a linear `H` and any `a > 0`, `H(x) = H(a·x) / a`.

```
peak  = max|x|
scale = headroom / peak                  headroom = 0.5
out   = fn(x · scale) / scale
```

`dsp.apply_linear(fn, wav, *, headroom=0.5, max_escalations=8)` returns the
operator's true output at the input's level.

### 7.3 Headroom and escalation

`headroom = 0.5` leaves `20·log10(2) ≈ 6 dB` for the operator's own gain, which
covers most filters in this package. If the output still reaches full scale
(`|out| ≥ 1 − 1e-6`; a saturated backend pins samples at exactly ±1, real audio
reaches that only by coincidence), the scale is divided by 8 (18 dB) and the call
repeated, up to `max_escalations` times; after that a `RuntimeError` is raised
rather than returning a quietly wrong result. Digital silence or a non-finite
peak is passed to `fn` unscaled.

### 7.4 Constraints

- **`fn` must not consume randomness.** An escalation calls it again, which would
  shift the RNG stream and break seeded reproduction
  ([Engineering contract](engineering_contract.md)). Draw random parameters
  outside and close over them.
- **Prefer the backend's switch when it has one.** `lfilter(clamp=False)` is exact
  and costs nothing; the random IIR and the compressor's envelope detector use
  it. `apply_linear` is for backends without a switch: `apply_hpf` and both
  filters of `apply_media_coloring`.

## Configuration

| Knob | Schema | Where it applies |
|---|---|---|
| `augmentation_src` | `SourceRateAugmentation` (`src_range` and `prob_each` of equal length) | `DeviceChain._sample_rate_conversion` |
| `augmentation_ir_response` | `SimpleProbAugmentation` | `DeviceChain._second_order_iir` |
| `augmentation_hpf` | `HighPassAugmentation` (`cutoff` and `prob_each` of equal length) | `DeviceChain._high_pass` |
| `augmentation_speed` | `ContinuousSpeedAugmentation` (`speed_range`) for NS, VI, TSE; `DiscreteSpeedAugmentation` (`speed_change`) for speaker embedding | `ns.py`, on the pair after mixing |
| `augmentation_speech.media_voice` | `MediaVoiceConfig` (`prob`, `hp_cutoff_range`, `lp_cutoff_range`, `compress_power_range`) | each synthetic interferer, before its RIR |

```yaml
augmentation_speech:
  media_voice:
    used: True
    prob: 0.30                    # per interferer
    hp_cutoff_range: [200, 400]   # Hz
    lp_cutoff_range: [3500, 7000] # Hz
    compress_power_range: [0.6, 0.9]
```

## Notes

- A torchaudio-backend call without `torch_backend_params` draws its own filter.
  Any caller that needs a second signal to follow must pass the returned
  parameters back, as the device chain does.
- The random IIR records only `iir_applied`, not its coefficients, so evaluation
  cannot bucket by response shape without extending the provenance scalars.
