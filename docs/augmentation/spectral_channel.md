# Spectral and channel filtering

繁體中文版本：[`spectral_channel.zh-TW.md`](spectral_channel.zh-TW.md)

Microphones have a frequency response, devices have rumble filters, and
transmission chains pass through resamplers. This chapter covers the design
equations and derivations behind every linear filtering technique in the
pipeline, plus speed perturbation, one deliberate nonlinearity (media
coloring), and the `apply_linear` mechanism that keeps all the linear
techniques genuinely linear.

## Algorithm

### 1. Biquads: second-order IIR filters

#### 1.1 Transfer function and difference equation

The biquad (bi-quadratic, second-order IIR) is the basic unit of all
equalisation and filtering:

```
H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (a0 + a1·z⁻¹ + a2·z⁻²)
```

Normalising `a0 = 1` gives the difference equation:

```
y[n] = b0·x[n] + b1·x[n−1] + b2·x[n−2] − a1·y[n−1] − a2·y[n−2]
```

Second order is the minimum that can provide "one resonance or one transition
band": two poles can form a complex-conjugate pair setting the resonant
frequency and quality factor, and two zeros determine stopband behaviour.
Higher-order responses (such as `ParametricEQ`) are built by cascading
biquads.

#### 1.2 The RBJ design equations

`get_biquad_params` uses the parameterisation from the RBJ Audio EQ Cookbook,
converting "the parameters an engineer wants" (gain, cutoff, Q) into
coefficients. Three intermediate variables:

```
A  = 10^(gain_dB / 40)        # square root of the amplitude gain
w0 = 2π · fc / fs             # normalised angular frequency (rad/sample)
α  = sin(w0) / (2·Q)          # bandwidth parameter
```

**Why `A` divides by 40 rather than 20.** In the shelving and peaking
coefficient formulas `A` appears multiplicatively in both numerator and
denominator (for instance `1 + α·A` against `1 + α/A`), so the resulting
response gain is `A²`. To achieve `gain_dB` of response, `A` must be the
square root of the amplitude ratio: `A = 10^(gain_dB/20 / 2) = 10^(gain_dB/40)`.

**What `w0` means.** Digital frequency is expressed in radians per sample;
`fc / fs` is the normalised frequency (Nyquist being 0.5), and multiplying by
`2π` gives the angular frequency.

**How `α` relates to Q.** The quality factor is defined as the resonant
frequency divided by the −3 dB bandwidth: `Q = f0 / Δf`. Larger Q means
narrower bandwidth and a sharper resonance. `α = sin(w0)/(2Q)` is the digital
equivalent of Q after the bilinear transform; the `sin(w0)` factor comes from
the frequency warping between the s and z planes.

Common values: `Q = 0.707 = 1/√2` gives a Butterworth response (maximally flat
passband, no overshoot) and is the default for HPF/LPF. `Q = 0.5` is critically
damped (fastest response without ringing); `Q > 1` produces a peak near the
cutoff frequency.

#### 1.3 Coefficients for the seven filter types

The implementation supports `high_shelf`, `low_shelf`, `peaking`, `lpf`,
`hpf`, `bpf`, and `notch`. Three representative sets:

```
LPF:       b = [(1 − cos w0)/2,  1 − cos w0,  (1 − cos w0)/2]
           a = [1 + α,  −2·cos w0,  1 − α]

HPF:       b = [(1 + cos w0)/2, −(1 + cos w0), (1 + cos w0)/2]
           a = [1 + α,  −2·cos w0,  1 − α]

peaking:   b = [1 + α·A,  −2·cos w0,  1 − α·A]
           a = [1 + α/A,  −2·cos w0,  1 − α/A]
```

Note the structural relationship between LPF and HPF: their denominators are
identical (pole positions depend only on `fc` and `Q`) and they differ only in
where the numerator places its zeros — LPF at Nyquist (`z = −1`), HPF at DC
(`z = 1`).

The symmetry in the peaking filter is also worth observing: the numerator's
`α·A` against the denominator's `α/A` are reciprocals, so `gain_dB → −gain_dB`
exactly swaps numerator and denominator and the response inverts precisely.

All coefficients are finally divided by `a0` to normalise.

#### 1.4 Assumptions and failure conditions

The RBJ equations are derived from analogue prototypes via the bilinear
transform, which warps frequency: the analogue frequency `Ω` relates to the
digital `ω` as `Ω = 2·tan(ω/2)`. The equations pre-warp at `w0` (placing the
cutoff frequency exactly where specified), but **bandwidth and gain still
deviate from the design values near Nyquist**.

Practical impact: at 16 kHz sampling, a shelving filter with `fc` above 6 kHz
shows a visible deviation from its theoretical curve. Within the speech band
(80 Hz – 4 kHz) this is negligible.

#### 1.5 ParametricEQ: the cascade structure

`ParametricEQ` cascades a low shelf, N peaking filters, and a high shelf:

```
H(z) = Π_i H_i(z)
```

Cascading means dB responses add
(`20·log10|Π H_i| = Σ 20·log10|H_i|`), which is exactly the intuitive behaviour
of an equaliser: each band is adjusted independently and the effects sum. The
implementation calls `scipy.signal.lfilter` (Direct Form I) stage by stage.

Cascading rather than summing in parallel: a parallel arrangement lets the
phase responses of the bands interfere, so two simultaneously boosted bands can
produce a dip at their crossover. In a cascade the phases merely accumulate and
the magnitude response stays predictable.

### 2. Random second-order IIR: transducer response

#### 2.1 Algorithm

Models "every microphone has a different frequency response" (source: A Hybrid
DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement, the
augmentation approach from the RNNoise line of work):

```
r0..r3 ~ U(−3/8, 3/8)
a = [1, r0, r1]        # denominator (poles)
b = [1, r2, r3]        # numerator (zeros)
```

Both numerator and denominator are random second-order polynomials, giving a
random, mild frequency response curve.

#### 2.2 Why ±3/8 guarantees stability

An IIR filter is stable when all poles lie inside the unit circle. For the
second-order denominator `1 + a1·z⁻¹ + a2·z⁻²`, the Schur–Cohn stability
criteria are:

```
|a2| < 1
|a1| < 1 + a2
```

Substituting `a1, a2 ∈ [−3/8, 3/8]`:

* First: `|a2| ≤ 3/8 < 1` always holds.
* Second: `|a1| ≤ 3/8`, while `1 + a2 ≥ 1 − 3/8 = 5/8 > 3/8` always holds.

Both criteria hold across the entire sampling range, so **no rejection
sampling or stability check is needed** — which is why 3/8 was chosen as the
bound.

#### 2.3 Gain range and its relationship with apply_linear

Poles near the unit circle produce peaks. Measured over these parameters, the
response peaks at a median of about +3.5 dB and up to +16 dB in the worst case.
This long tail is the direct reason `apply_linear` (§7) needs an escalation
loop: with only 6 dB of headroom, a +16 dB response still hits full scale.

#### 2.4 Why clamp=False is required

The implementation applies the filter with `lfilter(..., clamp=False)`. A
frequency response is a linear operator, but
`torchaudio.functional.lfilter` clamps its output to [−1, 1] by default. On a
hot signal that clamp is a waveshaper the recipe never asked for, and one that
reaches only the louder mixture and not the quieter target it is scored
against — so the SIR relationship is broken. Where the backend offers a switch,
turn it off directly; this is exact and free, unlike the scaling wrapper of §7.

### 3. High-pass (rumble filter)

Models a device's low-frequency roll-off (mains hum, wind noise, handling
noise). Implemented with `torchaudio.functional.highpass_biquad` (RBJ HPF):

* The cutoff is drawn from a weighted discrete list. Real devices choose their
  HPF cutoff by design (80/120/150 Hz are common), so a discrete list matches
  reality better than a continuous distribution.
* `Q ~ N(0.707, 0.1²)` clamped to [0.3, 1.3]: centred on Butterworth with small
  deviations allowed, since a real device's filter is never exactly
  Butterworth.

This backend offers no clamp switch, so it runs inside `apply_linear` (§7).

### 4. Resampling

#### 4.1 Purpose and principle

Models "the signal passed through a different sample rate somewhere in the
chain" (internal codec resampling, Bluetooth, driver layers). The method is
down-then-up:

```
fs → src_sr → fs
```

The net effect is **band limiting**. On downsampling the anti-aliasing filter
removes content above `src_sr/2`, and upsampling cannot recover it. Beyond the
bandwidth loss, the characteristics of that resampler's anti-aliasing filter
remain — transition band shape, passband ripple, residual images.

#### 4.2 Two backends

**sox `rate`**: a fixed, high-quality polyphase filter implementation.

**torchaudio windowed-sinc**: three controllable parameters, deliberately
randomised:

```
lowpass_filter_width ∈ {6, 16, 32, 64, 128}
rolloff ∈ U(0.8, 0.99)
window ∈ {sinc_interp_hann, sinc_interp_kaiser}
```

The physical meaning of each:

* `lowpass_filter_width` is the truncation length of the sinc kernel (counted
  in zero crossings). Ideal resampling needs an infinite sinc; the shorter the
  truncation, the wider the transition band, the worse the stopband
  attenuation, and the more image leakage. The range 6 to 128 spans "cheap
  implementation" through "high-quality implementation".
* `rolloff` is the anti-aliasing cutoff as a fraction of Nyquist. 0.8 cuts
  early (conservative: little aliasing but more bandwidth lost); 0.99 sits
  right at Nyquist (full bandwidth retained but possible residual aliasing).
* The window determines how the sinc is truncated. Hann has faster sidelobe
  decay but a wider main lobe; Kaiser lets that trade-off be tuned.

**Why randomise rather than fix.** Real-world resampler implementations differ.
Fixing one parameter set bakes a single vendor's filter signature into the
training distribution, and the model may learn to compensate for that specific
signature. Randomising widens the artefact distribution and forces the model to
be insensitive to resampling traces.

The device chain picks between the two backends 50/50 for the same reason.

#### 4.3 Parameter reuse for the target

The torchaudio path returns the three parameters it actually drew, and the
device chain uses them to resample the target identically. This is stricter
than "the same sample rates" — it must be the **same filter**, otherwise
mixture and target each carry different bandwidth losses and image artefacts,
and their difference is no longer "what the model should remove".

### 5. Speed and pitch

Three sox effects, all executed through `apply_linear` (sox round-trips through
a fixed-point format internally and saturates at full scale):

* **`speed`**: changes the playback rate and resamples back to the original
  `fs`. Time axis and pitch change **together** (this is not time stretching) —
  `speed = 1.1` makes speech 10% faster and raises the pitch by about
  1.6 semitones. This models the real covariation of rate and pitch, and is the
  cheapest speech augmentation available.
* **`pitch`**: changes pitch only (in cents), preserving duration. Used for
  speaker-variability augmentation in speaker-related tasks (TSE/SV).
* **`vol`**: pure gain.

The NS pipeline draws speed from a discrete grid:
`arange(lo, hi + 0.025, 0.05)`. The half-step added to the upper bound is
deliberate — a bare `arange(lo, hi, step)` excludes the endpoint, which would
silently make `speed_range: [0.9, 1.1]` sample only 0.9–1.05 and remove the
speed-up half. The same speed value is applied to mixture and target as a pair.

### 6. Media coloring: spectral and dynamic model of loudspeaker playback

Models "sound played back through a television or loudspeaker" in the scene,
applied to interferers flagged as media
([ch7](scene_construction.md)). Three steps:

```
1. HP biquad (Q=0.707) → LP biquad (Q=0.707)          # band limiting
2. y = sign(x) · (|x| / P)^p · P,   P = max|x|         # power-law waveshaping
3. y ← y · rms_in / rms_out                            # RMS restoration
```

#### 6.1 The physical basis for band limiting

Small loudspeakers have a well-defined passband: the low end is limited by
diaphragm size and enclosure volume (insufficient bass extension), the high end
by diaphragm mass and crossover design. The config's `hp_cutoff_range` and
`lp_cutoff_range` cover the typical television and laptop speaker range. Band
limiting is a linear operation, so it runs inside `apply_linear`.

#### 6.2 What the power-law waveshaping does

`|x|^p` with `p ∈ (0, 1]` is a **memoryless** static nonlinearity. On a
peak-normalised signal (`|x/P| ≤ 1`), `p < 1` lifts small values (for instance
`0.1^0.7 ≈ 0.2`) while leaving the maximum at 1, compressing the dynamic range.

This is a zeroth-order approximation of broadcast-chain compression: broadcast
and streaming content is heavily compressed, so what a device plays back has a
much smaller dynamic range than the original recording. A static nonlinearity
rather than a real compressor is sufficient here because timing accuracy is not
needed — this is a characteristic of an interfering source, not of anything
being scored.

#### 6.3 Why RMS restoration is necessary

Both band limiting and waveshaping change the signal's RMS. Without
restoration, coloring itself becomes a level change, and the downstream SIR
scaling would absorb it — entangling "was it colored" with "what SIR". Restoring
RMS confines coloring to spectrum and dynamics, leaving level alone.

#### 6.4 Why band limiting must be wrapped in apply_linear

Without the wrapper the biquads clip at full scale, and the RMS restoration of
§6.3 then scales the resulting damage back up to the input level — a waveshaper
wearing a filter's name. The RMS restoration is precisely what makes the error
hard to notice (levels look correct), which is why the wrapping must be
explicit.

#### 6.5 Division of labour with the compressor

Power-law waveshaping is a per-sample static nonlinearity with **no per-source
decomposition**: `(near + far)^p ≠ near^p + far^p`. It can therefore only be
used before mixing, acting on a single source ("that device in the room"), and
cannot model "the whole recording was compressed". The latter requires a
time-varying gain ([ch4](level_dynamics.md) §6), because only multiplication
distributes over addition.

### 7. apply_linear: keeping linear backends linear

#### 7.1 The problem

Every linear technique in this chapter rests on one premise: a linear operator
applied to mixture and target leaves the mixture still equal to the sum of its
sources at the specified SIR. That premise is broken by backend implementation
details:

* `torchaudio.functional.lfilter` clamps its output to [−1, 1] by default.
* Every `biquad` function is built on `lfilter` and does not forward the
  `clamp` argument.
* Every sox effect round-trips through a fixed-point sample format and
  saturates at full scale.

So whenever the input signal is hot, a stage modelling a **linear** phenomenon
(microphone response, rumble filter, preamp gain, resampler) silently stops
being linear.

**Why this matters more than the distortion itself.** These stages are applied
to mixture and target with identical parameters precisely to preserve their
level relationship. A ceiling at a fixed absolute level breaks that symmetry:
the mixture is the louder of the two and gets squashed, while the target passes
through untouched. The mixture then stops being the sum of its parts at the SIR
the recipe asked for.

This is not a theoretical risk; it fires frequently on shipped recipes, with
hit-rate statistics recorded in the `apply_linear` docstring.

#### 7.2 The fix

Use the definition of linearity itself. For a linear operator `H` and any
scalar `a > 0`:

```
H(x) = H(a·x) / a
```

So scale into the backend's legal range, apply, and scale back:

```
peak  = max|x|
scale = headroom / peak          # headroom = 0.5
out   = fn(x · scale) / scale
```

The result is the operator's true output and the saturation never fires.

#### 7.3 Headroom choice and the escalation loop

`headroom = 0.5` leaves 6 dB of room for the operator's own gain
(`20·log10(1/0.5) = 6.02 dB`), which covers most filters in this package — the
random second-order response peaks at a median of about +3.5 dB (§2.3).

But that response reaches +16 dB in the worst case, exceeding 6 dB of room. The
implementation therefore checks whether the output still touches full scale
(within a `1e-6` tolerance) and, if so, divides the scale by another factor of
8 (a further 18 dB) and retries, up to 8 times. Exhausting all attempts raises
an exception — better to fail than to return a number that is quietly wrong.

The test uses `>= 1 − 1e-6` rather than exact equality because a saturated
backend leaves samples pinned exactly at ±1, whereas real audio lands there
only by coincidence; a false positive costs one unnecessary retry and nothing
else.

#### 7.4 Constraints on fn

**`fn` must not consume randomness.** An escalation calls `fn` again, and if it
sampled it would shift the global RNG stream and break the bit-exact
seeded-item contract ([ch8](engineering_contract.md)). Every current call site
is a pure filter or gain.

Where random parameters are needed (such as the resampler parameters in §4.2),
they must be drawn **outside** `apply_linear` and passed in via a closure.

#### 7.5 When not to use it

When the backend has a clamp switch, use it instead of this wrapper:
`lfilter(clamp=False)` is exact, whereas the scaling wrapper costs at least one
extra peak scan and two multiplications.

## Engineering

### Config mapping

| Knob | Schema | Where it lands |
|---|---|---|
| `augmentation_src` | `SourceRateAugmentation` (`src_range` and `prob_each` equal length) | `DeviceChain._sample_rate_conversion` |
| `augmentation_ir_response` | `SimpleProbAugmentation` | `DeviceChain._second_order_iir` |
| `augmentation_hpf` | `HighPassAugmentation` (`cutoff` and `prob_each` equal length) | `DeviceChain._high_pass` |
| `augmentation_speed` | `ContinuousSpeedAugmentation` (enhancement tasks) / `DiscreteSpeedAugmentation` (speaker tasks) | `ns.py`, applied as a pair after mixing |
| `augmentation_speech.media_voice` | `MediaVoiceConfig` | Applied when sampling interferers, before their RIR |

### Ordering and mutual exclusion

* Inside the device chain the order is fixed: SRC → IIR → HPF
  ([ch6](device_chain.md)).
* Speed perturbation runs after mixing and before the whole-mix RIR (`ns.py`).
* Media coloring runs **before** that interferer's RIR: the device plays the
  sound first, then it propagates through the room — the physical order.
* The random second-order IIR coefficients are drawn on the mixture's call and
  the target reuses the same pair; coefficient sharing is that stage's
  contract.

### Pitfalls

* If the torchaudio path of `wav_resampling` is called without
  `torch_backend_params`, it draws its own three parameters. Any path needing
  determinism or target-following must forward the returned parameters (the
  device chain does this correctly; new call sites need to remember).
* `ParametricEQ.plot_eq` hard-codes nfft for 16 k and 32 k only. It is a
  debugging tool, not part of the pipeline.
* The second-order IIR is one of the few device chain stages that records no
  parameters (only `iir_applied`), so eval cannot bucket by response shape. If
  that analysis axis is needed, the provenance scalars must be extended first.
