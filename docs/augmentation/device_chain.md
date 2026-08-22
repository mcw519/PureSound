# Device chain

繁體中文版本：[`device_chain.zh-TW.md`](device_chain.zh-TW.md)

Once mixing is finished and before the signal reaches the model, everything
that remains belongs to **capture and transmission**: resampling, microphone
response, rumble filter, preamp gain, compression, A/D conversion, VoIP codecs,
packet loss. None of it knows anything about talkers or rooms, which is why it
lives in `puresound/task/device_chain.py::DeviceChain` rather than in the
middle of `__getitem__`.

## Algorithm

### 1. Structure of the chain

```
SRC → 2nd-order IIR → HPF → volume/clipping → compressor    # analogue path (linear group)
  ────────────────── _analogue_to_digital ──────────────────  # the A/D boundary
  → codec → packet loss                                      # digital transmission damage
```

The grouping is determined by **whether the target follows**, which in turn
follows from the physical phenomenon each stage models.

#### 1.1 The analogue group: the target follows

The analogue group models transducer frequency response, filtering, preamp
gain, and compression — all (time-varying) linear operators. The training pair
is defined such that **the target is what the model should recover *through*
this channel**. Whatever the channel does to the mixture must therefore also be
done to the target.

Linearity is what makes this work mathematically. For a linear operator `H`:

```
H(near + far + noise) = H(near) + H(far) + H(noise)
```

After `H` the mixture still equals the sum of its components with unchanged
proportions — the SIR the recipe asked for still holds downstream of the
channel. This is why every analogue stage is applied to mixture and target with
**identical parameters**.

#### 1.2 The transmission group: the target does not follow

Codec distortion and packet loss are **damage**, not channel characteristics.
The model should learn to repair them, not to reproduce them. Both stages
therefore act on the mixture only, leaving the target as the undamaged
reference.

This distinction is not a matter of convention but of task definition: if the
target also carried codec artefacts, the optimal learned behaviour would be to
preserve them; with a clean target, the model is asked to remove them.

### 2. What each stage does, and how the target follows

| Stage | Models | Mathematics | How the target follows |
|---|---|---|---|
| `_sample_rate_conversion` | Passing through a different sample rate | `fs → src_sr → fs` band limiting and filter traces ([ch5](spectral_channel.md) §4) | Same backend; the torchaudio path reuses the three filter parameters drawn for the mixture |
| `_second_order_iir` | Transducer frequency response | Random second-order IIR ([ch5](spectral_channel.md) §2) | Same `(a, b)` coefficients |
| `_high_pass` | Rumble filter | RBJ HPF, weighted cutoff draw, `Q ~ N(0.707, 0.1²)` clamped to [0.3, 1.3] ([ch5](spectral_channel.md) §3) | Same cutoff, same Q |
| `_volume` | Preamp gain or analogue overload | One draw decides: gain `U(perturbed_range)`, or quantile clipping ([ch4](level_dynamics.md) §3–4) | Gain uses the same ratio; clipping uses the mixture's **realised** quantile values |
| `_compressor` | Broadcast-style compression | Time-varying gain curve ([ch4](level_dynamics.md) §6) | Curve derived from the mixture, same curve multiplied into both |
| `_analogue_to_digital` | Gain staging | See §3 | Both divided by the same peak |
| `_codec` | VoIP / telephony coding | encode → decode round trip | Does not follow (mixture only) |
| `_packet_loss` | Packet loss | Per-packet Bernoulli drop, zeroed | Does not follow (mixture only) |

#### 2.1 The target's clipping deserves separate explanation

The clipping branch of `_volume` draws quantiles `(min_q, max_q)` for the
mixture and computes the actual thresholds `(lo, hi)`. The target is then
clipped with **those two absolute values**, not with its own quantiles.

The reason is physical: analogue overload happens at the microphone output and
its threshold is an absolute voltage set by the circuit. Mixture and target
went through the same overload, so the threshold is the same. Using each
signal's own quantiles would clip the quieter target at a lower absolute level,
which amounts to it having experienced a more severe overload.

### 3. The A/D boundary: the only place in the pipeline with a full scale

The semantics of `_analogue_to_digital` define the level model for every stage
upstream, which is why it gets its own section.

#### 3.1 What each side means

**Upstream is the acoustic and analogue path.** Level there is sound pressure
through a transducer and a preamp. Pressure has no full scale — a peak above
1.0 is not an error, it is a loud room. No upstream stage may treat 1.0 as a
ceiling, which is exactly why `apply_linear`
([ch5](spectral_channel.md) §7) exists.

**Downstream is digital.** A codec cannot encode samples beyond full scale, and
the model's own output is clamped to [−1, 1], so a target above it is one the
model could never reach however well it separates.

#### 3.2 Crossing the boundary is gain staging, not clipping

```
peak = max( max|noisy|, max|target| )
if peak > 1.0:
    noisy  ← noisy  / peak
    target ← target / peak
```

The real-world action this corresponds to is **an engineer setting the preamp
so the converter is not driven into its rails**. It is a single linear rescale,
so superposition survives and the mixture still equals its sources at the
specified SIR.

Using the two signals' **shared** peak rather than each signal's own is
necessary to preserve the level relationship: normalising them separately would
change their relative loudness.

#### 3.3 Why deliberate overload does not live here

Deliberate overload distortion is a different thing and lives in the clipping
branch of `_volume`: it is probability-controlled, recorded in provenance
(`volume_clipped`), and applied to both signals at the same absolute
thresholds.

If the A/D boundary also clipped, rows already clipped by `_volume` would be
clipped a second time, and that second time would carry no record — eval could
not distinguish "this row was clipped once" from "twice".

#### 3.4 Other properties

* This step **consumes no randomness**.
* It can be disabled with `overload_guard=False`. The TSE task historically has
  no such stage; making it a knob rather than behaviour a shared component
  quietly imposes keeps that task's training distribution unchanged.
* `overload_rescaled` records whether the row triggered it, letting eval
  identify rows whose level was pulled down as a whole.

### 4. Codec and packet loss

#### 4.1 Codec

The implementation performs a genuine encode → decode round trip via
`torchaudio.io.AudioEffector` rather than approximating with filters. The
resulting distortion therefore includes the codec's actual behaviour:
quantisation noise, bandwidth limiting, and spectral modification by its
psychoacoustic model.

Each codec is bound to a container format (`libopus` → ogg, `g722` →
matroska), chosen for round-trip reliability in voice-bandwidth use (avoiding
compatibility quirks such as Opus inside MKV).

Characteristics of the two codecs:

* **libopus**: the modern VoIP mainstream, with an adjustable bitrate (the
  per-codec `bitrate_range` in config). At low bitrates its behaviour is
  dominated by bandwidth limiting and parametric coding.
* **g722**: the ITU-T sub-band ADPCM wideband telephony standard. It forces
  16 kHz internally and has no bitrate knob (fixed 64 kbps), so a
  `bitrate_range` entry for it is ignored.

Length handling: a codec round trip can change the sample count (encoder delay,
frame alignment), so the implementation truncates or zero-pads back to the
original length, keeping downstream length assumptions intact.

#### 4.2 Packet loss

```
packet_samples = round(fs · packet_ms / 1000)
each packet is dropped with probability loss_rate (Bernoulli)
dropped packet intervals are zeroed
```

`packet_ms` defaults to 20 ms, the WebRTC default packet length; 60 ms is
common for low-bandwidth Opus.

The implementation zeroes the gap and performs no packet loss concealment
(PLC). This is a deliberate worst-case model: real VoIP endpoints extrapolate
or repeat waveform to fill gaps, but that is endpoint behaviour and differs
between implementations. The training goal is robustness to a signal genuinely
missing a section, not simulation of one particular endpoint's PLC algorithm.

**Limits of the independence assumption**: the implementation draws
independently per packet. Real network loss is bursty (several consecutive
packets lost together), so the independent model underestimates the probability
of long gaps. There is currently no burst model; if one is needed, a
Gilbert–Elliott two-state Markov chain would be the natural extension.

## Engineering

### Two contracts

These are stated explicitly in the module docstring and must be preserved by
any modification.

#### Contract one: stage order is load-bearing

Every stage draws from the shared RNG stream. Swapping two stages changes the
random numbers every subsequent stage receives, so the same seed produces
different data — even though no stage's own implementation changed.

The current order is the order the released checkpoints were trained with. The
order also has physical justification (transducer response before the preamp,
compression before the converter), but even two stages that could physically be
swapped should not be, merely for tidiness.

#### Contract two: a disabled stage never touches the RNG stream

Every guard has the same shape:

```python
block is not None and block.used and torch.rand(1) < block.prob
```

The probability draw sits **inside** the short circuit. Consequently, disabling
one stage (or a dataset simply not having that block) leaves every later stage
drawing exactly what it drew before.

This is the mechanism behind "adding a knob keeps old recipes bit-identical".
Full discussion in [ch8](engineering_contract.md).

### Provenance scalars

Every row records what the chain actually did (`DEVICE_CHAIN_SCALARS`), so eval
can bucket results by the channel a row went through — the same thing
`eval_indomain.py --by-bucket` already does with SIR, overlap, and DRR.

Conventions:

* `*_applied`, `*_clipped`, `*_rescaled` are 0.0 / 1.0.
* A parameter field is `NaN` on rows where its stage did not fire.
* **Every row carries every key** (the value may be NaN). This is what lets
  them ride the existing scalar collate (one `torch.cat` of 0-dim tensors per
  key) for free; a row missing a key would make the batch impossible to collate
  into a single tensor.
* The codec kind is emitted as a float code (`CODEC_CODES`), mirroring
  `MIX_MODE_CODES` — strings cannot enter the scalar collate.

The string-valued `RIR_PROVENANCE_KEYS` take a different collate path. The
module comments flag them as the cautionary example: added for traceability,
never read by any analysis. Before adding a new record field, establish who
will read it.

### Config mapping

| Stage | Config block | Schema |
|---|---|---|
| SRC | `augmentation_src` | `SourceRateAugmentation` |
| 2nd IIR | `augmentation_ir_response` | `SimpleProbAugmentation` |
| HPF | `augmentation_hpf` | `HighPassAugmentation` |
| volume | `augmentation_volume` | `VolumeAugmentation` |
| compressor | `augmentation_compressor` | `CompressorAugmentation` |
| codec | `augmentation_codec` | `CodecAugmentation` (NS / voice isolation only) |
| packet loss | `augmentation_packet_loss` | `PacketLossAugmentation` (same) |

`device_chain_from_blocks` builds the chain from the dataset's already-validated
blocks. A stage whose block the task does not have is simply absent at no cost
— this is how NS and TSE share one chain (a block that `getattr` cannot find is
passed as `None`, and the first condition of `_fires` short-circuits).

### Pitfalls

* **Any behavioural modification to this chain changes the synthesis
  distribution**, including obviously beneficial ones such as "make an
  imprecise approximation exact". Checkpoints trained either side of such a
  change cannot be compared directly on synthetic-domain metrics. See the
  distribution versioning section of [ch8](engineering_contract.md).
* The VAD reference is snapshotted **before** the chain (`ns.py`). VAD labels
  are computed on the clean (early-reverb) signal, because VADs such as Silero
  become unreliable on heavily distorted speech. Chain damage therefore never
  changes activity labels.
* The compressor curve and the signals are aligned by `min()`. The SRC stage
  can introduce a 1–2 sample length difference; uncovered samples at the end
  keep their original values.
* `_fires` has a `probability_draw=False` parameter with no current caller; it
  lets a stage honour `used` without drawing a probability. If it is ever used,
  note that it changes that row's RNG consumption.
