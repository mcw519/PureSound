# Scene construction

繁體中文版本：[`scene_construction.zh-TW.md`](scene_construction.zh-TW.md)

A training row describes a scene: who is talking, where they are, when they
speak, how loud they are, and what else the environment contains. This chapter
covers the scene-level techniques — row types, interferer sourcing, overlap
gating, mixing modes, residual echo, and the three noise sources.

The implementation spans `puresound/task/ns.py` (the synthesis skeleton),
`puresound/task/voice_isolation.py` (near/far row types),
`puresound/task/overlap_gating.py`, and `puresound/task/noise_stage.py`.

## Algorithm

### 1. Row types: the script for one row

`_plan_row` draws once at the start of each row to decide its type (`RowPlan` /
`VoiceIsolationRowPlan`). The synthesis skeleton then branches on the plan.

#### 1.1 Ordinary rows

Foreground talker plus (probabilistically) synthetic interferers plus noise.
This is the baseline path.

#### 1.2 Target-absent rows

Models "no near-field user present". The implementation uses **subtraction**
rather than "never add it":

```
target_in_mix = the foreground's contribution to the mixture (snapshot before interferers)
...synthesise the whole scene normally...
noisy  ← noisy − target_in_mix
target ← 0
```

**Why subtract instead of omitting.** Both the SIR draw (§4) and overlap gating
(§3) need a reference signal to define "how loud relative to the foreground"
and "how much to fill during the foreground's silences". Omitting the
foreground from the start would leave both mechanisms without a reference, and
the interferer levels and activity distribution on such rows would diverge from
every other row. Synthesising normally and subtracting at the end keeps this
row's interferer statistics exactly aligned with ordinary rows.

The subtraction must happen before any level rescaling, otherwise it does not
cancel (a rescaled mixture minus an unrescaled snapshot leaves a residual).

`force_interferer` guarantees the scene is not empty once the foreground is
removed.

#### 1.3 Real-far rows (voice isolation)

The far-field channel is **not convolved with an RIR**. Instead it is a
finished loudspeaker → air → microphone recording drawn from a pool.

Design rationale: a convolved far channel carries only the LTI part of a
capture chain ([ch2](room_acoustics.md) §1). A real end-to-end recording also
carries the non-LTI part — loudspeaker nonlinearity, device AGC behaviour, and
the actual noise floor structure. Using real recordings as the interfering
source is the direct way to cover those components in the training
distribution.

`lone_far_prob` makes a fraction of rows contain only far-field speech with no
near-field talker at all, producing the "no near-field anchor" shape.

#### 1.4 Real-near rows (voice isolation)

The **foreground** is replaced wholesale with a real near-field recording. Such
rows must skip the foreground RIR and also the whole-mix RIR
(`skip_whole_mix_reverb`) — the channel has to stay exactly as recorded, and
convolving another RIR would turn it into "a recording placed in a second
room".

Real-near rows can carry room and speaker labels. Real-far interferers on the
same row then preferentially draw from the **same room** and exclude the same
speaker. The former puts near and far in one acoustic environment; the latter
avoids foreground and interferer being the same person.

#### 1.5 Turn-taking overrides

Real rows may each carry a `turn_taking_prob` override (§3.3). The override
happens at row level rather than in config, so it does not affect the
distribution of any other row — this is how "give only this row type more
far-field monologues" is implemented.

### 2. Interferer sampling and coloring

#### 2.1 Speaker and utterance sampling

* The interferer count comes from `add_n_cases`, either a fixed value or a
  `[lo, hi]` random range.
* The speaker pool excludes the foreground speaker.
* Speakers whose sample rate does not match are handled with **bounded
  retries**: at most 5 redraws, after which that interferer is skipped (the
  count is already random, so one fewer does not invalidate the row).

**Why retries must be bounded.** An unbounded retry loops forever when a
speaker has no utterance at the required sample rate. Once a DataLoader worker
stalls, one DDP rank never reaches the next collective operation and the entire
training run deadlocks. The same lesson appears in the anti-silence crop retry
of `align_audio_list` (bounded at 10) — there the pathological case is a
waveform that is silent throughout, for which no offset can satisfy the
condition.

#### 2.2 Media coloring

Each interferer independently draws against `media_voice.prob` to become a
"media source" (television or loudspeaker playback). The coloring itself is
band limiting plus power-law waveshaping; the mathematics is in
[ch5](spectral_channel.md) §6.

In ordering terms, coloring runs **before** that interferer's RIR: the device
plays the sound first, then it propagates through the room — the correct
physical causality.

Role differences: the on-the-fly simulator places media sources against a wall
([ch2](room_acoustics.md) §8), strengthening their early reflections;
pre-generated banks have no such mechanism, so media and ordinary interferers
share one far-field channel pool and only the coloring differs.

### 3. Overlap gating: who talks when

#### 3.1 The problem

Without any timing treatment, synthetic rows are "two people talking
continuously for the whole row" — both the target and the interferer utterances
are unbroken speech, so the overlap rate is naturally high (above 80%). That
distribution lacks the two situations that decide real behaviour: an interferer
talking during the target's silences, and a far-field talker holding the floor
alone with no near-field talker anywhere in the row.

`OverlapGating` offers two mechanisms. They are not variants of each other and
produce fundamentally different timing structures.

#### 3.2 Per-frame Bernoulli (default)

Each interferer independently draws an overlap regime, then is gated frame by
frame.

**Regime sampling** uses one roll against two cumulative thresholds:

```
roll ~ U(0, 1)
roll < no_overlap_prob                        → p = 0
roll < no_overlap_prob + high_overlap_prob    → p = U(high_overlap_range)
otherwise                                      → p = U(mid_overlap_range)
```

The three regimes exist so a batch spans the whole difficulty spectrum rather
than clustering around the mean. Different interferers on the same row draw
their own regimes, so one row can carry an easy interferer and a hard one at
once.

**Per-frame decisions** use different probabilities on the target's active and
silent frames:

```
target-active frames:  speak with probability p (the drawn overlap rate)
target-silent frames:  speak with probability U(fill_on_silent_range)
```

The reason for separating them: an interferer that only ever speaks while the
target is speaking is a far narrower distribution than real conversation
produces. In reality an interferer continuing through the target's pauses is
the norm, so `fill_on_silent` is usually set higher than the overlap rate.

This mechanism **does not gate the target itself**; it only changes who talks
over it.

#### 3.3 Turn-taking

Near and far alternate in long conversational turns, modelling real turn
exchange.

**Turn script generation** (`_turn_script`) works on the VAD frame grid:

```
fps = fs / hop                                     # frame rate
far_first = (rand < far_first_prob)                # does the row open on a far turn

while pos < n_frames:
    turn_s = U(turn_far_seconds or turn_near_seconds)
    turn_f = round(turn_s · fps)
    corresponding mask[pos : pos+turn_f] = True

    if rand < 0.5:                                  # gap between turns
        pos = end + round(U(turn_gap_seconds) · fps)
    else:                                           # slight boundary overlap
        pos = max(pos+1, end − round(U(turn_overlap_seconds) · fps))

    switch near/far
```

Design points:

* **Separate length ranges for near and far turns.** Natural turn lengths
  differ between the two, and separate ranges let each be controlled
  independently.
* **Gaps and boundary overlaps at 50/50.** Real turn boundaries sometimes leave
  a pause and sometimes overlap; both occur. Gaps alone would leave the model
  never seeing overlap at boundaries.
* **`far_first_prob` decides who opens.** Rows opening on a far turn produce "a
  multi-second far-field monologue with no preceding near-field anchor" — a
  shape the Bernoulli fill can never produce (it does not gate the target, so
  the target is always present from the start), and the hardest opening in a
  streaming context.

**The target's label and its contribution must share one envelope.**
Turn-taking is the only mechanism that gates the target. The implementation
applies the same `near_mask` envelope to `target` (the label source) and
`target_mix` (the target's contribution to the mixture). Gating only one of
them would make the row claim the near-field talker spoke during frames where
the mixture is silent — a self-contradictory training sample.

#### 3.4 The anti-click envelope

A frame mask flipping from 0 to 1 within a single sample is a step
discontinuity in the waveform: audibly a click, spectrally a broadband impulse.
`_Envelope` handles it:

```
1. Upsample the frame mask to the sample rate by hop length (repeat_interleave)
2. Convolve with a normalised Hann window (length 2·fade_samples + 1)
3. Clamp to [0, 1]
```

The Hann window is normalised to unit sum, so within the flat regions of the
mask the convolution still yields 1 (no level change) and only the edges get a
smooth transition. The transition length is `fade_samples`, default 400
samples (25 ms at 16 kHz) — comparable to the syllabic time scale, long enough
to avoid clicks and short enough not to noticeably shift speech onsets.

This operation consumes no randomness.

#### 3.5 The reported overlap_fraction

The reported value is the **realised** one, not the sampled probability: the
intersection of interferer-active and target-active frames, divided by the
target-active frame count. The sampled probability is not the realised value
because the target's own silence distribution affects the outcome.

The turn-taking branch divides by the **gated** target activity. The near-field
talker is silent by construction during far turns, and counting those frames as
"target speaking" would report overlap that is not there.

The Bernoulli branch cannot divide by zero because `apply` returns early when
the target has no active frames at all; the turn-taking branch needs additional
protection, since the near envelope can legitimately miss every target-active
frame.

### 4. Mixing: how loud the foreground is against interferers

#### 4.1 Baseline: hard SIR

```
SIR ~ U(augmentation_speech.snr_range)
```

The mixing mathematics is in [ch4](level_dynamics.md) §2. This is the only mode
for the NS task and the fallback for voice isolation.

#### 4.2 mix_mode (voice isolation)

Samples among several level relationships by probability weight.
`_sample_mix_mode` selects by cumulative weight; weights need not sum to 1.

| Mode | Level relationship |
|---|---|
| `physical` | No relative rescale, signals are summed as they are |
| `distance_level` | Reconstructs the inverse-distance law from scene geometry ([ch3](distance_cues.md) §5) |
| Others (`moderate`, `counter_level`, …) | Each draws from its own explicit `sir_range` |

**An important caveat about `physical`.** The name invites misreading: it
applies no extra rescale, but that does **not** produce a physically correct
1/r level law. Levels have already been flattened twice upstream — the
load-time RMS rescale and the per-channel RIR peak normalisation
([ch2](room_acoustics.md) §3.1). "Summing naturally" therefore lands near 0 dB
SIR, independent of distance. The implementation computes and reports the
realised SIR honestly:

```
realized_SIR = 10 · log10( Σ fg² / Σ itf² )
```

Obtaining a genuine distance level law requires the `distance_level` mode.

#### 4.3 Real-far rows skip mix_mode

Real-far rows always use hard SIR. The level ratio between simulated near-field
speech and a real far-field recording is not physically meaningful — the two
levels are set by different recording and normalisation processes, and applying
a geometric formula would produce a number with no physical referent.

### 5. Residual echo (echo playback)

Models the device's own loudspeaker leaking into its microphone, with residual
left over by an upstream AEC.

The model: another utterance, through a channel of the **same room**, added at
`U(erle_db_range)` dB below the current mixture. The implementation reuses the
SNR mixing machinery directly:

```
noisy = add_bg_noise(noisy, [echo], snr_list=[erle_db])
```

**ERLE** (echo return loss enhancement) is an AEC performance metric: the
proportion of echo energy the AEC removed. Using ERLE as the SNR parameter
means "how much echo is left after the AEC" — an ERLE of 25 dB leaves residual
echo 25 dB below the mixture.

Three constraints:

* **Never in the target.** Echo is something to be removed.
* **Requires source-level reverb**, because it needs its own RIR channel.
* **Channel metadata is deliberately discarded**: echo is not part of the RIR
  lineage.

Known limitation: with a pre-generated bank, `distance_range_override` cannot be
honoured (the bank can only pick the existing channel closest to the requested
band), so the echo channel lands in the far-field pool. A genuinely near-field
echo channel requires the on-the-fly simulator. Physically the echo path should
be very short (device speaker to device microphone), so this limitation makes
the bank path's echo more "distant" than reality.

### 6. The three noise sources

`NoiseStage` provides three non-speech sources. There are three rather than one
because each answers a different question.

#### 6.1 Recorded noise (SNR-relative)

Drawn from a noise corpus and mixed at `U(snr_range)` relative to the current
mixture.

**Dynamic type** (1/4 probability): concatenates two different noise clips and
renormalises, so the noise scene changes within one row.

**Room coloring** (optional): the noise is first convolved with a channel of
the **same room**. Without it, dry noise against wet speech tells the model for
free which is which — reverberation itself becomes a marker for speech. This is
a fix for a data leak, not an extra realism feature.

It is injected through a `noise_transform` callable and convolves **before**
the SNR scaling, so coloring does not affect the final SNR (which is defined on
the convolved noise).

#### 6.2 White noise (SNR-relative)

On rows where the recorded noise did not take the dynamic branch, Gaussian
white noise is added with probability `prob_white_noise` at an SNR relative to
the **current mixture** (which already contains the recorded noise).

Its purpose is cheap broadband coverage that the recorded pool does not always
supply. Real device noise floors are often close to white or pink, whereas
recorded noise pools consist mostly of specific scenes (cafe, street, fan).

#### 6.3 Absolute capture floor

```
level_dbfs ~ U(level_dbfs_range)
floor = randn_like(noisy) · 10^(level_dbfs / 20)
noisy ← noisy + floor
```

The key difference: the level **does not scale with the speech**. A microphone's
self-noise and a room's own tone sit at the same level whoever is talking,
which an SNR-relative source cannot express — a fixed SNR means the noise gets
quieter whenever the speech does.

**"dBFS" is the level as drawn, not as delivered.** The A/D gain staging at the
end of the device chain ([ch6](device_chain.md) §3) rescales the whole row, so
a floor drawn at −45 dBFS on a row that was turned down 6 dB is delivered at
−51 dBFS.

This is **correct** for what it models: capsule self-noise and room tone both
sit upstream of the preamp and therefore follow its gain adjustment. A
converter's own electronic noise would not — that would have to be added after
the A/D, but at roughly −90 dBFS it is more than 40 dB below anything this knob
draws, which is why the pipeline has no such stage.

#### 6.4 Ordering and shared constraints

The order is fixed: **recorded noise → white noise → absolute floor**, all
before the device chain.

Placing the floor before the device chain is also deliberate: capsule
self-noise passes through the device's frequency response just as speech does,
so it should traverse the analogue path together with the speech.

All three are **added to the mixture only, never to the target** — putting
noise in the target would ask the model to reproduce it.

## Engineering

### Config mapping

| Block | Schema | Contents |
|---|---|---|
| `augmentation_speech` | `SpeechAugmentation` | `prob`, `add_n_cases`, `snr_range` (hard SIR), `is_target` |
| `augmentation_speech.media_voice` | `MediaVoiceConfig` | Band limits, compression power range |
| `augmentation_speech.overlap_control` | `OverlapControlConfig` | Three regime probabilities and ranges, `fill_on_silent_range`, `fade_samples`, turn-taking lengths/gap/overlap/`far_first_prob` |
| `augmentation_speech.echo_playback` | `EchoPlaybackConfig` | `distance_range`, `erle_db_range` |
| `augmentation_speech.mix_mode` | `MixModeConfig` / `MixModeEntry` | Mode list and weights (accepted only by the voice_isolation task) |
| `augmentation_target_absent` | `TargetAbsentAugmentation` | `prob`, `force_interferer` |
| `augmentation_realfar` / `augmentation_realnear` | `RealFarAugmentation` / `RealNearAugmentation` | `pool_manifest`, `prob`, `lone_far_prob`, `turn_taking_prob` |
| `augmentation_noise` | `NoiseAugmentation` | `snr_range`, `prob_white_noise`, `white_noise_snr_range`, `room_coloring`, `absolute_floor` |

`mix_mode` raises immediately on a plain NS dataset (checked in the
constructor), because near/far level relationships are implemented only by
voice isolation. This is the enforcement point for the principle "a block
exists only in the task that actually consumes it".

### Ordering

The scene portion of `__getitem__` (the full 16-station table is in
[ch8](engineering_contract.md)):

```
row plan → foreground channel → interferers (media coloring → RIR) → overlap gating
  → mixing (hard SIR / mix_mode) → target-absent subtraction → echo
  → clipping guard → speed → whole-mix RIR → the three noise sources
```

### Two RNG details

**Bernoulli regime sampling consumes a variable number of draws.**
`_draw_overlap_rate` rolls once to select the regime, and only the two non-zero
regimes then draw a value inside their range. The regime thresholds therefore
**cannot be reordered**: swapping `no_overlap_prob` and `high_overlap_prob`
gives the same seed not "a differently weighted distribution" but **entirely
different rows**, because RNG consumption downstream of that point is
misaligned.

**Turn-taking is forced off for target-absent rows**
(`allow_turn_taking=False`). The foreground is subtracted using a snapshot
taken **before** gating (§1.2). Gating the foreground would leave the
subtraction using an ungated snapshot, and the row would retain a residual of
exactly the voice it claims is absent.

### Pitfalls

* `is_target: true` copies the post-gating mixture directly into the target
  (the NS usage "all talkers are the target"). This is mutually exclusive with
  every near/far mechanism in voice isolation and must not be enabled
  alongside them.
* The VAD reference is snapshotted **after** speed perturbation and **before**
  noise and the device chain: the time axis must match the mixture (hence after
  speed), but labels must be computed on clean speech (hence before noise).
* `far_target` (the summed post-gating, post-SIR interferers) is consumed by no
  loss. It is an eval diagnostic output for leakage, not a training target. The
  comment above that field in `ns.py` records its history — it once named a
  loss that does not exist.
* The foreground of a real-near row passes through no RIR at all. When
  whole-mix reverb is also enabled, `skip_whole_mix_reverb` is the only thing
  preventing that row from being convolved a second time; do not bypass it.
