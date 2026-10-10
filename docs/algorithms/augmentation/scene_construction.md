# Scene construction

繁體中文版本：[scene_construction.zh-TW.md](scene_construction.zh-TW.md)

A training row describes a scene: who talks, from where, when, how loud, and
what else the environment contains. This page covers the scene-level stages —
row types, interferer sourcing, overlap gating, foreground/interferer mixing,
residual echo, the ambient lead and the three noise sources.

Code: `puresound/task/ns.py` (the synthesis skeleton,
`NoiseSuppressionDataset.__getitem__`), `puresound/task/voice_isolation.py`
(near/far row types), `puresound/task/session_rows.py` (conversational rows),
`puresound/task/overlap_gating.py`, `puresound/task/noise_stage.py`,
`puresound/audio/noise.py` (`add_bg_noise`).

## Algorithm

### 1. Row types

`_plan_row` decides once, at the start of each row, what kind of row it is and
returns a `RowPlan` (`VoiceIsolationRowPlan` in voice isolation). The skeleton
then branches on the plan's flags:

| Flag | Effect |
|---|---|
| `target_absent` | Foreground removed from the mixture at the end; target zeroed |
| `force_interferer` / `force_speech_interferers` | Interferer block fires without its probability draw |
| `skip_whole_mix_reverb` | No whole-mixture RIR on this row |
| `skip_overlap_gating` | The row brought its own turn script |
| `speed_perturb_companions` | Speed perturbation also applies to `background_speech_reference` |

In voice isolation the order of decisions is **session row → real-near row →
real-far row → synthetic target-absent**; each decision is guarded, so a
disabled block draws nothing.

#### 1.1 Ordinary rows

Foreground talker, synthetic interferers with probability
`augmentation_speech.prob`, then noise. Every other row type is a variation of
this path.

#### 1.2 Target-absent rows

`augmentation_target_absent` (`prob`, `force_interferer`) models "no near-field
user". It uses subtraction rather than omission:

```
target_in_mix = the foreground's contribution to the mixture (before interferers)
... synthesise the scene normally ...
noisy  ← noisy − target_in_mix
target ← 0
```

The SIR (§4) and the overlap gating (§3) both need the foreground as their
reference. Synthesising normally and subtracting at the end keeps the
interferer levels and activity on these rows distributed exactly as on ordinary
rows. The subtraction happens before any rescaling of the mixture (the mixing
step scales only the interferers), so it cancels exactly. `force_interferer`
guarantees the row is not empty once the foreground is gone.

#### 1.3 Real-far rows (voice isolation)

`augmentation_realfar` replaces the synthetic interferers with finished
loudspeaker → air → microphone recordings from a pool manifest, inserted with
**no RIR**. A convolved far channel carries only the LTI part of a capture chain
([room acoustics](room_acoustics.md) §1); a real recording also carries the
parts a convolution cannot — transducer non-linearity, directivity, the actual
noise floor, level and spectral tilt. The foreground keeps its simulated
channel.

* `prob`: share of rows that become real-far rows.
* `lone_far_prob`: of those rows, the share with no near talker at all
  (`target_absent` and `force_interferer`), drawn independently of
  `augmentation_target_absent` so the two pressures can be tuned separately.
* The interferer count comes from `augmentation_speech.add_n_cases`; each
  recording is loaded like corpus speech (RMS level, resampled) and its measured
  `distance_m` is carried as metadata with `origin: real`.

The manifest has one JSON object per line: `wav_path`, `distance_m`, `room`,
`speaker`, `mic` (built by `egs/voice_isolate/scripts/build_real_recording_pool.py`).

#### 1.4 Real-near rows (voice isolation)

`augmentation_realnear` replaces the **foreground** with a genuine close-mic
recording, and the target is that recording itself. The row simulates no room:
no foreground RIR, and `skip_whole_mix_reverb` keeps the whole-mixture RIR off —
a second RIR would place a recording in a second room. Interferers come from the
real-far pool when it is loaded: preferably from the same `room` as the near
recording (when that room has enough entries) and excluding its `speaker`, so
near and far differ mainly in distance and capture-chain identity is useless as
a cue. The target is always present on these rows.

`stitch_to_length` (both real blocks): a pool take shorter than the row would be
zero-padded, which on a keep row makes the target partly digital silence. With
it on, further takes of the same `(speaker, room, mic)` are appended until the
row is covered.

#### 1.5 Session rows (voice isolation)

`augmentation_session_rows` arranges the same machinery as a conversation: one
user, one or two bystanders, and a script of turns, with labels that say which
talker holds each turn. Ordinary rows almost always start with the target in the
first half second, rarely let an interferer hold the floor first, and never
render the same talker twice; session rows supply those cases.

* **Eligibility.** Only rows at least `min_seconds` long (default 12 s); the
  length test comes before the probability draw, so the short length buckets of
  a recipe draw exactly what they would without the block. The script covers at
  most `max_seconds`.
* **Script shapes** (`shape_probs`, weights): `user_first`, `bystander_first`
  (a bystander opens for `bystander_open_seconds`, the wrong-anchor case),
  `user_gap` (the user returns after `user_gap_seconds`; a bystander may talk
  inside the gap with `bystander_in_gap_prob`), and `overlap` (turn boundaries
  overlap: double talk). A shape the row is too short for is demoted
  (`user_gap` → `bystander_first` → `user_first`) and the realised shape is
  reported. Turns alternate with durations from `user_turn_seconds` and
  `bystander_turn_seconds`, separated by `turn_gap_seconds` or overlapping by
  `overlap_seconds` (probability `boundary_overlap_prob`, always in `overlap`).
  A script always has at least one user and one bystander turn.
* **Talkers.** User and bystanders come from the same speaker pool as the
  ordinary foreground, so being "the user" is a role on this row, never a
  property of a voice.
* **Channels.** With source-level reverb, the user goes through a near channel
  in `user_distance_range` (default 0.3–1.0 m); with `rir_move_prob` the later
  user turns use a second near channel of the same room (the user moved).
  Bystanders use `bystander_distance_range` (default 1.5–4.0 m), except that with
  `distance_matched_bystander_prob` one bystander is placed in the user's range,
  so proximity alone cannot identify the user. Each talker is gated by its turn
  mask with the same raised-cosine envelope as turn-taking (§3.4).
* **Level.** The bystander bus is mixed at an SIR from `sir_range`, or with
  `sir_low_tail_prob` from `sir_low_tail_range`, never below the bottom of
  `augmentation_speech.snr_range`. A capture floor from `floor_dbfs_range` is
  then always added to the mixture, so a user gap is room sound, not digital
  silence. A user gap is never labelled target-absent.
* **Labels** (`SESSION_LABEL_KEYS`, on the `vad_target` frame grid, rescaled by
  the row's speed factor): `user_active`, `bystander_active`, `turn_id`
  (1..K per single-talker turn, 0 in double talk; each turn keeps its longest
  contiguous run), `turn_role`, `turn_speaker`, `turn_chain`, `turn_distance`
  and `row_source_id`. Every row of a session-enabled recipe carries these keys
  (a non-session row has no turns and `row_source_id = −1`), plus the
  `SESSION_SCALAR_KEYS` diagnostics.
* **Paired rows** (mutually exclusive): `pair_prob` renders the source material
  from a slot seed inside a scope that restores the RNG streams afterwards, so
  two rows with the same `row_source_id` are the same source through two
  independent chains and noise draws; `paired_view_prob` instead runs a second
  device-chain draw on the identical finished mixture and collates it under
  `paired_view` (rows of at least `paired_view_min_seconds`).

The plan sets `skip_overlap_gating`, `skip_whole_mix_reverb`,
`force_speech_interferers` and `speed_perturb_companions`. An enabled session
block is refused at config load together with `augmentation_speech.is_target:
true` (the bystanders would become target) or an enabled
`augmentation_row_initial_ambient` (the lead would mask speech the script says
is there).

#### 1.6 Row-level turn-taking rates

Real-near and real-far blocks each have an optional `turn_taking_prob` that
overrides `overlap_control.turn_taking_prob` for their rows only (§3.3). The
override is per row rather than in the shared config, so no other row type's
distribution — or RNG consumption — changes.

### 2. Interferers

#### 2.1 Speaker and utterance sampling

* The count comes from `add_n_cases`, a fixed integer or a `[lo, hi]` range,
  clamped to `[1, speakers available]`.
* The speaker pool excludes the foreground speaker.
* A draw at the wrong sample rate is retried at most 5 times, then that
  interferer is skipped. An unbounded retry loops forever on a speaker with no
  utterance at the required rate; a stalled DataLoader worker stops one DDP rank
  from reaching the next collective and deadlocks training. `align_audio_list`
  bounds its search for a non-silent crop (10 tries) for the same reason.

#### 2.2 Media colouring

Each interferer independently becomes a media source (television, loudspeaker
playback) with probability `media_voice.prob`: a band limit from
`hp_cutoff_range` / `lp_cutoff_range` and power-law waveshaping from
`compress_power_range`, RMS restored
([spectral and channel effects](spectral_channel.md)). Colouring runs before the
interferer's RIR — the device plays first, then the room carries the sound. The
on-the-fly simulator places `media` sources against a wall
([room acoustics](room_acoustics.md) §8); a bank serves `media` from the same far
pool as other interferers, so there only the colouring differs.

### 3. Overlap gating

#### 3.1 The problem

Without timing treatment a synthetic row is two people talking continuously for
the whole row, so near-total overlap dominates. That distribution lacks the
cases that decide real behaviour: an interferer talking in the target's pauses,
and a far talker holding the floor alone with no near talker in the row.

`OverlapGating` (`augmentation_speech.overlap_control`) offers two mechanisms
that produce different timing structures. Both run on the frame grid of the
cheap energy VAD built from the `vad_label` block; without that block gating is
disabled.

#### 3.2 Per-frame Bernoulli (default)

Each interferer draws an overlap regime with one roll against two cumulative
thresholds:

```
roll ~ U(0, 1)
roll < no_overlap_prob                        → p = 0
roll < no_overlap_prob + high_overlap_prob    → p = U(high_overlap_range)
otherwise                                     → p = U(mid_overlap_range)
```

then is gated frame by frame:

```
target-active frames:  on with probability p
target-silent frames:  on with probability U(fill_on_silent_range)
```

Three regimes spread a batch over the whole difficulty range, and interferers on
one row draw independently, so one row can carry an easy and a hard interferer.
The separate silent-frame rate exists because real interferers keep talking
through the target's pauses; an interferer that only talks while the target does
is a much narrower distribution. The target itself is not gated.

Defaults: `no_overlap_prob` 0.25, `high_overlap_prob` 0.25,
`mid_overlap_range` [0.1, 0.5], `high_overlap_range` [0.5, 1.0],
`fill_on_silent_range` [0.3, 0.5].

#### 3.3 Turn-taking

With probability `turn_taking_prob` (default 0, per-row overrides in §1.6), near
and far alternate in long turns instead. `_turn_script` writes the script on the
VAD frame grid:

```
fps = fs / hop
far_first = rand < far_first_prob

while pos < n_frames:
    turn = U(turn_far_seconds if far else turn_near_seconds) · fps   (≥ 1 frame)
    mark mask[pos : pos + turn] for the current side
    if rand < 0.5:  pos = end + U(turn_gap_seconds) · fps              # pause
    else:           pos = max(pos + 1, end − U(turn_overlap_seconds) · fps)  # overlap
    switch side
```

* Near and far turn lengths have separate ranges (`turn_near_seconds` default
  [1.5, 3.0] s, `turn_far_seconds` [2.0, 4.5] s).
* Boundaries pause or overlap with equal probability; with pauses only the model
  would never see overlap at a turn boundary.
* `far_first_prob` (default 0.5) rows open on a far turn: a multi-second far
  monologue with no preceding near anchor, which the Bernoulli fill cannot
  produce (it never gates the target) and which is the hardest opening for a
  streaming model.

Turn-taking is the only mechanism that gates the target. The same near envelope
is applied to `target` (the label source) and `target_mix` (the target's
contribution to the mixture); gating only one would claim the near talker spoke
where the mixture is silent. Turn-taking is disabled on target-absent rows: the
foreground is subtracted with a pre-gating snapshot (§1.2), and a gated
foreground would leave a residual of exactly the voice the row claims is absent.

#### 3.4 The anti-click envelope

A frame mask switching within one sample is a step in the waveform: a click and
a broadband impulse. `_Envelope` upsamples the mask by the hop
(`repeat_interleave`), convolves it with a Hann window of length
`2 · fade_samples + 1` normalised to unit sum, and clamps to [0, 1]. Inside flat
regions the gain stays exactly 1; only the edges ramp, over `fade_samples`
(default 400 samples, 25 ms at 16 kHz). It consumes no randomness.

#### 3.5 Reported `overlap_fraction`

The realised value, not the drawn probability: frames where the target is active
and at least one interferer's gate is open, divided by the target-active frames.
The turn-taking branch divides by the **gated** target activity, because the
near talker is silent during far turns by construction. The Bernoulli branch
cannot divide by zero (`apply` returns early on a target with no active frame);
the turn-taking branch guards explicitly, since the near envelope can miss every
active frame.

### 4. Mixing the foreground with the interferers

#### 4.1 Hard SIR

```
SIR ~ U(augmentation_speech.snr_range)
noisy = fg + (rms(fg) / 10^(SIR/20)) · itf / rms(itf)
```

`add_bg_noise` normalises the summed interferer bus to unit RMS over the whole
row and scales it against the foreground's whole-row RMS
([level and dynamics](level_dynamics.md)). The level is set after gating, so a
sparsely gated interferer is louder while it talks. This is the only mode in
noise suppression and the fallback in voice isolation.

#### 4.2 `mix_mode` (voice isolation)

`augmentation_speech.mix_mode.modes` is a list of level relationships picked by
weight (`_sample_mix_mode`; weights need not sum to 1):

| Mode | Level relationship |
|---|---|
| `physical: true` | No relative rescale; signals summed as they are |
| `distance_level: true` | SIR from the inverse-distance law on the row's geometry ([distance cues](distance_cues.md) §5) |
| any other entry | SIR drawn from its own `sir_range` |

**`physical` is not a `1/r` law.** Levels are already flattened upstream — the
load-time RMS rescale and the per-source RIR peak normalisation
([room acoustics](room_acoustics.md) §3.2) — so summing "naturally" lands near
0 dB SIR regardless of distance. The realised SIR is computed and reported:

```
realized_SIR = 10 · log10( Σ fg² / Σ itf² )
```

A true distance level law needs `distance_level`. An enabled `mix_mode` in a
noise-suppression recipe is rejected at config load (and again by the dataset
constructor).

#### 4.3 Rows that skip `mix_mode`

Rows with real-far interferers (real-far rows, and real-near rows when the
real-far pool is loaded) always use the hard SIR: the level ratio between two
differently recorded and normalised signals has no physical referent, and a
geometric formula would produce a number that means nothing.

#### 4.4 Session rows

Session rows mix through `SessionRowBuilder.mix`: the same `add_bg_noise` at the
session's own SIR draw (§1.5), then the forced capture floor on the mixture.

### 5. Residual echo

`augmentation_speech.echo_playback` models the device's own loudspeaker leaking
into its microphone after an upstream AEC. Another utterance goes through a
channel of the **same room** (`distance_range_override = distance_range`,
default 0.2–1.0 m) and is added `U(erle_db_range)` dB below the current mixture
(default 20–35 dB):

```
noisy = add_bg_noise(noisy, [echo], snr_list=[erle_db])
```

ERLE (echo return loss enhancement) is the echo energy an AEC removes, so using
it as the SNR means "how much echo the AEC leaves". Echo is never in the target,
requires source-level reverb (it needs its own channel of the row's room), and
its channel metadata is not reported in the RIR lineage. With a pre-generated
bank the echo channel is requested as an `interferer`, so it comes from the far
pool (the in-band channel if any, else the one nearest the band); a genuinely
short device-speaker-to-microphone path needs the on-the-fly simulator.

### 6. Row-initial ambient lead

`augmentation_row_initial_ambient` opens a row with scene sound only. With
probability `prob`, a lead of `U(lead_seconds_range)` s (default 1–4 s) is masked
out of every speech component — mixture, target and background reference —
with a raised-cosine ramp of `fade_ms` (default 50 ms) at its edge; a lead that
would leave less than 0.5 s of row is skipped. It runs after speed perturbation
and whole-mixture reverb (timings are final) and before the noise stage, which
then fills the lead with the row's own noise and floor; a row whose noise does
not fire gets a silent lead, which is also a real stream start. It is applied to
keep and suppress rows alike, so the lead carries no label information.

### 7. Noise sources

`NoiseStage` (`augmentation_noise`) adds three non-speech sources, because each
answers a different question.

#### 7.1 Recorded noise (relative to the mixture)

With probability `prob`, a noise clip is mixed at an SNR drawn uniformly from
`snr_range`, or piecewise-uniformly when `snr_bands` is set (a band by its
`prob`, then uniform inside it). The SNR is relative to the current mixture's
whole-row RMS (foreground, interferers, echo). Sources: `noise_folder` (every
file equally likely) or `noise_sources` (a corpus by `weight`, then a file
uniformly inside it).

* **Dynamic type** (probability `prob / 4` on rows where recorded noise fires):
  two different clips, each RMS-normalised, concatenated and renormalised, so the
  noise changes within the row.
* **Room colouring** (`room_coloring.prob`, only on rows with a room scene): each
  clip is first convolved with a channel of the **same room** (role
  `interferer`). Without it, dry noise against reverberant speech tells the model
  which is which for free. The convolution happens before the SNR scaling, so
  the SNR is defined on the coloured noise.

The reported `noise_snr` is this SNR, NaN when the source did not fire.

#### 7.2 White noise (relative to the mixture)

On rows where recorded noise fired without the dynamic type, Gaussian white
noise is added with probability `prob_white_noise` at an SNR from
`white_noise_snr_range`, relative to the mixture that already contains the
recorded noise. It is cheap broadband cover that the recorded pool (mostly
specific scenes) does not always provide.

#### 7.3 Absolute capture floor

```
level_dbfs ~ U(absolute_floor.level_dbfs_range)          (default −55 to −35)
noisy ← noisy + randn_like(noisy) · 10^(level_dbfs / 20)
```

The level does **not** scale with the speech. A microphone's self-noise and a
room's tone sit at the same level whoever talks, which an SNR-relative source
cannot express. It fires with `absolute_floor.prob`, independently of the
recorded-noise draw (the `augmentation_noise` block itself must be enabled).

"dBFS" is the level as drawn, not as delivered: the converter at the end of the
[device chain](device_chain.md) gain-stages the whole row, so a floor drawn at
−45 dBFS on a row turned down 6 dB arrives at −51 dBFS. That is correct for what
it models — capsule self-noise and room tone sit upstream of the preamp and
follow its gain. A converter's own electronic noise would not, and would have to
be added after the A/D stage; at roughly −90 dBFS it is far below this range,
which is why there is no such stage.

#### 7.4 Order and shared rules

**Recorded noise → white noise → capture floor**, all before the device chain,
so the floor also passes through the device's frequency response like the speech
does. All three are added to the mixture only, never to the target: noise in the
target would ask the model to reproduce it.

## Engineering

### Config mapping

| Block | Schema | Contents |
|---|---|---|
| `augmentation_speech` | `SpeechAugmentation` | `prob`, `add_n_cases`, `snr_range` (hard SIR, dB), `is_target` |
| `augmentation_speech.media_voice` | `MediaVoiceConfig` | `prob`, `hp_cutoff_range`, `lp_cutoff_range` (Hz), `compress_power_range` |
| `augmentation_speech.overlap_control` | `OverlapControlConfig` | Regime probabilities and ranges, `fill_on_silent_range`, `fade_samples`, `turn_taking_prob`, turn lengths, gap/overlap ranges, `far_first_prob` |
| `augmentation_speech.echo_playback` | `EchoPlaybackConfig` | `prob`, `distance_range` (m), `erle_db_range` (dB) |
| `augmentation_speech.mix_mode` | `MixModeConfig` / `MixModeEntry` | Mode list: `name`, `prob`, `physical`, `distance_level`, `sir_range`, `jitter_db` (voice isolation only) |
| `augmentation_target_absent` | `TargetAbsentAugmentation` | `prob`, `force_interferer` |
| `augmentation_realfar` | `RealFarAugmentation` | `pool_manifest`, `prob`, `lone_far_prob`, `turn_taking_prob`, `stitch_to_length` |
| `augmentation_realnear` | `RealNearAugmentation` | `pool_manifest`, `prob`, `turn_taking_prob`, `stitch_to_length` |
| `augmentation_session_rows` | `SessionRowsConfig` | `enabled`, `prob`, script, distance, SIR, floor and pairing knobs (§1.5) |
| `augmentation_row_initial_ambient` | `RowInitialAmbientAugmentation` | `prob`, `lead_seconds_range` (s), `fade_ms` |
| `augmentation_noise` | `NoiseAugmentation` | `prob`, `noise_folder` / `noise_sources`, `snr_range`, `snr_bands`, `prob_white_noise`, `white_noise_snr_range`, `room_coloring`, `absolute_floor` |
| `vad_label` | `VadLabelConfig` | Frame grid for gating and labels |

The real-row and session blocks exist only in the voice-isolation recipe schema;
config models forbid unknown keys, so a block a task does not consume is rejected
at config load. A curriculum can move the
`prob` values between epochs; the dataset re-derives its stages from the blocks
when they change (`rebind_augmentation_blocks`).

### Ordering

```
load and crop foreground
→ row plan (session / real-near / real-far / target-absent)
→ foreground channel (source-level draw, room scene, full + target RIR)
→ interferers (sampling → media colouring → RIR or real recording)
→ overlap gating → sum → mixing (hard SIR / mix_mode / session) → is_target copy
→ target-absent subtraction → residual echo → clipping guard
→ speed perturbation → whole-mixture RIR → row-initial ambient lead
→ noise (recorded → white → floor) → VAD reference snapshot
→ device chain (+ optional paired view) → crop to row length, labels
```

The full stage table, with every block's RNG behaviour, is in the
[engineering contract](engineering_contract.md).

### RNG details

* **Bernoulli regimes consume a variable number of draws.**
  `_draw_overlap_rate` rolls once and only the two non-zero regimes draw a value.
  Swapping `no_overlap_prob` and `high_overlap_prob` therefore does not just
  reweight the distribution: the same seed produces different rows, because the
  downstream stream is shifted.
* **A row that gates nothing draws nothing**: gating off, no VAD block, no
  interferers, a silent target, or `turn_taking_prob` of 0 all return before
  their draws.
* **Session rows draw in a fixed order**, listed in the `session_rows.py` module
  docstring; paired rows seed only the source-material scope and restore the
  streams afterwards.

### Pitfalls

* `overlap_control` does nothing unless `vad_label` is enabled: the gating
  labeler is built from that block.
* `is_target: true` copies the mixed speech into the target (the
  noise-suppression use: every talker is the target). It contradicts every
  near/far mechanism of voice isolation; do not combine them.
* The VAD reference is the target snapshotted after speed perturbation,
  whole-mixture reverb and the ambient lead (its timing matches the mixture) and
  before the device chain (labels are computed on clean speech; the target never
  carries noise).
* `background_speech_reference` is a pre-speed snapshot except on session rows
  (`speed_perturb_companions`); a per-frame label derived from it on a
  speed-perturbed ordinary row is off by the speed factor.
* The clipping guard divides mixture and target by the same factor when the
  mixture peak exceeds 1, keeping their ratio; `background_speech_reference` is
  not rescaled with them.
* `far_target` (the scaled interferer bus) is consumed by no loss; it is an
  evaluation output for far-speech leakage
  (`egs/voice_isolate/scripts/eval_indomain.py`).
* A real-near row has no room. If the real-far pool is not loaded, its
  interferers fall back to synthetic corpus speech that is aligned but not
  convolved — dry interferers against a real recording. Enable
  `augmentation_realfar` (its `prob` may be 0) alongside `augmentation_realnear`.
* `skip_whole_mix_reverb` is the only thing keeping a real-near or session row
  from a second convolution when whole-mixture reverb is enabled; do not bypass
  it.
