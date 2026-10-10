# Room acoustics (the RIR axis)

繁體中文版本：[room_acoustics.zh-TW.md](room_acoustics.zh-TW.md)

A room turns a dry source into what the microphone receives. This page covers
the physical model of that transformation, where the pipeline gets its room
impulse responses (RIRs), and what the convolution step does to the training
pair beyond convolving.

Code: `puresound/audio/impulse_response.py` (`wav_apply_rir`,
`compute_drr_db`), `puresound/audio/augmentation.py`
(`AudioEffectAugmentor.apply_rir`), `puresound/audio/room_simulator.py`
(`RoomImpulseResponseSimulator`), `puresound/audio/rir/bank/loader.py` (bank
loaders), `puresound/dataset/dynamic_base.py` (per-source reverb helpers).

## Algorithm

### 1. The linear time-invariant model

At small amplitudes the wave equation is linear, and if the room geometry and
boundaries do not change over time the system is also time-invariant. "Room
plus capture" is then an LTI system described by one impulse response `h[n]`,
and the microphone signal is the convolution of the dry source with it:

```
y[n] = (h * x)[n] = Σ_k h[k] · x[n − k]
```

One `h[n]` places any dry utterance at that position in that room without a
recording.

**Covered**: propagation delay, the direct path, early reflections, late
reverberation, the frequency response caused by room modes, and the linear
frequency response of source and microphone.

**Not covered**:

| Phenomenon | Reason | Where it is modelled instead |
|---|---|---|
| Loudspeaker non-linearity, AGC, compression | Not linear | [Level and dynamics](level_dynamics.md), [device chain](device_chain.md) |
| Talker movement, AEC convergence | Not time-invariant | A convolved channel is a stationary talker; session rows switch channels between turns ([scene construction](scene_construction.md) §1.5) |
| Additive noise | Not a channel effect | Noise stage ([scene construction](scene_construction.md) §7) |
| Air turbulence, temperature gradients | Not time-invariant | Negligible over indoor distances |

### 2. The time structure of an RIR

```
amplitude
 │    ┃ direct path (single impulse, t = d/c)
 │    ┃
 │    ┃  ╿ ╿  early reflections (countable specular paths, individually resolvable)
 │    ┃  │ │ ╿ ╷
 │    ┃  │ │ │ │ ╷╷.,·-.,_ late reverberation (dense enough to be statistically
 └────┸──┴─┴─┴─┴─┴──────── an exponentially decaying Gaussian noise)
      0        ~50–80 ms                                             → time
```

* **Direct path**: delay `t = d/c` (`c ≈ 343 m/s`), amplitude `∝ 1/d`.
* **Early reflections**: one or a few specular wall reflections. Reflections
  within about 50 ms fuse with the direct sound into one auditory event
  (precedence or Haas effect) and improve clarity, which is why the clarity
  index C50 uses a 50 ms boundary (ISO 3382-1).
* **Late reverberation**: paths are no longer resolvable and the field
  approaches a diffuse state, uniform in space and decaying exponentially in
  time. RT60 is the time for the level to fall 60 dB after the source stops.

These regions set the two families of time windows used below: the target
windows (§4) and the DRR window (§5).

### 3. What `wav_apply_rir` does around the convolution

`wav_apply_rir(wav, impaulse, sample_rate, rir_mode)` takes `wav` `[C, L]`, an
RIR `[R, L_h]` and `rir_mode ∈ {full, early, direct}`, and returns a signal of
the input length `L` (a multi-channel RIR requires a mono `wav` and returns
`R` channels). Three steps around the FFT convolution are not neutral.

#### 3.1 Window cut

```
peak   = argmax h              (largest positive sample)
direct: h ← h[:, 0 : peak + 6 ms]
early:  h ← h[:, 0 : peak + 50 ms]
full:   h unchanged
```

#### 3.2 Peak normalisation removes the distance level cue

```
h ← h / max|h|
```

This keeps the wet signal at roughly the input level. Each source is convolved
with its own single-channel RIR, so each source is normalised by its own
direct-path peak. In a free field the direct peak is proportional to `1/d`: a
3 m RIR peaks about 9.5 dB below a 1 m one on disk, and after normalisation both
peak at 1.0. **The level difference due to distance is removed.**

Consequences:

* Mixtures do not inherit the `1/r` level law; the relative loudness of near and
  far talkers is set by the mixing stage ([scene construction](scene_construction.md) §4).
* The surviving distance cues are DRR, decay shape and spectral tilt
  ([distance cues](distance_cues.md)).
* The `distance_level` mixing mode reinstates the level cue explicitly
  ([distance cues](distance_cues.md) §5).
* `mix_mode: physical` and every hard-SIR range in the recipes assume this
  normalisation; changing it changes what all of them mean.

#### 3.3 Propagation delay removal

After convolution the output is `y[delay : delay + L]` with
`delay = argmax|h|` of the first RIR channel. The wet signal is sample-aligned
with the dry one, which is what a training pair needs, but the relative time of
flight between sources is removed: a source at 3 m should arrive about 7.3 ms
after one at 0.5 m (`2.5 m / 343 m/s`), and after alignment both start at the
same instant. Timing cues survive only inside the RIR, after its peak.

#### 3.4 The three modes share one normalisation factor

Truncation keeps the direct-path peak, so `max|h|` is the same for `full`,
`early` and `direct` on the same RIR. The `full` mixture and the `early` target
therefore have identical direct-path levels and differ only in reverberant
energy. This is what makes "the target is the near-field part of the mixture"
hold; renormalising after truncation would add an unknown gain between target
and mixture and ask the model to correct gain as well as remove reverberation.

### 4. The target window

`augmentation_reverb.target_rir_type` sets how much reverberation the target
keeps:

| Setting | Target | What the model is asked to do |
|---|---|---|
| `full` | Full RIR convolution | Separate and denoise, no dereverberation |
| `early` | `peak + 50 ms` | Remove late reverberation, keep early reflections |
| `direct` | `peak + 6 ms` | Keep only the direct path and the closest reflections |
| `anechoic` | Dry source | Full dereverberation |

The target is built by re-fetching the mixture's RIR by its `rir_id` and
convolving with the chosen mode (`anechoic` returns the dry source). `early`
follows the perceptual boundary of §2: reflections within 50 ms help
intelligibility, so treating them as distortion makes the task harder for no
benefit. The 6 ms of `direct` is about a 2 m path difference and keeps only
reflections hugging the direct sound, such as from a desk or floor.

### 5. DRR: the direct-to-reverberant ratio

```
DRR_dB = 10 · log10( Σ_{peak ≤ n < peak+w} h[n]²  /  Σ_{n ≥ peak+w} h[n]² )
```

`compute_drr_db(rir, sample_rate, direct_window_ms=2.5)` implements it with
`peak = argmax|h|` and `w = round(2.5 ms · fs)`. Energy before the peak is
ignored; the result is `+inf` when the tail is empty (anechoic or truncated
RIR). Every RIR the simulator or a bank serves carries its `drr_db` in its
metadata.

**Why DRR measures distance.** Direct energy falls as `1/d²`, while in diffuse-
field theory the steady-state reverberant energy density is uniform in the room
and independent of the source position. Hence

```
DRR_dB(d) = 20 · log10(r_c / d)
```

about 6 dB per doubling of distance, where the critical distance `r_c` (DRR =
0 dB) is `r_c ≈ 0.057 · sqrt(Q · V / T60)` m for volume `V` in m³, `T60` in s
and source directivity factor `Q` (Kuttruff, *Room Acoustics*). For offices and
meeting rooms `r_c` is of the order of a metre, which is where the recipes put
the boundary between their near and far distance ranges. This relation is also
why deterministic DRR contrast is written in `log10(d)`
([distance cues](distance_cues.md) §3).

**The two window lengths are not interchangeable.** Target construction uses
6 ms / 50 ms (perceptual fusion); DRR measurement uses 2.5 ms (the direct sound
only, so the value does not depend on where the first reflections land). DRR
contrast cuts its tail at the same 2.5 ms so that its shift lands exactly on
this measure.

### 6. Where RIRs come from

`augmentation_reverb` selects one source at dataset construction
(`DynamicBaseDataset.init_augmentor`), and `apply_rir` serves from it:

| Config | Source |
|---|---|
| `simulator.used: true`, `simulator.pregenerated.used: true` | Pre-generated bank |
| `simulator.used: true`, no enabled `pregenerated` | On-the-fly image-source simulator |
| `simulator` absent or off | Folder of RIR WAV files (`rir_folder`) |

**(a) Pre-generated bank** (`simulator.pregenerated`). A bank scene is one room;
the foreground and every interferer draw their own channel from that scene, so
room consistency is structural. Channel selection by role:
`foreground` draws from the near labels (`near_labels`, default
`near_0, near_1`), `interferer`/`media`/`echo` from the far labels
(`far_0, far_1, far_2`), any other role from all channels; a channel is not
handed out twice in one scene while unused ones remain. A
`distance_range_override` keeps only channels inside the band and falls back to
the channel nearest the band centre. Three loaders:

* `PreGeneratedRoomBank` — a folder of rendered rooms (`bank_type: room`).
* `PreGeneratedReleaseBank` — one recipe of a release manifest
  (`bank_type: release`, the default when `recipe_id` is set). It audits the
  release on construction, can require a production certificate
  (`require_production`), refuses a recipe that is not ready, and samples
  origin, then variant, then item by the recipe's frozen weights. Its `split`
  must equal the dataset's pipeline role, which the dataset injects as
  `usage_role`; this keeps train, validation and test RIRs disjoint.
* `UnionRoomBank` — a `banks:` list served as one pool at per-member `weight`
  (sampling probabilities, not item counts). A curriculum can move the weights
  between epochs (`set_weights`).

Bank formats and generation: [RIR bank](../audio/rir_bank_v2.md); loader API:
[bank loaders](../audio/rir_bank.md).

**(b) On-the-fly simulator** (`RoomImpulseResponseSimulator`). Generated per
call from sampled room dimensions, RT60 and distance (§7–§8). Parameters are
continuous and nothing is stored, at the cost of per-row generation time and
shoebox-only geometry.

**(c) Folder RIRs.** Files drawn uniformly. They carry no metadata (no
distance, RT60 or role), so every geometry-dependent technique skips them, and
they can only be used for whole-mixture reverberation.

**Cache.** Bank and simulator RIRs are stored in an LRU cache of 32 entries
under `rir_id` (`bank-N`, `simulated-N`); folder RIRs are re-read by file key.
The target re-fetches the same impulse by that id and changes only the window
(§3.4, §4). DRR contrast and direct smear are therefore applied before an RIR
enters the cache; applied afterwards, mixture and target would be built from two
different impulses.

### 7. The image-source method

`rir_generator` (Habets' implementation of Allen & Berkley's image-source
method) computes the RIR of a rectangular room. A reflection off a wall is
equivalent to a mirrored source behind that wall; repeated mirroring gives the
higher orders:

```
h(t) = Σ_i  (β^{n_i} / (4π·d_i)) · δ(t − d_i/c)
```

`d_i` is the image-to-receiver distance, `n_i` the reflection count of that
path and `β` the wall reflection coefficient; `1/d_i` is spherical spreading and
`β^{n_i}` the accumulated absorption.

**From RT60 to β.** The config gives RT60, not `β`. With all six surfaces sharing
one absorption coefficient `α`, Sabine's equation

```
RT60 = 0.161 · V / (S · α)
```

(`V` in m³, total surface `S` in m²) gives `α`, and `β = sqrt(1 − α)` (`α` is an
energy coefficient, `β` acts on amplitude). An RT60 shorter than
`0.161 · V / S` would need `α > 1` and the generator raises. Sabine assumes a
diffuse field and is accurate for low absorption; at high absorption Eyring's
formula is closer, so the decay of a very dry simulated room only approximates
the requested RT60.

**Assumptions and their consequences**:

| Assumption | Fails when | Consequence |
|---|---|---|
| Frequency-independent `β` | Real surfaces absorb more at high frequencies | High frequencies decay too slowly; reverberation too bright |
| Specular reflection only | Furniture and rough surfaces scatter | The late tail lacks diffuse-field statistics |
| Rectangular room | Any other geometry, occluding furniture | No diffraction or shadowing |
| Point source, omnidirectional receiver | Mouths and microphones are directional | No direction-dependent colouring |
| No air absorption | Long paths at high frequencies | Too much high-frequency energy in the far field |

These limitations are why the hybrid generator exists: wave-based low
frequencies for correct modal behaviour, geometric high frequencies with
frequency-dependent materials, joined by a crossover
([hybrid RIR](../audio/hybrid_rir.md)).

`hp_filter: true` applies the generator's high-pass that removes the DC
component of the image-source sum; `order: -1` means no limit on reflection
order; `nsample` is the RIR length and defaults to `RT60 · fs` samples.

### 8. Scene geometry sampling

`sample_scene` draws the room and receiver: each dimension uniformly from
`room_dim_range`, RT60 uniformly from `rt60_range`, and the receiver uniformly
inside the room at least `receiver_margin` from every wall. `generate` then
places one source per call according to its role.

**Distance range by role**: `distance_range_override` if given; else
`foreground_distance_range` for `foreground`; for `media`, `media_distance_range`
falling back to `interferer_distance_range`; `interferer_distance_range` for
`interferer`; otherwise `source_receiver_distance_range`.

**Stage one: uniform in the room, with rejection.** A point is drawn uniformly
inside the room (at least `source_margin` from the walls) and kept if its
distance lies in range; up to 64 attempts. This keeps the spatial distribution
uniform, but a narrow range (for example `[0.3, 0.5]` m) is a thin shell that
occupies a tiny fraction of the room, and rejection mostly fails.

**Stage two: shell sampling.** A point at a qualifying distance is constructed
directly (uniform radius, uniform direction) and rejected only against the room
boundary; up to 256 attempts. If the shell does not intersect the room at all,
the source is placed toward the farthest in-room corner with the radius clamped
to the available span — the closest achievable distance.

Shell sampling is not spatially uniform (directions toward near walls are
rejected more often). That is accepted because the realised distance is written
into metadata and consumed downstream as ground truth (deterministic DRR
contrast, `distance_level` mixing, the scalar labels); a wrong distance label is
worse than a placement bias.

**Media wall placement.** `media` models a television or loudspeaker, usually
against a wall. In stage one the source is snapped to within
`source_margin + U(0, media_wall_offset_max)` of an x or y wall. The wall image
is then close to the source, so a strong reflection arrives just after the
direct-path window and the DRR is lower than for a free-standing talker at the
same distance. Stage two drops the wall snap: the distance semantics outrank the
placement.

Returned metadata: `room_dim`, `receiver`, `source`, `rt60`, `source_role`,
`source_receiver_distance` (m) and `drr_db`.

### 9. Further reading

| Document | Contents |
|---|---|
| [Hybrid RIR](../audio/hybrid_rir.md) | Wave-based low band plus geometric high band, crossover |
| [RIR scene](../audio/rir_scene_v2.md) | Material-first scene schema |
| [RIR bank](../audio/rir_bank_v2.md) | Production bank format: generation, QC, splits, release |
| [Bank loaders](../audio/rir_bank.md) | Training-side loaders used in §6 |
| [RIR metrics](../audio/rir_metrics.md) | RT60, DRR, EDC and related measures |
| [Spatial RIR](../audio/spatial_rir.md) | Arrays, FOA and binaural rendering |
| [Audio and RIR index](../audio/index.md) | Every other RIR page: late field, impedance, modal damping, calibration |

## Engineering

### Config mapping

| Block | Schema | Consumer |
|---|---|---|
| `augmentation_reverb` | `ReverbAugmentation` | `init_augmentor`; the whole-mixture branch in `NoiseSuppressionDataset.__getitem__` |
| `augmentation_reverb.simulator` | `RoomSimulatorConfig` | `RoomImpulseResponseSimulator` |
| `augmentation_reverb.simulator.pregenerated` | `PreGeneratedBankConfig` | Bank loaders (room / release / union) |
| `augmentation_reverb.drr_contrast` | `DrrContrastConfig` | [Distance cues](distance_cues.md) §3 |
| `augmentation_reverb.direct_smear` | `DirectSmearConfig` | [Distance cues](distance_cues.md) §4 |

Only the keys a recipe writes are forwarded to the simulator and bank
constructors, so their own defaults apply to the rest (simulator:
`receiver_margin` 0.4 m, `source_margin` 0.4 m, `media_wall_offset_max` 0.2 m,
`sound_speed` 343 m/s, `order` −1, `hp_filter` true; bank: `drr_window_ms` 2.5,
`cache_size` 64).

```yaml
augmentation_reverb:
  used: true
  prob: 1.0
  target_rir_type: early
  simulator:
    used: true
    source_level: true          # one channel per source (required for near/far rows)
    pregenerated:
      used: true
      banks:
        - {name: core, weight: 0.5, bank_type: room, folder: /path/to/bank/core}
        - {name: wide, weight: 0.5, bank_type: room, folder: /path/to/bank/wide}
```

### Per-source and whole-mixture paths

* **Source-level** (`simulator.source_level: true`): one draw per row,
  `torch.rand(1) < augmentation_reverb.prob`, in
  `should_apply_source_level_reverb()`. On a hit the row samples one room scene,
  the foreground takes a `foreground` channel (mixture `full`, target
  `target_rir_type`), and each interferer takes an `interferer` or `media`
  channel of the same scene.
* **Whole-mixture**: on a row without source-level reverb (and whose row plan
  does not set `skip_whole_mix_reverb`), a second draw against the same `prob`
  convolves the finished speech mixture with one RIR after mixing and speed
  perturbation. The RIR comes from the configured source with role `source`
  (all bank channels; the simulator's default distance range). A multi-channel
  result keeps channel 0.
* The two never both apply to one row. With `source_level: true`, a row that
  misses the first draw can still take the whole-mixture path on the second.
* A disabled block consumes no randomness ([engineering contract](engineering_contract.md)).
  RIR selection draws from NumPy (simulator geometry) and Python `random` (bank
  scene and channel), so a seeded item must reseed all three generators.
* RIR lineage (`rir_release_id`, `rir_variant_id`, `rir_renderer_profile_id`,
  and the other `RIR_PROVENANCE_KEYS` in `puresound/task/ns.py`) is emitted per
  row as strings for traceability; no loss reads it.

### Pitfalls

* Folder RIRs have no metadata, so every geometry-dependent technique silently
  skips on that path. When a distance knob seems to have no effect, check the
  RIR source first.
* The cache holds 32 entries. A flow that inserted many other `apply_rir` calls
  between the foreground and its target re-fetch could evict the entry and give
  the target a freshly drawn RIR. The current call order never does.
* Window cuts anchor on `argmax h`, normalisation and alignment on
  `argmax |h|`. They agree when the direct arrival is the largest positive
  sample, as it is for simulated RIRs; an RIR with inverted polarity gets its
  target window anchored on a later sample.
* `_last_rir_meta` holds only the most recent call's metadata and is for REPL
  inspection. Library code takes metadata from the return value,
  `RirApplied.detail.metadata`; the side channel mis-attributes as soon as
  anything else convolves in between.
