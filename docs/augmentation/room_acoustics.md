# Room acoustics (the RIR axis)

繁體中文版本：[`room_acoustics.zh-TW.md`](room_acoustics.zh-TW.md)

A room turns a dry source into what the microphone receives. This chapter
covers the physical model of that transformation, how the pipeline implements
it, and what each seemingly mundane implementation step actually does to the
training data.

## Algorithm

### 1. The linear time-invariant model

Sound propagation in air is governed by the wave equation. At small amplitudes
(linear acoustics) the wave equation is linear, and if the room geometry and
boundary conditions do not change over time the system is also
time-invariant. "Room plus capture" is therefore an LTI system, fully
described by a single impulse response `h[n]`. What the microphone receives is
the convolution of the dry source with that impulse response:

```
y[n] = (h * x)[n] = Σ_k h[k] · x[n − k]
```

This equation is the foundation of the whole RIR axis. Its practical meaning:
obtain one `h[n]` and any dry utterance can be placed into that room at that
position, without recording anything.

**What the model covers**: propagation delay, the direct path, early
reflections, late reverberation, the frequency response caused by room modes,
and the linear frequency response of the microphone and loudspeaker.

**What it does not cover**:

| Phenomenon not covered | Reason | Consequence |
|---|---|---|
| Loudspeaker nonlinearity, AGC, compression | Violates linearity | Needs the separate nonlinear stages in ch4/ch5 |
| Talker movement, AEC convergence | Violates time-invariance | A convolved channel is a "stationary person" |
| Additive noise | Not a channel effect | Added separately by the noise stage (ch7) |
| Air turbulence, temperature gradients | Violates time-invariance | Negligible over indoor distances |

Whether the gap between synthetic data and real recordings falls on the
LTI-explainable side or the other is an experimental question, answered by
measurement records kept internally.

### 2. The time structure of an RIR

A typical room impulse response has three regions along the time axis, divided
by reflection path density:

```
amplitude
 │    ┃ direct path (single impulse, t = d/c)
 │    ┃
 │    ┃  ╿ ╿  early reflections (countable specular paths, individually resolvable)
 │    ┃  │ │ ╿ ╷
 │    ┃  │ │ │ │ ╷╷.,·-.,_ late reverberation (density so high it is statistically
 └────┸──┴─┴─┴─┴─┴──────── equivalent to exponentially decaying Gaussian noise)
      0        ~50–80 ms                                             → time
```

* **Direct path**: travels straight from source to receiver with delay
  `t = d/c` (`d` the distance, `c ≈ 343 m/s`). Its amplitude falls as `1/d`
  (the inverse square law acts on energy; amplitude is its square root).
* **Early reflections**: arrive after one or a few specular wall reflections.
  Perceptually, reflections arriving within about 50 ms fuse with the direct
  sound into a single auditory event rather than being heard as echoes (the
  precedence effect, also called the Haas effect) and they *improve* speech
  clarity. This is why the speech clarity index C50 uses 50 ms as its
  boundary.
* **Late reverberation**: reflection count is high enough that individual paths
  are unresolvable and the sound field approaches a diffuse state — energy
  roughly uniform in space, decaying exponentially in time. The decay rate is
  described by RT60: the time for the sound pressure level to fall 60 dB after
  the source stops.

This three-region structure directly determines the pipeline's two sets of time
windows (see §4 and §5).

### 3. Three side effects of the convolution implementation

`puresound/audio/impulse_response.py::wav_apply_rir` performs the convolution,
but does three things around it that are not mathematically neutral. Their
effect on the training data is larger than that of the convolution itself.

#### 3.1 Peak normalisation removes the distance level cue

```
h ← h / max|h|
```

The purpose is to keep the wet signal at roughly the input signal's magnitude
so downstream levels do not run away. The cost is that distance level
information is erased.

Derivation: in a free field the sound pressure at distance `d` is `p ∝ 1/d`,
and the direct-path peak of the RIR is proportional to that pressure. An RIR
recorded at 3 m therefore has a peak about a third of one at 1 m (roughly
−9.5 dB) as stored on disk. After dividing each channel by its own peak, every
channel's direct path becomes 1.0 — **the level difference due to distance is
removed entirely**.

Scope of the consequences:

* Mixtures do not inherit the 1/r level law. The relative loudness of a near
  and a far talker in the mixture is set by the SIR draw, independent of their
  geometric distance.
* The surviving distance cues are DRR, decay shape, and spectral tilt (full
  list in [ch3](distance_cues.md)).
* Recovering the level cue requires reinstating it explicitly at the mixing
  stage. This is where the `distance_level` mixing mode comes from (ch3 §5).
* `mix_mode: physical` and every hard-SIR range in the recipes rest on this
  premise. Change this line and the semantics of all of them change with it.

#### 3.2 Propagation delay removal erases inter-source arrival differences

After convolution the implementation takes `y[delay : delay + L]`, where
`delay = argmax|h|` (the direct-path peak position).

This leaves the wet signal sample-aligned with the dry one, which is
convenient for building training pairs. The side effect is that **the relative
time of flight between sources is removed**: a source at 3 m should arrive
about 7.3 ms later than one at 0.5 m (`(3−0.5)/343 s`), but after alignment
both utterances start at the same instant. Timing cues survive only inside the
RIR, after its peak.

#### 3.3 All three rir_modes share one normalisation factor

The `early` and `direct` modes work by truncating `h` before convolving:

```
direct: h[0 : peak + 6 ms]
early:  h[0 : peak + 50 ms]
```

Truncation preserves the direct-path peak, so `max|h|` is unchanged and all
three modes get the **same** normalisation factor for the same source RIR. The
result is that the `full` mixture and the `early` target have identical
direct-path levels and differ only in late reverberant energy.

This property is what makes the training assumption "the target is the
mixture's near-field component" hold. If truncation were followed by a fresh
normalisation, an unknown gain difference would appear between target and
mixture, and the model would be asked to perform dereverberation and gain
correction at the same time.

### 4. Choosing the target's time window

`target_rir_type` determines how much reverberation the training target keeps:

| Setting | Target signal | What the model is asked to do |
|---|---|---|
| `full` | Full RIR convolution | No dereverberation (separation/denoising only) |
| `early` | `peak + 50 ms` | Remove late reverberation, keep early reflections |
| `direct` | `peak + 6 ms` | Keep only the direct path, near-full dereverberation |
| `anechoic` | Dry signal | Full dereverberation |

`early` is the default, on the perceptual grounds given in §2: reflections
within 50 ms contribute positively to intelligibility and clarity, so listing
them as "distortion to be removed" is both unnecessary and makes the task
harder. The 6 ms window of `direct` corresponds to roughly a 2 m path
difference and covers only reflections hugging the direct sound, such as from
a floor or desk.

### 5. DRR: the direct-to-reverberant energy ratio

DRR quantifies "how close this source sounds" as an energy ratio:

```
DRR_dB = 10 · log10( Σ_{n∈[peak, peak+w]} h[n]²  /  Σ_{n>peak+w} h[n]² )
```

Implemented in `compute_drr_db` with a default window of `w = 2.5 ms`. It
returns `+inf` when the tail carries no energy (anechoic or truncated RIR).

**Why DRR is a proxy for distance.** Direct-path energy falls as `1/d²`, while
diffuse-field theory says the reverberant energy density in a room is
approximately uniform in steady state and independent of source distance.
Hence

```
DRR(d) ∝ (1/d²) / const   →   DRR_dB ≈ −20·log10(d) + C
```

DRR falls by about 6 dB per doubling of distance, with the constant `C` set by
the room's absorption and volume. This relation is the physical justification
for the `log10(d)` form used by deterministic DRR contrast (ch3 §3).

**Critical distance** is the distance at which `DRR = 0 dB`, i.e. where direct
and reverberant energy are equal. In a typical office or meeting room it lies
around 1–2 m, which also explains why the near/far decision boundary falls in
that interval: beyond the critical distance the reverberant component starts
to dominate.

**The two window lengths are not interchangeable.** Target construction uses
6 ms / 50 ms (following the perceptual fusion boundary); DRR measurement uses
2.5 ms (taking only the direct sound itself, which keeps the measurement
insensitive to where the early reflections happen to land). DRR contrast
deliberately aligns its tail boundary with the 2.5 ms window; the reason is in
ch3 §3.

### 6. The three sources of RIRs

`AudioEffectAugmentor.apply_rir` selects one of three, in priority order. They
are mutually exclusive.

**(a) Pre-generated room bank** (the `pregenerated` block)

An offline-generated multi-source room bank. One bank scene corresponds to one
room, and the foreground plus every interferer draw their own channel from that
same scene, so room consistency is guaranteed structurally rather than by an
extra constraint. `release`-type banks carry a production contract on top:
`recipe_id`, and a `split` that must equal the dataset's `usage_role` (which
prevents train/test sharing the same RIRs), plus QC and production gates. A
`banks:` list serves several banks as one weighted pool (`UnionRoomBank`),
used to combine different generators or different parameter regions for
coverage.

**(b) On-the-fly image source simulator** (`RoomImpulseResponseSimulator`)

Generated per row, with parameters (room dimensions, RT60, distance) sampled
from config ranges. Its advantages are continuously controllable parameters and
no disk footprint; its costs are per-row generation time and shoebox-only
geometry. It is the only source that supports per-role distance overrides
(`distance_range_override`).

**(c) Folder RIRs**

WAV files in a directory, drawn at random. They carry no metadata (no
distance, RT60, or role information), so they can only be used for whole-mix
reverberation and cannot participate in any technique that needs geometry.

**The shared cache mechanism.** RIRs from the first two sources are stored in
an LRU cache (32 entries) under a `rir_id`. The target signal re-fetches the
same RIR by that id and only changes the truncation window. This is the
implementation of "the target is the mixture's near-field component" from
§3.3, and it is why DRR contrast and direct smear must be applied **before**
the RIR enters the cache — applied afterwards, mixture and target would be
built from two different impulse responses.

### 7. The image source method: model and assumptions

`rir_generator` (Habets' implementation) solves the impulse response of a
rectangular room with the image source method (ISM).

**Principle.** A wall reflection is made equivalent to a mirrored source: the
reflection of a source in one wall equals a virtual source at the mirrored
position on the far side of that wall, radiating directly to the receiver.
Multiple reflections correspond to repeated mirroring. The RIR is the sum of
all image source contributions:

```
h(t) = Σ_i  (β^{n_i} / (4π·d_i)) · δ(t − d_i/c)
```

`d_i` is the distance from the i-th image source to the receiver, `n_i` the
reflection count of that path, and `β` the wall reflection coefficient.
`1/d_i` is spherical spreading loss; `β^{n_i}` is the accumulated absorption
over multiple reflections.

**Deriving the reflection coefficient.** Config supplies RT60, not `β`. The
conversion uses Sabine's equation:

```
RT60 = 0.161 · V / (Σ_i S_i · α_i)
```

with `V` the room volume (m³), `S_i` each surface area (m²), and `α_i` each
absorption coefficient. Assuming all six surfaces share one absorption
coefficient, a single `α` can be solved for, and energy conservation gives the
reflection coefficient `β = sqrt(1 − α)` (`α` is defined on energy, `β` acts
on amplitude).

Sabine's equation itself assumes a **diffuse field**: energy uniformly
distributed in the room with equal incidence probability from all directions.
It is reasonably accurate for low absorption (`α < 0.2`) and roughly cubic
proportions; for elongated rooms or one strongly absorbing surface (say
acoustic foam on a single wall), Sabine underestimates RT60.

**Other assumptions and their failure conditions**:

| Assumption | Fails when | Observed deviation |
|---|---|---|
| Frequency-independent reflection coefficient | Real surfaces absorb noticeably more at high frequencies | High-frequency decay is too slow, reverb sounds "too bright" |
| Specular reflection (no scattering) | Furniture, shelves, rough surfaces diffuse the field | Late tail lacks the statistics of a real diffuse field |
| Rectangular room | Any non-rectangular geometry, furniture occlusion | No diffraction or shadowing |
| Point source, omnidirectional receiver | The mouth is directional and microphones have polar patterns | No direction-dependent spectral coloring |
| No air absorption | High frequencies attenuate noticeably over distance | Too much high-frequency content in the far field |

These deviations are why the hybrid RIR generation path exists: low frequencies
from a numerical solution of the wave equation (FDTD) to get modal behaviour
right, high frequencies from a geometric method with frequency-dependent
material absorption, the two bands joined by a Linkwitz–Riley crossover (see
the reading map in §9).

`hp_filter=True` adds a high-pass to remove the low-frequency DC artefact of
ISM discretisation; `order=-1` means the reflection order is unbounded
(computed until the RIR length is exhausted).

### 8. Scene geometry sampling

`RoomImpulseResponseSimulator.sample_scene` samples the room and receiver;
`_sample_source` then places the source according to its role.

**Scene level**: the three room dimensions are each sampled uniformly, RT60
uniformly, and the receiver position uniformly inside the room (keeping
`receiver_margin` from the walls).

**Source level** takes a distance range by role (`foreground_distance_range`,
`interferer_distance_range`, `media_distance_range`, overridable by
`distance_range_override`), then satisfies the distance constraint in two
stages:

*Stage one: uniform-in-room sampling with rejection.* Sample a point uniformly
inside the room (keeping `source_margin` from walls), check whether its
distance to the receiver falls in range, up to 64 attempts. This keeps the
spatial distribution uniform, but when the distance range is narrow (say
`[0.3, 0.5]`) the qualifying spherical shell occupies a tiny fraction of the
room volume and the rejection rate becomes prohibitive.

*Stage two: direct shell sampling.* Switch to constructing a point at a
qualifying distance directly — uniform radius plus uniform direction — with
rejection only against the room boundary, up to 256 attempts. If the shell has
no intersection with the room at all (the distance requirement is
geometrically unsatisfiable), walk toward the farthest in-room corner and clamp
the radius to the available span, giving the closest achievable distance.

The design trade-off in this fallback is worth noting: shell sampling is **not**
spatially uniform inside the room (directions toward nearby walls get rejected
more often). Uniformity is deliberately sacrificed here because distance is
written into metadata and consumed downstream as ground truth by other
techniques (deterministic DRR contrast, `distance_level` mixing). A wrong
label is considerably more harmful than a sampling bias.

**Wall snapping for the media role.** `media` represents a television or
loudspeaker, which is normally placed against a wall. The implementation
snaps the source onto a wall along the x or y axis
(`source_margin + U(0, media_wall_offset_max)`). The physical effect is that
**the path difference of the first reflection shrinks**, making early
reflections stronger relative to the direct sound and therefore giving a lower
DRR than a free-standing talker at the same distance. Note that entering stage
two (shell sampling) abandons wall snapping — again, distance semantics
outrank placement semantics.

### 9. Hybrid RIR generation: reading map

RIR generation has its own in-depth documentation (all under `docs/audio/`,
each with a zh-TW twin), grouped by reading purpose:

**Generation mainline**

| Document | Contents |
|---|---|
| `hybrid_rir` | The low-frequency FDTD (pytARD) plus high-frequency geometric architecture, crossover design |
| `rir_realism_algorithm` | Algorithm-to-code mapping table; useful as an index |
| `multiband_fdn` | Late reverberation synthesis with a multiband feedback delay network |
| `rir_late_coupling` | Energy and timing coupling between the PathEvent early field and the FDN late field |

**Materials and low-frequency physics**

| Document | Contents |
|---|---|
| `modal_damping` | Deriving low-frequency modal damping from material impedance |
| `impedance_priors` | Phase-aware low-frequency impedance priors |
| `impedance_measurements` / `impedance_tube_protocol` | Impedance measurement data and the measurement protocol |
| `complex_impedance_source_audit` | Provenance audit of the impedance data |
| `modal_validation` | Validation methods for modal behaviour |

**Banks and measurement**

| Document | Contents |
|---|---|
| `rir_scene_v2` | Material-first scene schema |
| `rir_bank` / `rir_bank_v2` | Bank loader API, M6 production bank contract |
| `rir_measurement_campaign` | Controlled-room measurement and inverse calibration contract |
| `spatial_rir` | Spatial RIRs and BRIRs |

**Metrics**

| Document | Contents |
|---|---|
| `rir_metrics` | Definitions and implementations of RT60, DRR, EDC and others |
| `rir_attribution` | Energy attribution across direct/early/late |

Known implementation pitfalls (details in `hybrid_rir` and ):
pytARD must run in its lossy mode, since the lossless or Unit settings halve
the usable bandwidth ceiling; pyroomacoustics' RT60 convention and its fixed
40-sample delay need aligning and compensating; the two frequency bands are
joined with a Linkwitz–Riley crossover so that their sum has a flat magnitude.

## Engineering

### Config mapping

| Block | Schema | Consumer |
|---|---|---|
| `augmentation_reverb` | `ReverbAugmentation` | `dynamic_base` init plus the whole-mix branch in `ns.py` |
| `augmentation_reverb.simulator` | `RoomSimulatorConfig` | `RoomImpulseResponseSimulator` |
| `augmentation_reverb.simulator.pregenerated` | `PreGeneratedBankConfig` | Bank loader (room / release / union) |
| `augmentation_reverb.drr_contrast` | `DrrContrastConfig` | [ch3](distance_cues.md) |
| `augmentation_reverb.direct_smear` | `DirectSmearConfig` | [ch3](distance_cues.md) |

Principal knobs:

* `target_rir_type ∈ {full, early, direct, anechoic}` — how much reverberation
  the target keeps (§4).
* `simulator.source_level` — `true` gives every source its own channel (the
  prerequisite for near/far scenes); `false` puts one RIR on the whole mixture.
* Per-role distance ranges — the main knob controlling the near/far
  distribution.
* The release bank's `usage_role == split` hard constraint — prevents RIR
  leakage across splits.

### Ordering and mutual exclusion

* `source_level` rows take the per-source-channel path and **skip** the
  whole-mix RIR branch. Non-source-level rows convolve a whole-mix RIR after
  mixing and speed perturbation instead. The two paths are mutually exclusive,
  decided by one draw per row in `should_apply_source_level_reverb()`.
* The probability draw sits inside the short circuit: a disabled reverb block
  consumes no RNG. See [ch8](engineering_contract.md).
* RIR lineage (`release_id`, `variant_id`, `renderer_profile_id` and others) is
  emitted per sample as strings (`RIR_PROVENANCE_KEYS`) for traceability only;
  it never reaches training.

### Pitfalls

* Folder RIRs have no metadata, so every geometry-dependent technique silently
  skips on that path. This is deliberate (better to do less than to mislabel),
  but during debugging it is easily mistaken for a knob having no effect —
  check the RIR source first.
* The LRU cache holds only 32 entries. If some flow inserted many other
  `apply_rir` calls between the foreground call and the target's re-fetch, the
  entry could already have been evicted and the target would get a freshly
  drawn RIR. The current call order never triggers this, but it is worth
  keeping in mind when adding stages.
* `_last_rir_meta` is a REPL debugging side channel holding only the most
  recent call's metadata. Production paths must take metadata from the return
  value `RirApplied.detail.metadata`, otherwise attribution goes wrong as soon
  as anything else convolves in between.
