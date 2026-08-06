# RIR Algorithm-to-Code Reference

繁體中文版本：`rir_realism_algorithm.zh-TW.md`

This document is the **algorithm ↔ code** reference for `puresound.audio.rir`:
every piece of physics and signal processing is mapped to the module,
function, and policy string that implements it. It describes what the system
**is** and **where it lives**; experiment results, review verdicts, and the
working plan are in [`RIR_EXP_LOG.md`](../../RIR_EXP_LOG.md), and usage
instructions are in
[`egs/rir_generation/README.md`](../../egs/rir_generation/README.md).

Versioned policy strings carried in output contracts (e.g.
`puresound.iso9613_1.minimum_phase_direct.v2`) are behavioral identity tags: an
implementation change bumps the policy version, so bank metadata can tell
apart data produced by different behavior.

---

## 1. System Overview

One synthetic-RIR generation path:

```
Scene sampling (materials first)
  └─ scene/            RoomSceneV2: geometry + surface materials + environment + transducers
Low band (20–1000 Hz)          High band (1000 Hz–Nyquist)
  └─ render/low_frequency/     └─ render/high_frequency/
     ARD/DCT wave solve             pyroomacoustics | PathEvents (M3) | PathEvents+FDN (M4)
     + per-mode material damping
        └────────── render/crossover.py ──────────┘
                causal 4th-order Linkwitz–Riley @ 1 kHz
                + RMS energy match over [700, 1300] Hz
  └─ render/spatial.py         (optional) FOA / synchronized array / binaural
  └─ bank/                     M6 contract: manifest → QC → release → evaluation → production decision
```

Package layout:

| Subpackage | Responsibility |
|---|---|
| `puresound/audio/rir/contracts.py` | Shared contracts: `HybridRIRConfig`, axis-order/dtype constants, crossover gain clamp |
| `puresound/audio/rir/scene/` | Scene schema, material catalog, geometry, sampling |
| `puresound/audio/rir/physics/` | Propagation (air absorption), impedance, FDTD reference solves |
| `puresound/audio/rir/path_events/` | Wave-path events: generation, geometry, boundary interaction, directivity, rendering |
| `puresound/audio/rir/render/` | Low/high-frequency backends, crossover, FDN, spatial rendering, orchestration |
| `puresound/audio/rir/metrics/` | Acoustic measurements: decay, clarity, spectrum, echo density, spatial |
| `puresound/audio/rir/calibration/` | M5 measurement-campaign contract and inverse calibration |
| `puresound/audio/rir/bank/` | M6 training bank: contract, QC, release, evaluation, production decision, measured ingest |

Public entry point: `puresound.audio.rir.api` re-exports the stable surface;
`render/hybrid.py` exposes exactly one public function, `generate_hybrid_rir`
(orchestration). `render/backend.py` defines the backend protocol and
`BackendCapabilities` (including the determinism declaration).

---

## 2. Scene: Materials First (scene/)

Core principle: **the scene records the acoustic causes — geometry, materials,
environment; the acoustic outcome (RT60) is a derived quantity.**

| Concept | Code |
|---|---|
| v2 scene schema | `scene/schema.py` — `RoomSceneV2`: `dimensions_m`, `surfaces` (with patches), `materials`, `environment`, `sources`/`receivers` (`TransducerConfig` + `Pose`), `objects` |
| Material absorption/scattering spectrum | `MaterialSpectrum`: coefficients at 7 octave centers `[125…8000]` Hz; `.at(f)` interpolates, `.to_pra_dict()` converts to the pyroomacoustics format |
| Environment → sound speed | `EnvironmentConfig`: when `sound_speed_m_s` is not given, uses `331.3 + 0.606·T + 0.0124·RH` |
| Surface patch mixing | `RoomSceneV2.effective_boundary_materials()`: patches on the same wall are area-weighted into one equivalent material per boundary |
| Sabine RT60 prediction | `RoomSceneV2.predicted_octave_rt60_s()`: `RT60(f) = (24·ln10 / c) · V / Σᵢ Sᵢ·αᵢ(f)`, clamped to [0.02, 20] s |
| Compatibility field `rt60` | `RoomSceneV2.rt60` property = median of the 500/1000 Hz predictions; metadata tags `rt60_origin: surface_material_sabine_prediction` — **a derived quantity, not a sampled input** |
| Material catalog and room types | `scene/materials.py` — `ROOM_TYPE_RECIPES` (classroom / living room / meeting room / office), `sample_materialized_shoebox()`: one room latent drives the correlated choice of wall/floor/ceiling materials, patches, and furniture |
| v1 → v2 upgrade | `scene/sampling.py` — `upgrade_hybrid_scene_to_v2()`: materializes a v1 geometry into a material-first v2 scene; **the v1 `rt60` is deliberately not copied** (v2's `rt60` is derived from materials) |
| Geometry helpers | `scene/geometry.py` (polygon/room geometry), `schema.shoebox_surface_areas()` |

Known limitations (evidence in the exp log):

- The Sabine prediction assumes a diffuse field sampling all surfaces
  uniformly; when absorption is concentrated on one boundary (e.g. an
  absorptive ceiling with hard walls) it systematically underestimates RT60.
  It is a good **room description**, not a safe direct **render target**
  (`RIR_EXP_LOG.md` §6.6).
- The Sabine sum only includes boundary surfaces; `objects` (furniture) are
  not part of it.
- On the default path (v1→v2 upgrade), `HybridRIRConfig.rt60_range` does not
  constrain the resulting derived-RT60 distribution (`RIR_EXP_LOG.md` §6.6.5).

---

## 3. Low-Frequency Band: Wave Solve + Material Modal Damping (render/low_frequency/)

| Concept | Code |
|---|---|
| ARD/DCT wave solve | `low_frequency/pytard.py` — `GpuARDPytARDBackend` (CPU) / `GpuARDPytARDCuPyBackend` (GPU): a modal/DCT grid `omega[z,y,x]` for a rectangular room, time-stepped with a per-mode recurrence |
| Excitation | policy `puresound.pytard.green_delta.v1`: a single-sample delta (`nonzero_sample_count: 1`), solver band-limited to 1.2 kHz; replaces the legacy bipolar FIR (its fixed comb signature has been retired) |
| Material modal damping | `low_frequency/modal_damping.py` — `material_modal_decay_rates()`: each mode `(nx,ny,nz)` gets a decay rate γ from its **surface-participation**-weighted material absorption at the six walls, capped at `0.95·ω` for stability; model string `surface_participation_sabine` |
| Exact damped recurrence | `pytard.py`: a damped-cosine recurrence with `damped_omega = √(ω²−γ²)` and `pole_radius = exp(−γ·dt)`, applied per mode with no global envelope (`global_rt60_envelope_applied: false`) |
| Causality | policy `zero_samples_before_floor_distance_over_sound_speed`: each channel is zeroed before the geometric-arrival sample (a leading numerical artifact of the finite-voxel solve) |
| Legacy global RT60 envelope | `apply_rt60_decay_envelope()`: a fallback used only in **non**-material-damping mode (`material_modal_damping=False`) |
| Impedance-modal backend | `low_frequency/impedance_modal.py` — `ImpedanceModalLowFrequencyBackend` + `RectangularImpedanceBoundaryConfig`: a modal solve with complex-impedance boundaries (M2 experimental line) |
| Analytic modal backend | `low_frequency/analytic_modal.py`: a lossless analytic reference |

Note: `material_modal_damping_metadata(max_modes=128)` is a **metadata summary
only** — it records only the lowest 128 modes; the solver itself damps the
entire DCT grid. Reading frequency coverage off the metadata leads to the
wrong conclusion (this trap is recorded in `RIR_EXP_LOG.md` §6.6.1).

M2's surface-participation loss law **has not passed its exit gate** (noted in
both the module docstring and `RIR_EXP_LOG.md`): it is infrastructure, not an
accepted production model.

---

## 4. High-Frequency Band: Three Backends (render/high_frequency/)

### 4.1 `pyroomacoustics.py` — Geometric Acoustics (ISM + Ray Tracing)

- v2 scenes go through `pra.Material(absorption.to_pra_dict(), scattering.to_pra_dict())`:
  the **full spectrum** (7 bands) is passed per surface, not a single
  broadband coefficient.
- v1 scene fallback: `pra.inverse_sabine(scene.rt60, room_dim)` back-solves an
  absorption coefficient and an ISM order.
- Air absorption uses pra's built-in `set_air_absorption()` (**not** this
  package's own ISO 9613-1 implementation).
- Determinism warning (documented in the docstring): libroom's ray tracing
  keeps a process-global RNG that per-task seeding cannot reach, so **the
  same seed does not guarantee byte-identical reproduction**.

### 4.2 `path_event.py` — M3 Coherent PathEvents

| Concept | Code |
|---|---|
| Path-event set | `path_events/schema.py` (`PathEventSet`), `path_events/generator.py` (image-source expansion) |
| Geometry kernel | `path_events/geometry.py`: fold/unfold image distances (cross-checked to 1e-11); AABB coarse culling `apply_scene_object_visibility()` (a cheap axis-aligned bounding-box test first, then an exact prism test; invariant: the coarse test never rejects a true intersection) |
| Boundary filtering | `path_events/renderer.py` — `boundary_filter_for(model, incidence_cosine)`: builds a passive, pole-<1 reflection filter from a material admittance model (`physics/impedance/admittance.py`, Cayley transform); memoized on `(model, cosine, fs)` |
| Directivity | `path_events/directivity.py`: first-order pressure patterns such as `speech_cardioid`, applied per path |
| Air absorption | `physics/propagation.py` — `apply_air_absorption()`: **ISO 9613-1** minimum-phase filter, policy `puresound.iso9613_1.minimum_phase_direct.v2` (v2 fixes a humidity-unit bug — ISO works in percent throughout; v1's `h` came out 100x too small; see `RIR_EXP_LOG.md` §4.3b) |
| Causality | a one-sided Lagrange fractional-delay kernel — causal **by construction**, not by post-hoc zeroing |
| Furniture occlusion | `high_frequency/obstacles.py` — model string `direct_early_occlusion_with_diffuse_recovery`: attenuates only the direct-path window and recovers it linearly; the late field is untouched (reduces DRR, not reverberation) |

### 4.3 `fdn.py` — M4 PathEvents Early Field + Multiband FDN Late Field

| Concept | Code |
|---|---|
| Backend | `PathEventFDNHighFrequencyBackend` (inherits M3; the early field is identical to §4.2) |
| Late-field target | `_target_rt60_s_by_hz()`: takes `scene.predicted_octave_rt60_s()` and applies an air-absorption correction (`air_adjusted_rt60_s`, policy `FDN_RT60_adds_sound_speed_times_atmospheric_loss`); lower bound `max(500 Hz, crossover/2)` |
| Multiband FDN | `render/multiband_fdn.py` — `render_multiband_fdn()`: a passive, deterministic FDN; the filterbank is a cascaded binary split, **endpoint-complete** (the top band is a highpass reaching Nyquist) |
| Coupling | `render/coupling.py` — `couple_path_event_rir_with_fdn()`, policy `PATH_EVENT_FDN_COUPLING_POLICY`: crossfades with `equal_power_transition_weights()` over `mixing_time_s` (default 24 ms) after the direct sample; the early field is preserved sample-exact before the transition window |
| Late-field energy anchor | `coupling.py` — `extrapolated_path_tail_energy_target()`: extrapolates the finite PathEvent tail's energy out to the render boundary using the material decay law, and anchors the FDN gain to that extrapolated value, not to the truncated tail sum; premise = the material decay law holds over the extrapolated span — revisit if scene RT60s exceed ~1.5 s |
| Seed | `_channel_seed()`: `blake2b(fdn_seed, scene_id, channel)` — deterministic per channel |

The metadata's `opt_in: true / production_default_changed: false` describes
the `generate_hybrid_rir` layer (the M4/M5 exit gate anchors that layer's
default to pyroomacoustics); the M6 wrapper `generate_m6_bank.py` has
**defaulted to this backend since 2026-08-04** and always passes
`--high-backend` explicitly. The selection rationale (measured decay-shape
comparison) is in `RIR_EXP_LOG.md` §6.6.6.

Known limitation: the FDN late field **hits its target** (measured 0.94–1.00×
of it), so the target's own error becomes the output's error — the Sabine
target cannot respond to the spatial distribution of absorption
(`RIR_EXP_LOG.md` §6.6).

---

## 5. Crossover and Assembly (render/crossover.py, render/hybrid.py)

| Concept | Code |
|---|---|
| Band split | `hybrid_crossover()` / `hybrid_crossover_with_metadata()`: **causal** 4th-order Linkwitz–Riley at `HybridRIRConfig.crossover_hz` (default 1000); causal filtering keeps both bands' leading zeros intact |
| Energy matching | `match_low_band_to_high_band()`: RMS-matches over `effective_crossover_match_band()` (default [700, 1300] Hz), returns `(scaled, gain, raw_gain)`; the gain is clamped to `HybridRIRConfig.crossover_match_gain_range = (1e-4, 8.0)`, and the metadata records `low_band_gain_requested_by_channel` and `low_band_gain_clipped_channels` |
| High-band alignment | `align_high_band_direct()`; after alignment, `clip_rir_before_physical_arrival()` zeroes each channel |
| Tail fade | 20 ms raised-cosine² (`tail_fade` metadata) |
| Causality contract | holds end to end: low band clip → causal LP4 keeps the zero; high band align → clip → causal HP4 keeps the zero; PathEvents are constructive; FDN coupling is sample-exact before the transition (measured in `RIR_EXP_LOG.md` §2.2) |
| Orchestration | `render/hybrid.py` — `generate_hybrid_rir()`: the sole public entry point, composing scene → two bands → crossover → output calibration |
| Output level | `output_mode`: `calibrated` (physical-SPL semantics, peak may exceed 1) or `peak_normalized` |
| Array shape | `render/arrays.py`: the RIR array shape shared across the render layer (axis order `contracts.RIR_AXIS_ORDER`) |

Known root cause (unfixed, recorded in the exp log): the pytARD low band is
**per-item peak-normalized** (`signal/peak · target_peak · 1/distance`), so
its absolute level is not physical; the RMS match is the mechanism that pulls
the low band back to the high band's level, and the (1e-4, 8.0) clamp exists
because of it.

---

## 6. Spatial Rendering (render/spatial*.py, binaural.py)

| Concept | Code |
|---|---|
| FOA / room spatial RIR | `render/spatial.py` — `render_room_scene_spatial_rir()`, policy `SPATIAL_ROOM_RIR_POLICY`; first-order Ambisonics uses **SN3D/ACN** |
| Diffuse-field isotropy | level is solved with **one shared gain** (`one_shared_array_gain_preserves_spatial_ratios`): solving per channel would break the Y/Z/X-to-W isotropy ratio `E[Y²]=E[W²]/3` (−4.77 dB); enforced by the standing validator `egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_foa_diffuse_isotropy.py` |
| Synchronized-array late field | `render/spatial_late_field.py` (M4.5) |
| Binaural | `render/binaural.py` (M4.6, FOA→BRIR, optional) |

---

## 7. Acoustic Metrics (metrics/)

Single entry point `analyze_rir(signal, fs, ...)` returns the full metrics
report; individual functions are also callable on their own.

| Metric | Function | Notes |
|---|---|---|
| Schroeder decay | `schroeder_decay_db` / `noise_compensated_schroeder_decay_db` | the latter compensates for the noise floor |
| T20/T30 | `estimate_decay_time` → `DecayEstimate` (includes fit R²) | interpretation rule: check the fit R² threshold before trusting the seconds value |
| C50/C80 | `clarity_db` | |
| DRR | `compute_drr_db`, `direct_sample` | direct window defaults to 2.5 ms |
| Spectral tilt | `spectral_tilt_db_per_octave` | validity limits against measured references: `RIR_EXP_LOG.md` §4.4 |
| Noise floor | `estimate_noise_floor_lundeby` → `NoiseFloorEstimate` | Lundeby iteration; needs ≥15 dB of dynamic range |
| Echo density / mixing time | `analyze_echo_density`, `abel_normalized_echo_density_profile`, `estimate_abel_mixing_time` (policy `ABEL_ECHO_DENSITY_POLICY`) | |
| Octave banding | `octave_band_rir`, `valid_octave_centers`, `DEFAULT_OCTAVE_CENTERS_HZ` | |
| Multiband late field | `analyze_multiband_late_field` (policy `MULTIBAND_LATE_FIELD_POLICY`) | M4.2 contract |
| Spatial | `analyze_binaural_iacc` (`IACC_POLICY`), `analyze_array_spatial_coherence`, `diffuse_field_coherence` (`DIFFUSE_FIELD_COHERENCE_POLICY`) | only meaningful for synchronized receiver arrays |

Low-frequency-specific: `physics/wave/low_frequency.py` (modal peak/Q
estimation), `physics/wave/fdtd.py` (an independent 3-D FDTD reference solve,
for cross-checking modal validation).

---

## 8. Measured RIR Ingest (bank/measured_ingest.py)

Prepares published-corpus measured RIRs into a `measured` variant that can
pass M6 QC. Core problem: M6 defines `t=0` as the emission instant, while
published corpora reference their own direct sound as the origin — ingest
re-inserts the propagation delay the corpus removed; the room response itself
is never touched.

| Concept | Code |
|---|---|
| Alignment policy | `MEASURED_TIME_ORIGIN_POLICY = puresound.measured_time_origin.iso3382_onset_to_geometric_arrival.v1` |
| Onset criterion | ISO 3382-1: scanning **forward from the start**, the first crossing above peak−20 dB (and above the noise floor +20 dB); the criterion was chosen against BRUDEX as the calibration standard (that corpus carries its own propagation delay, so a correct criterion must demand zero shift for it) |
| Alignment | `align_measured_channel()`: shifts the channel so the onset lands at `floor(d/c·fs) + fade`, mutes everything before that, and a 4-sample raised-cosine fade-in touches only sub-threshold samples |
| Rejection criteria | `earlier_arrival` (an earlier arrival **separated** from the onset by a gap — level alone can't decide this, since the pre-onset region legitimately carries a rising edge), `removed_energy` (>1% of energy discarded), `implausible_shift` (delay >100 ms or advance >250 ms) |
| Sound-speed assumption | corpora don't publish temperature/humidity → `ASSUMED_MEASURED_ENVIRONMENT` (20 °C/50% → 344.04 m/s) is written into every item's scene and tagged `provenance: assumed`; QC recomputes the arrival from the same number |
| Packaging | `build_measured_m6_bank()`: one renderer profile per corpus (the measurement chain **is** the renderer), `signal_variant=measured`, `level_policy=native_measured`; rejected items are recorded in the ingest report, not the manifest |

CLI: `egs/rir_generation/phases/m6_bank/scripts/ingest_measured_m6_variant.py`.
Selection rationale and validation data (pre-arrival energy 44–74 dB above the
noise floor, five-corpus run results) are in `RIR_EXP_LOG.md` §5.

---

## 9. Calibration (calibration/)

| Concept | Code |
|---|---|
| M5 measurement-campaign contract | `calibration/measured_campaign.py`: controlled room, repeated ESS, room/transducer/environment/asset all sha-addressed, room-disjoint split (`RIRMeasurementCampaign` and related dataclasses + `audit_measurement_campaign`) |
| Calibration runner | `calibration/measured_runner.py` |
| Multi-objective loss | `calibration/loss.py`: multi-resolution STFT distance + direct-relative EDC distance + causality/pre-arrival penalty |
| M4 inversion | `calibration/inverse_m4.py`: `M4_PROFILE_CONVERGENCE_POLICY = puresound.m4_profile_convergence.stable_minimum.v1` — convergence is defined as "the fit reaches a stable minimum" (relative tolerance 1e-3, verified by restarting from `result.x`), not scipy's success flag |
| M5 inversion / residual | `calibration/inverse_m5.py`, `calibration/residual.py` (constrained residual model), `calibration/synthetic_recovery.py` (synthetic recovery self-check) |
| Impedance measurement | `physics/impedance/`: `measurements.py`, `tube.py` (impedance-tube protocol), `fitting.py`, `priors.py`, `modes.py`, `residues.py` |

---

## 10. M6 Training Bank (bank/)

Six layers, each fail-closed, fully content-addressed (a single
`canonical_json_sha256`: sort_keys + fixed separators + `allow_nan=False`).

| Layer | Code | Key points |
|---|---|---|
| Contract | `bank/schema.py` | `RIRBankManifest`/`RIRBankItem` (WAV+JSON pairs sharing a stem, per-file sha); `BankSplitPolicy`: `sha256(policy_id, seed, acoustic_space_id)` decides the split — room- and acoustic-space-disjoint, deterministic; `BankGeneratorProvenance` (code revision, config sha, task-plan sha); `BankRendererProfile` (evidence tier: `development/empirical_candidate/production_approved`); libsndfile `PEAK`-chunk timestamp zeroing (`canonicalize_float_wav_header`) makes float-WAV bytes reproducible |
| Generation | `egs/rir_generation/generate_hybrid_rir.py` | the task plan is materialized before execution, seeding is per item, resume is pinned to the code revision (a version change never blends old and new items), `generation_run` records `elapsed_seconds`/`items_failed` (a worker exception aborts the run before the manifest is written, so `failed=0` is a design fact rather than an unverified claim) |
| QC | `bank/qc.py` | `RIRBankQCPolicy` (versioned thresholds: pre-arrival ≤1e-7 relative peak, arrival error ≤1 ms, tail energy, T20 fit R²≥0.7 coverage, octave-decay coverage, echo density, finiteness/shape/identity); `run_rir_bank_qc` produces a pass-only candidate index plus a quarantine index; `audit_rir_bank_qc_release` **recomputes** every hash and membership |
| Release | `bank/release.py` | `build_m6_variant_release`: `synthetic_calibrated`/`synthetic_peak_normalized` variants + `real_native`/`mixed_calibrated_real` recipes (honestly `blocked` without a measured bank); lineage is verified sample-by-sample (child = parent × common gain); `prune_bank_to_qc_passed`: a release variant must be entirely pass — the pruned copy goes into the release, the unpruned bank stays as a record |
| Evaluation | `bank/evaluation.py` | `compare_release_distributions` (variant scale-invariance + synthetic↔measured normalized Wasserstein); `validate_throughput_report` / `validate_listening_report` (the empirical tier **recomputes** the estimate and the paired-t CI; fewer than 20 participants is never empirical) / `validate_downstream_report` (recomputes the CI lower bound); `evaluate_m6_release` separates the implementation exit from the empirical exit |
| Production decision | `bank/production.py` | 13 `PRODUCTION_DECISION_CHECK_NAMES`; `audit_m6_production_evidence` (bundle, three role sign-offs, per-profile approval — all bound to the release/evaluation hashes); `validate_m6_production_certificate` **recomputes** the decision components, so the certificate cannot be forged |
| Evidence generation | `bank/evidence.py`, `bank/listening.py` | **producers** of approvals/sign-offs/bundles/throughput (all fail-closed: no named approver, no timing, or a missing file is rejected); listening-assignment design (room-disjoint, blinded labels, hidden reference/degraded anchor) and response ingest; a dry run goes through the `contract_fixture` tier plus `explicitly_not_human_responses` and **produces no response data** |
| Reading | `bank/loader.py`, `bank/storage.py` | `PreGeneratedReleaseBank`: split is required, `split == usage_role` is cross-checked, a missing manifest fails closed; provenance travels with every training sample (`rir_release_sha256`, etc.) |

Note that the two `EVIDENCE_TIERS` vocabularies differ: renderer profiles use
`(development, empirical_candidate, production_approved)` (`bank/schema.py`),
while the listening/evaluation contract uses `(contract_fixture, empirical)`
(`bank/evaluation.py`).

Approval must be stamped into the manifest **before** QC (the QC summary is
bound to the manifest hash), so release is a two-pass process; the driving
CLI is `egs/rir_generation/phases/m6_bank/scripts/build_m6_evidence.py`
(evaluate → approve → attest).

---

## 11. Invariants and Test Mapping

| Invariant | Verified at |
|---|---|
| AABB coarse culling never rejects a true intersection | `test/test_rir_path_events.py` (conservativeness invariant) |
| Boundary filter is passive, poles <1 | path_events tests (Cayley passivity) |
| FOA diffuse isotropy within ±1 dB | `validate_foa_diffuse_isotropy.py` (includes adversarial counter-examples) |
| Causality contract end to end | M6 item QC `prearrival_energy` gate + crossover tests |
| Same-parameter fresh run is byte-reproducible | `RIR_EXP_LOG.md` §2.1 (measured record); `validate_m6_reproducible_generation.py` |
| ISO 9613-1 numerically correct | `test/test_rir_air_absorption.py`: checked against tabulated standard values plus an independently transcribed double check |
| BRUDEX zero-shift (measured alignment) | `test/test_rir_measured_ingest.py` (includes a forward-scan regression guard) |
| Listening statistics are recomputable | `test/test_rir_m6_evidence.py` (producer output run through the validator's recomputation) |
| API surface frozen | `test/test_rir_r0_api_inventory.py` (`__all__`, cross-package private imports, dead-code declarations) |
| Import layering | `test/test_rir_r0_import_boundaries.py` |
| M6 certificate cannot be forged | `test/test_m6_production_decision_validator.py` |

---

## 12. Known Limitations and Open Items (see the exp log)

| Topic | Location |
|---|---|
| Validity of Sabine as the FDN late-field target | `RIR_EXP_LOG.md` §6.6 |
| Low band runs fast (0.62–0.79×, shared by both arms) | §6.6.7 |
| pyroomacoustics high-frequency decay runs too long (~2.2×) | §6.6.6 |
| `rt60_range` does not constrain v1→v2 sampling | §6.6.5 |
| pytARD low-band level is non-physical (root cause: peak normalization) | §4.2 |
| Measured references are not usable for tilt/noise floor | §4.4, §4.5 |
| Remaining M6 production-decision blockers (human listening, downstream training) | §6 |
