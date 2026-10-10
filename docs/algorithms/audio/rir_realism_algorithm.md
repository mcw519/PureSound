# RIR implementation guide

繁體中文版本：[rir_realism_algorithm.zh-TW.md](rir_realism_algorithm.zh-TW.md)

This page maps the RIR generation pipeline to the modules that implement it and
links the page that explains each stage. Commands are in the
[RIR generation guide](../../../egs/rir_generation/README.md); the package
layering and import rules are in [RIR package layout](rir_package_layout.md).

## Pipeline

```text
RoomSceneV2 (geometry, materials, environment, transducers)
  ├─ low-frequency backend  (wave / modal)          ─┐
  ├─ high-frequency backend (image sources / PathEvents [+ FDN]) ─┤
  └─ causality clip, energy match, LR4 crossover, output level ◄──┘
       ├─ optional: receiver array, FOA, BRIR (spatial renderer)
       └─ WAV + JSON item → bank manifest → QC → release → training loader
```

| Stage | Module | Page |
|---|---|---|
| Data contracts, `HybridRIRConfig`, array layout | `puresound.audio.rir.contracts` | [hybrid RIR](hybrid_rir.md) |
| Scene schema, material catalog, scene sampling | `puresound.audio.rir.scene` | [scene schema](rir_scene_v2.md) |
| Air absorption, impedance, wave reference | `puresound.audio.rir.physics` | below; [impedance priors](impedance_priors.md), [modal validation](modal_validation.md) |
| Coherent reflection paths | `puresound.audio.rir.path_events` | below |
| Backends, crossover, coupling, spatial assembly | `puresound.audio.rir.render` | [hybrid RIR](hybrid_rir.md), [late coupling](rir_late_coupling.md), [FDN](multiband_fdn.md), [spatial](spatial_rir.md) |
| Acoustic metrics and attribution | `puresound.audio.rir.metrics` | [metrics](rir_metrics.md), [attribution](rir_attribution.md) |
| Measured-room calibration | `puresound.audio.rir.calibration` | [measurement campaign](rir_measurement_campaign.md) |
| Bank format, QC, release, loaders | `puresound.audio.rir.bank` | [bank format](rir_bank_v2.md), [loaders](rir_bank.md) |

Only `bank.loader` is imported by training; everything else runs offline when a
bank is built.

## Scene

`RoomSceneV2` stores causes, and RT60 is derived from them: seven-octave
material spectra (125 Hz – 8 kHz), area-weighted patches, sound speed from
temperature and humidity, and a Sabine prediction per octave. The Sabine
estimate assumes a diffuse field and counts only boundaries; it is least
reliable when absorption is concentrated on one surface or furniture dominates.
See [scene schema](rir_scene_v2.md).

## Physics

- **Air absorption** (`physics/propagation.py`). ISO 9613-1 attenuation in dB/m
  from temperature, humidity and pressure. `apply_air_absorption` filters a
  channel with a causal minimum-phase FIR for the direct distance (policy
  `puresound.iso9613_1.minimum_phase_direct.v2`). `num_taps` (129 by default)
  sizes the linear-phase prototype; its minimum-phase half has
  `(num_taps + 1) // 2` taps, 65 by default.
  `air_adjusted_rt60_s` combines material decay with the loss along a path that
  grows at `c`: `60 / (60 / RT60 + a(f) · c)`.
- **Impedance** (`physics/impedance/`). Passive admittance models, phase-aware
  priors, tube reduction, measurement ingestion and fitting, rectangular
  impedance modes. See [impedance priors](impedance_priors.md),
  [impedance measurements](impedance_measurements.md) and
  [impedance-tube protocol](impedance_tube_protocol.md).
- **Wave reference** (`physics/wave/`). A small 3-D FDTD solver used to validate
  modal backends, low-frequency resonance metrics, and the free-field `1/r`
  source convention shared by the bands. See [modal validation](modal_validation.md).

## Path events

A `PathEvent` is one physical arrival — distance, fractional delay, ordered
surfaces, complex gain spectrum — that stays inspectable until the waveform is
assembled; delay and gain are stored separately so propagation phase is never
counted twice.

| Step | Module |
|---|---|
| Ordered shoebox image-source lattice, angle-aware complex boundary gains | `path_events/generator.py` |
| Visibility against vertical-prism objects | `path_events/geometry.py` (`puresound.closed_vertical_prism_segment_visibility.v1`) |
| Opt-in continuous occlusion: each object a Fresnel–Kirchhoff screen on every path leg (`occlusion_model="fresnel_kirchhoff"`) | `path_events/occlusion.py` (`puresound.fresnel_kirchhoff_screen.v1`) |
| Object transmission, bounded knife-edge diffraction, diffuse scattering split `(1 − s)` / `s / N` | `path_events/interactions.py` |
| First-order source/receiver directivity; measured frequency-dependent talker pattern `speech_human` as a band gain | `path_events/directivity.py` |
| Band gains (`PathEvent.band_gain`) realized as minimum-phase FIRs | `path_events/band_filter.py` |
| Rendering with causal forward-Lagrange fractional delays (samples before `floor(delay · fs)` are exactly zero) and passive digital boundary filters; opt-in windowed-sinc delays (`path_events/fractional_delay.py`) | `path_events/renderer.py` (`puresound.causal_forward_lagrange.v1`) |

Boundary filters come from `material_absorption_relaxation_models`: a passive,
causal one-pole admittance fitted to each material's low- and high-frequency
absorption. Absorption alone does not determine reflection phase, so this is a
declared phase prior, not a measured impedance. Known limitations of this
stack and the opt-in corrections are listed in
[Known limitations of the PathEvent stack](path_event_audit.md);
moving sources are rendered by [`render/dynamic.py`](dynamic_scene.md).

## Low-frequency backends

`render/low_frequency/`:

| Backend | Model |
|---|---|
| `pytard.py` (`GpuARDPytARDBackend`, CuPy variant) | pytARD adaptive-rectangular-decomposition solve at an internal rate (16 kHz by default) with a Green-delta source, resampled to the output rate. Its raw amplitude is peak-normalized and `1/r` re-imposed (`calibrate_pytard_signal`), so the crossover energy match sets its level. The lossless solve gets either a broadband RT60 envelope from the direct arrival or, with material damping, a decay per mode |
| `analytic_modal.py` | Rectangular-room modes with reciprocal source/receiver coupling; fast fallback for tests |
| `impedance_modal.py` | Separable complex-impedance modes for passive rational wall admittances, residues calibrated against FDTD |
| `modal_damping.py` | Per-mode amplitude decay rate from boundary participation, `γ = s · c · Σ_axes (α₋ + α₊) / N_axis / 8` with `N_axis = L` for mode index 0 and `L/2` otherwise, capped at 0.95 ω; the recurrence uses the damped frequency `√(ω² − γ²)` and pole radius `exp(−γ Δt)` |

The material-damping law is not validated against measurements and is an
opt-in model; see [modal damping](modal_damping.md).

## High-frequency backends

| Backend | Model |
|---|---|
| `pyroomacoustics.py` | Image sources plus ray tracing with the effective per-boundary absorption and scattering spectra; fixed filter delay removed; furniture applied afterwards as an attenuation ramp over `obstacle_occlusion_recovery_ms` and a scatter tap (`obstacles.py`). Reproducible only with `set_rng_seed` |
| `path_event.py` | Coherent PathEvents up to order 12 with interactions, directivity and air absorption; objects affect individual paths, not the whole channel |
| `fdn.py` | PathEvent early field coupled to a multiband FDN late field ([late coupling](rir_late_coupling.md)) |

## Assembly

`render/crossover.py` clips each band at the physical arrival, aligns the
Pyroomacoustics direct path, matches the low band to the high band around the
crossover and applies the causal fourth-order Linkwitz–Riley crossover;
`render/hybrid.py::generate_hybrid_rir` orchestrates it and applies the tail
fade and output level (`calibrated` or `peak_normalized`). See
[hybrid RIR](hybrid_rir.md). Synchronized arrays, FOA and BRIR are
`render/spatial.py`, `render/spatial_late_field.py` and `render/binaural.py`
([spatial](spatial_rir.md)).

## Policy strings

Strings such as `puresound.iso9613_1.minimum_phase_direct.v2` are stamped into
item metadata and name an observable behaviour. Changing what a stage computes
requires a new string, so a bank records which behaviour produced it.

| Policy | Where |
|---|---|
| `rir_scene.v2`, `puresound-materials.v1` | Scene schema and material catalog |
| `puresound.iso9613_1.minimum_phase_direct.v2` | Air absorption |
| `puresound.causal_forward_lagrange.v1` | PathEvent fractional delay |
| `puresound.fresnel_kirchhoff_screen.v1` | Opt-in continuous object occlusion |
| `puresound.dynamic_geometric_fdn.v2` | Moving-source renderer |
| `puresound.pytard.green_delta.v1` | pytARD excitation |
| `puresound.multiband_fdn.v2`, `puresound.path_event_fdn_coupling.v1` | FDN and coupling |
| `puresound.spatial_room_rir.v1`, `puresound.spatial_late_field.v1`, `puresound.ambisonic_binaural_decoder.v1` | Spatial renderer |
| `puresound.rir_bank.v2`, `puresound.m6_split.sha256_acoustic_space.v1`, `puresound.rir_bank_qc.physical.v1`, `puresound.rir_bank_release.v1` | Bank format |
| `puresound.measured_time_origin.iso3382_onset_to_geometric_arrival.v1` | Measured-RIR ingest |
