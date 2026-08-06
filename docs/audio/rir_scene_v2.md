# Material-first RIR scene schema

繁體中文版本：`rir_scene_v2.zh-TW.md`

`puresound.audio.rir.scene.schema` defines the versioned `rir_scene.v2` metadata used
by the M1 generator. The scene stores physical inputs rather than a requested
broadband RT60:

```text
six named surface meshes + material spectra + window/door area patches
  + temperature / humidity / pressure / sound speed
  + source and receiver pose / pattern / calibration
  + interior objects
```

`RoomSceneV2.to_json()` and `RoomSceneV2.from_json()` round-trip the canonical
representation. `to_metadata()` additionally emits `room_dim`, `mic_pos`,
`source_pos`, `channel_map`, and a material-derived `rt60` compatibility value,
so existing `PreGeneratedRoomBank` readers continue to work.

## Schema types

| Type | Purpose |
|------|---------|
| `MaterialSpectrum` | Values and uncertainty at octave or third-octave centers; interpolation is logarithmic in frequency. |
| `SurfaceMaterial` | Absorption, scattering, transmission, provenance, and optional complex impedance in Pa·s/m. |
| `SceneSurface` | Named polygon mesh for one of `west/east/south/north/floor/ceiling`. |
| `SurfacePatch` | Area fraction and material of a window or door on a boundary. |
| `EnvironmentConfig` | Temperature, relative humidity, pressure, and derived or explicit sound speed. |
| `Pose` | Position plus yaw/pitch/roll orientation. |
| `TransducerConfig` | Source/receiver identity, pose, directivity, level, calibration, and array identity. |
| `SceneObject` | Furniture closed-polygon footprint, height interval, family, material, absorption/scattering, and energy transmission. |
| `RoomSceneV2` | Complete scene, validation, serialization, effective wall mixtures, and predicted octave RT60. |

The `RoomSceneV2.rt60` property is only a migration bridge: it is the median
500/1000 Hz Sabine prediction from the serialized surfaces. It is never sampled
or copied from the v0 requested-RT60 field.

## Material catalog

`puresound.audio.rir.scene.materials` contains `puresound-materials.v1`. The initial
absorption priors are an inspectable subset of the Pyroomacoustics 0.10.1
materials database. PureSound adds explicit uncertainty, scattering, and
transmission engineering priors. These are population priors for simulation,
not certified measurements of a particular installed product.

The catalog covers:

- masonry, plasterboard, and wood wall constructions;
- hard, wood, and carpeted floors;
- hard and absorptive ceilings;
- glazing, wooden doors, curtains, and upholstered/wood furniture.

Each sampled room shares a room-level latent absorption shift, then receives a
smaller material-specific level and spectral-slope perturbation. Room-type
recipes correlate finishes:

| Room type | Typical correlation |
|-----------|---------------------|
| `office` | plasterboard/brick walls, carpet or linoleum, usually acoustic ceiling tile |
| `meeting_room` | plasterboard/wood walls, mostly carpet, absorptive ceiling |
| `classroom` | brick/plasterboard, mostly hard floor, mixed hard/acoustic ceiling |
| `living_room` | plaster/brick/wood walls, carpet or wood floor, mostly hard ceiling |

Every room also receives one window patch and one door patch on different
walls. The current shoebox backend uses the area-weighted material spectrum of
the base wall and patches. Their component materials and fractions remain in
metadata for a later explicit mesh/path backend.

## Complex impedance contract

`SurfaceMaterial.impedance_real` and `impedance_imag` store the real and
imaginary parts of surface impedance in SI units of Pa·s/m. Both spectra must
be present, share frequency centers, and have a non-negative real part for a
passive surface. `impedance_at(frequency_hz)` performs the same log-frequency
interpolation as the other material spectra.

Absorption does not uniquely determine impedance because it specifies only the
magnitude of a reflection coefficient, not its phase. The material catalog
therefore leaves impedance absent until a phase-aware measurement or an
explicitly versioned prior is available. It does not silently synthesize
impedance from the absorption table.

When every component of an area-patched boundary has declared impedance,
`effective_boundary_materials()` mixes normal admittance in parallel by area:

```text
Y_effective = sum(area_fraction_i / Z_i)
Z_effective = 1 / Y_effective
```

If any component has unknown impedance, the effective impedance remains
unknown. Absorption, scattering, and transmission continue to use their
existing area-weighted mixtures.

The phase-aware references in [`impedance_priors.md`](impedance_priors.md) are
kept in a separate experimental catalog. They are not automatically mapped to
`SurfaceMaterial`: a 100 mm rigid-backed glass-wool reference is not a valid
substitute for an unspecified carpet, ceiling tile, or installed wall system.

## High-frequency rendering

For a v2 scene, `PyroomacousticsHighFrequencyBackend` constructs one
frequency-dependent `pra.Material` per boundary, including both absorption and
scattering spectra. Temperature and humidity are passed to the room, and the
same scene sound speed is used by both hybrid bands for direct-path alignment.

M3 also provides an explicit opt-in `PathEventHighFrequencyBackend`. It treats
each `SceneObject` as a closed vertical prism, tests every segment of the
source/reflections/receiver polyline for visibility, and keeps blocked events
inspectable in metadata. A blocked direct path may create:

- a straight-through event with pressure gain equal to the square root of the
  product of object energy transmissions;
- up to two shortest visible vertical-edge diffraction detours per blocker;
- first-order deterministic wall-scattering branches whose `1-s` and `s/N`
  energy partitions sum exactly to the parent event.

The interaction model is intentionally scoped: it is not a general triangle
mesh or full UTD implementation. Diffraction is a bounded 1 kHz reference
model, and controlled scattering currently applies only to first-order wall
paths.

Source `speech_cardioid` directivity is evaluated per event from source
orientation and departure direction. Receivers remain omnidirectional in this
backend; unsupported patterns raise instead of silently falling back to omni.

M4 (`PathEventFDNHighFrequencyBackend`) subclasses the M3 backend and replaces
its sparse late field with a deterministic multiband feedback-delay network
(FDN) tail. The coherent M3 early response is rendered first and then
crossfaded into the FDN tail with an equal-power transition centered at a
configurable mixing time (`--fdn-mixing-time-ms`, default 24 ms) over a
configurable transition duration (`--fdn-transition-ms`, default 16 ms). The
FDN's per-band RT60 targets come from the scene's own predicted octave decay
rather than a fixed constant, its delay-line count is a power of two
(`--fdn-delay-lines`, default 16), and it is seeded deterministically per
room/source (`--fdn-seed`). A positive-quadratic-root gain solve keeps the
post-transition energy equal to what the original PathEvent response would
have carried, so M4 changes the late-field character without changing the
direct/early samples that M3 already produced.

### There is no single default backend

There is no single "the default backend" for this system — the default
depends on which entry point is called:

| Entry point | Flag | Choices | Default |
|---|---|---|---|
| `generate_hybrid_rir.py` (low-level generator) | `--high-backend` | `pyroomacoustics`, `path-events-m3`, `path-events-m4` | `pyroomacoustics` |
| `generate_m6_bank.py` (M6 one-command wrapper) | `--backend` | `pyroomacoustics`, `path-events-m4` | `path-events-m4` |

`generate_hybrid_rir.py`'s own default is deliberately pinned to
`pyroomacoustics` and is not expected to change: the M5 measured-calibration
exit gate checks the parsed default backend's type, the M4 coupling
validator's `production_default_unchanged` check reads the FDN metadata, and a
dedicated test (`test_m6_emission_is_opt_in_and_default_backend_is_unchanged`)
asserts the parsed default directly. Flipping this lower layer would
retroactively rewrite already-closed milestones' exit contracts, so instead
the change was made one layer up.

`generate_m6_bank.py` — the recommended entry point for training data, see
[`egs/rir_generation/README.md`](../../egs/rir_generation/README.md) — defaults
its own `--backend` to `path-events-m4` (since commit `3525008`,
2026-08-04) and always passes `--high-backend` explicitly to the low-level
script on every invocation, so the M6 default lives entirely in the wrapper
and never touches `generate_hybrid_rir.py`'s own default. The change was made
on measured evidence: against 1465 measured RIRs from five corpora, M4's
normalized octave decay shape deviates from measured decay by `0.081` where
Pyroomacoustics deviates by `0.628` (`RIR_EXP_LOG.md` §6.6.6) — Pyroomacoustics's
high-frequency reverberation runs roughly 2.2x too long at 2 kHz, the wrong
direction entirely relative to every one of the five reference corpora.
Pyroomacoustics remains available at both layers as a first-class explicit
choice and as the A/B arm; the cause of its overshoot remains an open question.

With `record_realized_metrics=True`, every output sidecar records broadband and
valid octave-band DRR, C50/C80, EDT, T20, T30, fit quality, and spectral tilt
using `puresound.audio.rir.metrics`. Predicted material decay and realized RIR
decay are separate fields.

## Calibrated output

`HybridRIRConfig.output_mode` has two modes:

- `peak_normalized` preserves v0 behavior and scales the entire five-channel
  item to `normalize_peak`;
- `calibrated` never performs per-item peak normalization. Source SPL at 1 m
  and receiver calibration are converted to per-channel amplitude gains
  relative to the 94 dB SPL (approximately 1 Pa) reference.

This preserves distance, source-level, and cross-room gain differences. A
downstream recipe may normalize the convolved training mixture, but that is a
separate, recorded decision.

## Generator

The M1 reference configuration is
[`egs/rir_generation/phases/m1_material/config/material_v1.json`](../../egs/rir_generation/phases/m1_material/config/material_v1.json).

```bash
python egs/rir_generation/generate_hybrid_rir.py \
  --output-dir egs/rir_generation/exp/rir_realism/m1/hybrid_rir_material_v1_16k \
  --scene-version v1 --room-type mixed \
  --output-mode calibrated --record-realized-metrics \
  --n-rooms 1000 --rir-per-room 10 \
  --sample-rate 16000 --duration 1.6 \
  --low-backend pytard --num-workers 22
```

The old command remains v0 by default. This is intentional: selecting
`--scene-version v1` is an explicit bank-version change.

## Current boundary

M1 made the high-frequency decay material-first. M2 added experimental
per-mode and complex-impedance low-frequency paths. M3 now renders furniture
visibility, scalar reference-frequency transmission/diffraction/scattering,
and source cardioid through an opt-in PathEvent backend. M4 adds a
deterministic multiband FDN late field to that same mono/per-channel hybrid
generator (opt-in, see above), and — through the separate
`puresound.audio.rir.render.spatial` API described in
[`spatial_rir.md`](spatial_rir.md) — synchronized receiver arrays, first-order
Ambisonics, and an optional binaural decoder.

Surface patches are still area-weighted for shoebox backends instead of being
explicit polygons. General meshes, frequency-dependent diffraction, and
measured (rather than idealized first-order) source/receiver directivity
patterns remain later milestones.
