# Material-first RIR scene schema

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
paths. The backend is selected with
`--high-backend path-events-m3`; Pyroomacoustics remains the CLI default.

Source `speech_cardioid` directivity is evaluated per event from source
orientation and departure direction. Receivers remain omnidirectional in this
backend; unsupported patterns raise instead of silently falling back to omni.

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
and source cardioid through an opt-in PathEvent backend.

Surface patches are still area-weighted for shoebox backends instead of being
explicit polygons. General meshes, frequency-dependent diffraction, measured
source/receiver patterns, and spatial late fields remain later milestones.
