# RIR scene schema — `puresound.audio.rir.scene`

繁體中文版本：[rir_scene_v2.zh-TW.md](rir_scene_v2.zh-TW.md)

`RoomSceneV2` (schema string `rir_scene.v2`) describes a shoebox room by its
physical causes — geometry, surface materials, environment, transducers and
furniture. Decay is derived from those causes; it is never an input. This keeps
a scene's RT60 consistent with its absorption, its level with its distances and
source power, and makes every rendered item traceable to a serializable scene.

## Types

| Type | Fields (units) |
|---|---|
| `MaterialSpectrum` | `center_frequencies_hz` (increasing), `values`, optional `uncertainty_std`; `at(f)` interpolates linearly in log2 frequency and holds the edge values |
| `SurfaceMaterial` | `absorption`, `scattering`, `transmission` spectra in [0, 1]; `provenance`; optional `impedance_real`, `impedance_imag` (Pa·s/m) |
| `SceneSurface` | One of the six boundaries `west, east, south, north, floor, ceiling`; vertices (m); base `material_id`; `patches` |
| `SurfacePatch` | Window, door or other patch: `role`, `area_fraction` in [0, 1), `material_id` |
| `EnvironmentConfig` | `temperature_c` (−50..60), `relative_humidity_percent`, `pressure_pa`, `sound_speed_m_s` |
| `Pose` | `position_m` [x, y, z], `orientation_ypr_deg` [yaw, pitch, roll] |
| `TransducerConfig` | `kind` (`source` / `receiver`), `pose`, `directivity_id`, `calibration_gain_db`, `power_db_spl_at_1m` (required for sources), `array_id`, `channel_index` |
| `SceneObject` | Vertical prism: `footprint` polygon (m), `z_min`, `z_max`, `material_id`, broadband `absorption`, `scattering`, `transmission` |
| `RoomSceneV2` | `scene_id`, `room_type`, `dimensions_m`, `surfaces`, `materials`, `environment`, `sources`, `receivers`, `objects`, `catalog_version` |

Construction validates the scene: exactly the six named boundaries, patch
fractions summing to less than one, every referenced material present, objects
inside the room, and no transducer inside an object.

## Derived quantities

**Sound speed.** When `sound_speed_m_s` is not given,
`c = 331.3 + 0.606 · T + 0.0124 · RH` m/s (T in °C, RH in %).

**Effective boundary material** (`effective_boundary_materials`). A boundary with
patches is reduced to one material per boundary by area weighting: absorption,
scattering and transmission are area-weighted means, and their uncertainties
combine as a root sum of squares. Impedance is mixed through admittance, because
patches on one wall see nearly the same pressure and their normal admittances
add in parallel:

```text
Y_eff = Σ_i  a_i / Z_i          (a_i = area fraction)
Z_eff = 1 / Y_eff               (real part clamped to ≥ 0)
```

If any component lacks impedance, the effective impedance stays unknown;
absorption, scattering and transmission are still mixed.

**Predicted RT60** (`predicted_octave_rt60_s`). Sabine per octave band:

```text
RT60(f) = 24 ln(10) · V / (c · Σ_b S_b · α_b(f)),   clipped to [0.02, 20] s
```

with `V` the volume and `S_b`, `α_b` the area and effective absorption of
boundary `b`. Sabine assumes a diffuse field and counts only the boundaries, so
it is metadata and a late-field target, not a guarantee: it is least reliable
when absorption is concentrated on one surface or furniture dominates.

**`rt60`** is the median of the 500 Hz and 1 kHz predictions. It exists so code
written for RT60-driven scenes can read a `RoomSceneV2`; it is never copied from
a requested RT60.

**Transducer gains** (`transducer_channel_gains`). Per source,
`10^((power_db_spl_at_1m − reference + receiver calibration_gain_db) / 20)`,
used by the hybrid renderer's `calibrated` output.

## Serialization

`to_json()` / `from_json()` round-trip the canonical scene; `from_json` accepts a
JSON string or a path. `to_metadata()` adds compatibility fields read by bank
loaders and QC: `room_dim`, `rt60` (with `rt60_origin`), `mic_pos`,
`source_pos`, `source_labels`, `source_distances`, and the `channel_map` list
(`channel`, `label`, `source_pos`, `distance_m`, `horizontal_distance_m`,
`power_db_spl_at_1m`, `directivity_id`). `load_room_scene(dict)` parses a
`rir_scene.v2` mapping and returns any other mapping unchanged.

## Material catalog

`puresound.audio.rir.scene.materials` (`MATERIAL_CATALOG_VERSION =
"puresound-materials.v1"`) holds population priors for common finishes on seven
octave bands, 125 Hz – 8 kHz (`MATERIAL_BANDS_HZ`): brickwork, plasterboard,
wooden lining, glass window, wooden door, 16 mm wood, linoleum, two carpets,
cotton curtain, three ceiling finishes and upholstered furniture. Absorption is
adapted from the Pyroomacoustics materials database; scattering, transmission
and uncertainty are engineering priors, not certified measurements of an
installed product. The catalog stores no impedance: absorption does not
determine reflection phase, so none is invented.

`ROOM_TYPE_RECIPES` gives, for `office`, `meeting_room`, `classroom` and
`living_room`, weighted choices of wall, floor and ceiling finish.
`sample_materialized_shoebox` draws one finish per surface class, one window
patch (6–22 % of a wall) and one door patch (2.5–7.5 % of another wall), and
perturbs each coefficient in logit space:

```text
logit(α') = logit(α) + s · (0.22 · z_room + ε_level + ε_slope · u(f))
```

`z_room ~ N(0, 1)` is shared by every material in the room, so finishes become
more or less absorptive together; `ε_level ~ N(0, 0.18)` and
`ε_slope ~ N(0, 0.12)` are per material, `u(f)` runs from −1 to 1 across the
bands, and `s` is `material_variation_scale`; scattering uses half the room
term. Furniture takes upholstered or wooden material depending on its family.

## Sampling a scene

`sample_material_first_rir_scene(config, seed=...)` samples room geometry,
source and receiver positions and furniture from `HybridRIRConfig`, then calls
`upgrade_hybrid_scene_to_v2`, which adds materials and draws the environment
(18–25 °C, 30–70 % RH, 98–103 kPa), one room-level source power
`N(70, 2)` dB SPL at 1 m plus `N(0, 1.5)` dB per source, source orientation
(uniform yaw, pitch within ±15°) with the `speech_cardioid` pattern, and one
omnidirectional receiver `mic_0`. The RT60 of the input geometry is not carried
over.

In `generate_hybrid_rir.py`, `--scene-version v0` renders RT60-driven
`HybridRIRScene`s and `--scene-version v1` renders material-first `RoomSceneV2`s
(`--room-type`, `--material-variation-scale`).

## How backends use a scene

| Backend | What it reads |
|---|---|
| Pyroomacoustics | Effective per-boundary absorption and scattering spectra, temperature, humidity, sound speed, source directivity; furniture through a post-hoc occlusion model (attenuation ramp from the direct arrival over `obstacle_occlusion_recovery_ms`, plus a scatter tap) |
| PathEvents | Explicit image-source paths with passive boundary filters, source and receiver directivity, object visibility and transmission, first-order edge diffraction and diffuse scattering, air absorption |
| PathEvents + FDN | The above for the early field, plus a multiband late field whose octave targets are the predicted RT60s corrected for air absorption |

Limits: patches are area-weighted, not placed polygons, in the shoebox
backends; diffraction is a bounded first-order model, not a mesh solver; a
receiver pattern is honoured only where the backend declares it.

## Complex impedance

A material may carry real and imaginary surface impedance in Pa·s/m. Both
spectra must be present on the same centers and the real part must be
non-negative (a passive surface). The low-frequency impedance backend uses them;
see [impedance priors](impedance_priors.md) and
[impedance measurements](impedance_measurements.md).
