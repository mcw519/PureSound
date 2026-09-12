# RIR scene schema

繁體中文版本：[rir_scene_v2.zh-TW.md](rir_scene_v2.zh-TW.md)

`puresound.audio.rir.scene.schema` defines `rir_scene.v2`. The scene stores
physical inputs—geometry, materials, environment, and transducers—while RT60
is derived from those inputs.

## Schema

| Type | Purpose |
|---|---|
| `MaterialSpectrum` | Frequency values and uncertainty with log-frequency interpolation |
| `SurfaceMaterial` | Absorption, scattering, transmission, provenance, and optional impedance |
| `SceneSurface` | Named room boundary mesh |
| `SurfacePatch` | Window or door material occupying part of a boundary |
| `EnvironmentConfig` | Temperature, humidity, pressure, and sound speed |
| `Pose` | Position and yaw/pitch/roll |
| `TransducerConfig` | Source or receiver identity, pose, pattern, level, and calibration |
| `SceneObject` | Furniture footprint, height, material, and transmission |
| `RoomSceneV2` | Complete validated scene and serialization |

`to_json()` and `from_json()` round-trip the canonical scene.
`to_metadata()` also emits compatibility fields used by existing bank
readers. The `rt60` compatibility field is the median 500/1000 Hz Sabine
prediction; it is never copied from a requested v1 RT60.

## Material catalog

`puresound.audio.rir.scene.materials` contains population priors for common
room finishes. These are simulation priors, not certified measurements of an
installed product.

Room sampling uses correlated room-level and material-level perturbations.
Room-type recipes choose plausible combinations of walls, floors, ceilings,
windows, doors, and furniture.

Surface patches are retained individually in metadata. Shoebox renderers use
their area-weighted effective material.

## Complex impedance

A surface may store real and imaginary impedance in Pa·s/m. Both spectra must
be present at matching frequencies, and real impedance must be non-negative.

Absorption does not determine impedance phase, so the catalog does not invent
impedance for absorption-only materials.

When all components of a patched boundary have impedance, effective
admittance is mixed by area:

```text
Y_effective = sum(area_fraction_i / Z_i)
Z_effective = 1 / Y_effective
```

If one component lacks impedance, effective impedance remains unknown.
Absorption, scattering, and transmission can still use area weighting.

See [Impedance priors](impedance_priors.md) and
[Complex impedance](impedance_measurements.md).

## Rendering behavior

Pyroomacoustics receives frequency-dependent absorption and scattering for
each boundary.

PathEvents additionally supports:

- explicit coherent reflection paths;
- source directivity;
- bounded object transmission and early-path occlusion;
- first-order reference diffraction and scattering.

PathEvents with FDN preserves the coherent early response and adds a
deterministic multiband late field.

These models are intentionally limited: surface patches are not explicit
polygons in shoebox backends, diffraction is not a general mesh solver, and
receiver patterns are only supported where the backend declares them.

## Backend defaults

Defaults belong to command-line entry points, not the scene schema:

| Entry point | Default |
|---|---|
| `egs/rir_generation/generate_hybrid_rir.py` | `pyroomacoustics` |
| `egs/rir_generation/phases/m6_bank/scripts/generate_m6_bank.py` | `path-events-m4` |

Always set the backend explicitly in a release recipe.

## Output level

`HybridRIRConfig.output_mode` supports:

- `calibrated`: preserves source level, distance, receiver calibration, and
  cross-room gain; peak may exceed 1;
- `peak_normalized`: normalizes the complete item to the configured peak.

Mixture normalization is a separate downstream decision and should be
recorded separately.

## Generate data

```bash
python egs/rir_generation/generate_hybrid_rir.py --help
```

Use [RIR generation](../../egs/rir_generation/README.md) for complete recipes.
Generated experiment directories are not source files and should remain
outside version control.
