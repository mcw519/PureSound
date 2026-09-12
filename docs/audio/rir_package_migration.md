# RIR import migration

繁體中文版本：[rir_package_migration.zh-TW.md](rir_package_migration.zh-TW.md)

RIR code now lives under `puresound.audio.rir`. The previous flat modules and
compatibility shims have been removed. Use this page when updating an old
script, notebook, or branch.

## Import map

| Previous module | Current module |
|---|---|
| `puresound.audio.hybrid_rir` contracts | `puresound.audio.rir.contracts` |
| `puresound.audio.rir_scene` | `puresound.audio.rir.scene.schema` |
| `puresound.audio.rir_materials` | `puresound.audio.rir.scene.materials` |
| `puresound.audio.acoustic_impedance` | `puresound.audio.rir.physics.impedance.admittance` |
| `puresound.audio.impedance_measurements` | `puresound.audio.rir.physics.impedance.measurements` |
| `puresound.audio.impedance_fitting` | `puresound.audio.rir.physics.impedance.fitting` |
| `puresound.audio.impedance_modes` | `puresound.audio.rir.physics.impedance.modes` |
| `puresound.audio.fdtd_reference` | `puresound.audio.rir.physics.wave.fdtd` |
| `puresound.audio.low_frequency_modes` | `puresound.audio.rir.physics.wave.low_frequency` |
| `puresound.audio.rir_path_events` | `puresound.audio.rir.path_events` |
| `puresound.audio.rir_metrics` | `puresound.audio.rir.metrics` |
| `puresound.audio.rir_calibration` | `puresound.audio.rir.calibration.loss` |
| `puresound.audio.rir_measurement_campaign` | `puresound.audio.rir.calibration.measured_campaign` |
| `puresound.audio.rir_bank_manifest` | `puresound.audio.rir.bank.schema` |
| `puresound.audio.rir_bank` | `puresound.audio.rir.bank.loader` |
| `puresound.audio.rir_bank_qc` | `puresound.audio.rir.bank.qc` |
| `puresound.audio.spatial_rir` | `puresound.audio.rir.render.spatial` |
| `puresound.audio.binaural_renderer` | `puresound.audio.rir.render.binaural` |

Low- and high-frequency backends are under
`puresound.audio.rir.render.low_frequency` and
`puresound.audio.rir.render.high_frequency`.

## Public entry points

Render a hybrid RIR:

```python
from puresound.audio.rir.render.hybrid import generate_hybrid_rir
```

Use contracts without importing the renderer stack:

```python
from puresound.audio.rir.contracts import HybridRIRConfig
```

Use path events:

```python
from puresound.audio.rir.path_events import PathEvent, render_path_events
```

`puresound.audio.rir.api` re-exports common names for convenience, but it is
not the compatibility boundary. Library code should import from the owning
layer.

## Replacing private helpers

Old helpers exported from `hybrid_rir` now have public names in their owning
modules:

| Previous helper | Current location |
|---|---|
| `_sample_point` | `rir.scene.sampling.sample_point` |
| `_sample_source_in_horizontal_shell` | `rir.scene.sampling.sample_source_in_horizontal_shell` |
| geometry helpers | `rir.scene.geometry` |
| `_hybrid_crossover_with_metadata` | `rir.render.crossover.hybrid_crossover_with_metadata` |
| `_align_high_band_direct` | `rir.render.crossover.align_high_band_direct` |
| `_clip_rir_before_physical_arrival` | `rir.render.crossover.clip_rir_before_physical_arrival` |
| `_solve_modal_ard` | `rir.render.low_frequency.pytard.solve_modal_ard` |
| `_apply_rt60_decay_envelope` | `rir.render.low_frequency.pytard.apply_rt60_decay_envelope` |
| `_coerce_rir_array` | `rir.render.arrays.coerce_rir_array` |

## Dependency layers

```text
api
└─ render / calibration / bank
   └─ path_events / scene / metrics
      └─ physics / contracts
         └─ numpy / scipy
```

`test/test_rir/test_rir_import_boundaries.py` enforces this direction and keeps
contracts and metadata readers free from unnecessary renderer dependencies.
