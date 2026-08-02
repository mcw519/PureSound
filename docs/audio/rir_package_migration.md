# RIR package migration guide

The RIR code moved from flat modules under `puresound/audio/` into the
`puresound.audio.rir` domain package, following
[`RIR_MODULARIZATION_PLAN.md`](../../RIR_MODULARIZATION_PLAN.md) stages R0–R7.

The flat modules are **gone**, and so are the compatibility shims that briefly
stood in for them: every caller in `egs/`, `test/` and `puresound/` now imports
the canonical path. This guide is the old-to-new map, kept because commits,
notebooks and branches predating the migration still reference the old names.

## Why it was worth doing

Two practical reasons, beyond tidiness:

- **Import cost.** `puresound.audio.rir.scene.schema`, `rir.bank.schema` and
  the whole `metrics` package import without `torch`, `torchaudio` or
  Pyroomacoustics, and `test/test_rir_r0_import_boundaries.py` enforces it.
  A manifest reader no longer pays for the renderer stack.
- **Finding things.** `rir_metrics.py` was 1,516 lines covering time, spectrum,
  decay, density and spatial coherence. Those are now five modules whose names
  say which one you want.

## Where everything went

### Contracts and scene

| Old | New |
|---|---|
| `puresound.audio.hybrid_rir.HybridRIRConfig` | `puresound.audio.rir.contracts` |
| `puresound.audio.rir_scene` | `puresound.audio.rir.scene.schema` |
| `puresound.audio.rir_materials` | `puresound.audio.rir.scene.materials` |
| `hybrid_rir` polygon/room geometry | `puresound.audio.rir.scene.geometry` |
| `hybrid_rir` scene sampling, `HybridRIRScene`, `PolygonObstacle` | `puresound.audio.rir.scene.sampling` |

### Physics

| Old | New |
|---|---|
| `puresound.audio.acoustic_impedance` | `puresound.audio.rir.physics.impedance.admittance` |
| `puresound.audio.impedance_priors` | `…physics.impedance.priors` |
| `puresound.audio.impedance_measurements` | `…physics.impedance.measurements` |
| `puresound.audio.impedance_tube` | `…physics.impedance.tube` |
| `puresound.audio.impedance_fitting` | `…physics.impedance.fitting` |
| `puresound.audio.impedance_modes` | `…physics.impedance.modes` |
| `puresound.audio.impedance_residues` | `…physics.impedance.residues` |
| `puresound.audio.fdtd_reference` | `…physics.wave.fdtd` |
| `puresound.audio.low_frequency_modes` | `…physics.wave.low_frequency` |
| `puresound.audio.rir_source_convention` | `…physics.wave.source_convention` |
| `puresound.audio.rir_air_absorption` | `puresound.audio.rir.physics.propagation` |

### Path events

`puresound.audio.rir_path_events` split into
`puresound.audio.rir.path_events.{schema,geometry,interactions,generator,directivity,renderer}`.
The package `__init__` re-exports the same public names, so
`from puresound.audio.rir.path_events import PathEvent, render_path_events`
covers most uses.

### Render

| Old | New |
|---|---|
| `hybrid_rir` low-frequency backends | `puresound.audio.rir.render.low_frequency.{pytard,analytic_modal,impedance_modal,modal_damping}` |
| `hybrid_rir` high-frequency backends | `puresound.audio.rir.render.high_frequency.{pyroomacoustics,path_event,fdn,obstacles}` |
| `hybrid_rir` crossover / alignment / causal clip | `puresound.audio.rir.render.crossover` |
| `hybrid_rir.RIRBackend` | `puresound.audio.rir.render.backend` |
| `hybrid_rir.generate_hybrid_rir` | `puresound.audio.rir.render.hybrid` |
| `puresound.audio.multiband_fdn` | `puresound.audio.rir.render.multiband_fdn` |
| `puresound.audio.rir_late_coupling` | `puresound.audio.rir.render.coupling` |
| `puresound.audio.spatial_late_field` | `puresound.audio.rir.render.spatial_late_field` |
| `puresound.audio.spatial_rir` | `puresound.audio.rir.render.spatial` |
| `puresound.audio.binaural_renderer` | `puresound.audio.rir.render.binaural` |

### Metrics, calibration, bank

| Old | New |
|---|---|
| `puresound.audio.rir_metrics` | `puresound.audio.rir.metrics` (`core`/`temporal`/`spectral`/`density`/`spatial`/`report`) |
| `puresound.audio.rir_attribution` | `puresound.audio.rir.metrics.attribution` |
| `puresound.audio.rir_calibration` | `puresound.audio.rir.calibration.loss` |
| `puresound.audio.rir_inverse_calibration` | `…calibration.synthetic_recovery` |
| `puresound.audio.rir_m4_inverse_calibration` | `…calibration.inverse_m4` |
| `puresound.audio.rir_m5_pipeline` | `…calibration.inverse_m5` |
| `puresound.audio.rir_measured_calibration` | `…calibration.measured_runner` |
| `puresound.audio.rir_measurement_campaign` | `…calibration.measured_campaign` |
| `puresound.audio.rir_constrained_residual` | `…calibration.residual` |
| `puresound.audio.rir_bank_manifest` | `puresound.audio.rir.bank.schema` |
| `puresound.audio.rir_bank` | `puresound.audio.rir.bank.loader` |
| `puresound.audio.rir_bank_qc` | `puresound.audio.rir.bank.qc` |
| `puresound.audio.rir_bank_release` | `puresound.audio.rir.bank.release` |
| `puresound.audio.rir_bank_evaluation` | `puresound.audio.rir.bank.evaluation` |
| `puresound.audio.rir_bank_production` | `puresound.audio.rir.bank.production` |
| `hybrid_rir.write_hybrid_rir_dataset_item` | `puresound.audio.rir.bank.storage` |

## Renamed helpers

Several private helpers that recipes and tests reached for through the
underscore got real names in their new home:

| Old (`hybrid_rir._x`) | New |
|---|---|
| `_sample_point` | `rir.scene.sampling.sample_point` |
| `_sample_source_in_horizontal_shell` | `rir.scene.sampling.sample_source_in_horizontal_shell` |
| `_min_feasible_rt60` | `rir.scene.sampling.min_feasible_rt60` |
| `_obstacle_floor_coverage` | `rir.scene.sampling.obstacle_floor_coverage` |
| `_point_in_polygon`, `_polygons_overlap`, `_polygon_area`, `_polygon_distance`, `_distance_point_to_polygon`, `_distance_point_to_segment`, `_segments_intersect`, `_clip_position_to_room`, `_max_room_distance_from_point`, `_max_room_horizontal_distance_from_point` | `rir.scene.geometry.<same name without underscore>` |
| `_hybrid_crossover_with_metadata` | `rir.render.crossover.hybrid_crossover_with_metadata` |
| `_align_high_band_direct` | `rir.render.crossover.align_high_band_direct` |
| `_clip_rir_before_physical_arrival` | `rir.render.crossover.clip_rir_before_physical_arrival` |
| `_solve_modal_ard` | `rir.render.low_frequency.pytard.solve_modal_ard` |
| `_calibrate_pytard_signal` | `rir.render.low_frequency.pytard.calibrate_pytard_signal` |
| `_apply_rt60_decay_envelope` | `rir.render.low_frequency.pytard.apply_rt60_decay_envelope` |
| `_pytard_green_delta_excitation` | `rir.render.low_frequency.pytard.pytard_green_delta_excitation` |
| `_coerce_rir_array`, `_pad_or_trim` | `rir.render.arrays.coerce_rir_array`, `…pad_or_trim` |
| `_material_modal_decay_rates` | `rir.render.low_frequency.modal_damping.material_modal_decay_rates` |

## New in the package

Two things exist only on the new side:

- **`puresound.audio.rir.api`** — a stable façade re-exporting the 37 names most
  callers need. Convenient, but it loads the renderer stack (including `torch`);
  import a layer module directly if you care about import cost.
- **`puresound.audio.rir.contracts`** — `RIRArray`, `RenderContext`,
  `BackendCapabilities`, `validate_rir_metadata`, `resolve_sound_speed`, and the
  dtype/layout constants. These formalize conventions the code already followed.
  `resolve_sound_speed` in particular resolves a real ambiguity: the causality
  boundary uses the **scene environment** sound speed, not
  `HybridRIRConfig.sound_speed`, and the two differ enough to move the boundary
  by a sample.

## Layering

Dependencies point downwards only:

```text
api  ->  render, calibration, bank  ->  path_events, scene, metrics
     ->  physics, contracts  ->  numpy / scipy
```

`test/test_rir_r0_import_boundaries.py` enforces the direction and keeps
`contracts` free of every project import. If you add a module under
`puresound/audio/rir/`, put it in a layer listed in that test's `LAYER_RANK`.

## What the migration removed

- **35 compatibility shims.** They existed to let the eight migration stages be
  verified one at a time; once the last stage landed they were pure duplication,
  offering a second spelling for every name. `test_rir_r0_api_inventory.py`
  fails if one reappears.
- **The underscore aliases.** `render/hybrid.py` no longer re-exports
  `_solve_modal_ard` and friends; the fifteen private helpers external code used
  are public names in their own modules now. One cross-package private import
  remains — the M6.5 validator's `_paired_t_confidence_interval` — and it is
  recorded in the inventory test.
- **`render/hybrid.py`'s re-export surface.** It exported 22 names for the
  shim's benefit; it now exports one, `generate_hybrid_rir`, which is what an
  orchestration module should own.

Two dead-code candidates are deliberately *not* removed:
`PytARDWaveBackend` and `scene.sampling._sample_source_in_shell`. Both are
recorded in `KNOWN_UNUSED` in the inventory test, which fails if the claim stops
being true in either direction — someone starts using them, or someone deletes
them without updating the record.
