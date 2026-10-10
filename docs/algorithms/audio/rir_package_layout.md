# RIR package layout — `puresound.audio.rir`

繁體中文版本：[rir_package_layout.zh-TW.md](rir_package_layout.zh-TW.md)

How the RIR package is layered, which module to import for each job, and the
rules that keep the layers apart. What each stage computes is in the
[implementation guide](rir_realism_algorithm.md).

## Layers

Dependencies point only downwards; a module may import its own layer or any
layer below it.

| Rank | Layer | Holds |
|---|---|---|
| 4 | `api` | Convenience re-exports |
| 3 | `render`, `calibration`, `bank` | Backends and assembly; measured-room calibration; bank storage, manifest, QC, release, loaders |
| 2 | `scene`, `path_events`, `metrics` | Scene schema and sampling; coherent paths; acoustic metrics |
| 1 | `physics` | Propagation, impedance, wave solvers |
| 0 | `contracts` | Data conventions, `HybridRIRConfig`, `RIRArray`, `BackendCapabilities` |

`contracts` imports only the standard library and NumPy: no Torch, no
renderer, no bank module, no filesystem access. Modules that must stay
importable without Torch, Pyroomacoustics or CuPy include the scene, metrics and
path-event layers, `physics.propagation`, the FDN and coupling modules, and the
bank schema, QC, release, evaluation and production modules — a manifest reader
must not need the rendering stack. `bank/__init__.py` imports nothing for the
same reason. `test/rir/test_rir_import_boundaries.py` enforces both rules
(its `LAYER_RANK` table and lightweight-module list) and fails the build on a
violation.

Only `bank.loader` is imported by training. Changing any other module affects
no training run until a bank is rebuilt and re-released.

## Entry points

| Job | Import |
|---|---|
| Render a hybrid RIR | `from puresound.audio.rir.render.hybrid import generate_hybrid_rir` |
| Configuration and data contracts, without the renderer stack | `from puresound.audio.rir.contracts import HybridRIRConfig, RIRArray, validate_rir_metadata` |
| Scenes | `from puresound.audio.rir.scene.schema import RoomSceneV2`; sampling in `scene.sampling` |
| Path events | `from puresound.audio.rir.path_events import PathEvent, render_path_events` |
| Backends | `render.low_frequency`, `render.high_frequency` |
| Crossover helpers | `render.crossover` (`hybrid_crossover_with_metadata`, `align_high_band_direct`, `clip_rir_before_physical_arrival`) |
| Spatial output | `render.spatial.render_room_scene_spatial_rir` |
| Metrics | `from puresound.audio.rir.metrics import analyze_rir` |
| Bank manifest (Torch-free) | `bank.schema` |
| Training loaders | `bank.loader` (`PreGeneratedRoomBank`, `PreGeneratedReleaseBank`, `UnionRoomBank`) |
| Writing one item | `bank.storage.write_hybrid_rir_dataset_item` |

`puresound.audio.rir.api` re-exports a common subset for convenience. It is not
a stability boundary and loads the whole renderer stack, Torch included; library
code imports from the owning layer.
