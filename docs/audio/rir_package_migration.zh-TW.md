# RIR package 遷移指南

English version: `rir_package_migration.md`

RIR 相關程式碼，已經從 `puresound/audio/` 底下的扁平模組（flat module），
搬進 `puresound.audio.rir` 這個領域套件（domain package），依照
[`RIR_EXP_LOG.md`](../../RIR_EXP_LOG.md) 所記錄的 R0–R7 階段進行。

那些扁平模組現在**已經不存在**，連曾經暫時頂替它們的相容性 shim 也一起
消失了：`egs/`、`test/` 與 `puresound/` 裡的每個呼叫端，現在都改用新的
canonical path 匯入。這份指南就是舊名到新名的對照表，之所以保留，是因為
遷移之前的 commit、notebook 與分支，仍然會引用舊名稱。

## 為什麼值得做這件事

除了讓程式碼更整齊之外，還有兩個實際的理由：

- **匯入成本。** `puresound.audio.rir.scene.schema`、`rir.bank.schema`，
  以及整個 `metrics` package，匯入時完全不需要 `torch`、`torchaudio` 或
  Pyroomacoustics，而且 `test/test_rir_r0_import_boundaries.py` 會強制
  檢查這件事。一個只是要讀 manifest 的程式，不再需要為整套 renderer stack
  的匯入成本買單。
- **找東西。** 舊的 `rir_metrics.py` 有 1,516 行，同時涵蓋時間域、頻譜、
  衰減、density 與空間 coherence。現在拆成五個模組，每個模組的名字就
  說明了你要找的是哪一種指標。

## 東西都搬去哪了

### Contracts 與 scene

| 舊 | 新 |
|---|---|
| `puresound.audio.hybrid_rir.HybridRIRConfig` | `puresound.audio.rir.contracts` |
| `puresound.audio.rir_scene` | `puresound.audio.rir.scene.schema` |
| `puresound.audio.rir_materials` | `puresound.audio.rir.scene.materials` |
| `hybrid_rir` 的多邊形／房間幾何 | `puresound.audio.rir.scene.geometry` |
| `hybrid_rir` 的 scene sampling、`HybridRIRScene`、`PolygonObstacle` | `puresound.audio.rir.scene.sampling` |

### Physics（物理層）

| 舊 | 新 |
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

`puresound.audio.rir_path_events` 拆成了
`puresound.audio.rir.path_events.{schema,geometry,interactions,generator,directivity,renderer}`。
這個 package 的 `__init__` 會 re-export 同樣的公開名稱，所以
`from puresound.audio.rir.path_events import PathEvent, render_path_events`
就能涵蓋大多數用法。

### Render

| 舊 | 新 |
|---|---|
| `hybrid_rir` 的低頻 backend | `puresound.audio.rir.render.low_frequency.{pytard,analytic_modal,impedance_modal,modal_damping}` |
| `hybrid_rir` 的高頻 backend | `puresound.audio.rir.render.high_frequency.{pyroomacoustics,path_event,fdn,obstacles}` |
| `hybrid_rir` 的 crossover／對齊／causal clip | `puresound.audio.rir.render.crossover` |
| `hybrid_rir.RIRBackend` | `puresound.audio.rir.render.backend` |
| `hybrid_rir.generate_hybrid_rir` | `puresound.audio.rir.render.hybrid` |
| `puresound.audio.multiband_fdn` | `puresound.audio.rir.render.multiband_fdn` |
| `puresound.audio.rir_late_coupling` | `puresound.audio.rir.render.coupling` |
| `puresound.audio.spatial_late_field` | `puresound.audio.rir.render.spatial_late_field` |
| `puresound.audio.spatial_rir` | `puresound.audio.rir.render.spatial` |
| `puresound.audio.binaural_renderer` | `puresound.audio.rir.render.binaural` |

### Metrics、calibration、bank

| 舊 | 新 |
|---|---|
| `puresound.audio.rir_metrics` | `puresound.audio.rir.metrics`（`core`／`temporal`／`spectral`／`density`／`spatial`／`report`） |
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

## 改過名字的 helper

有幾個 recipe 與測試原本透過底線（underscore）直接呼叫的私有 helper，
搬新家之後都有了正式名稱：

| 舊（`hybrid_rir._x`） | 新 |
|---|---|
| `_sample_point` | `rir.scene.sampling.sample_point` |
| `_sample_source_in_horizontal_shell` | `rir.scene.sampling.sample_source_in_horizontal_shell` |
| `_min_feasible_rt60` | `rir.scene.sampling.min_feasible_rt60` |
| `_obstacle_floor_coverage` | `rir.scene.sampling.obstacle_floor_coverage` |
| `_point_in_polygon`、`_polygons_overlap`、`_polygon_area`、`_polygon_distance`、`_distance_point_to_polygon`、`_distance_point_to_segment`、`_segments_intersect`、`_clip_position_to_room`、`_max_room_distance_from_point`、`_max_room_horizontal_distance_from_point` | `rir.scene.geometry.<去掉底線的同名函式>` |
| `_hybrid_crossover_with_metadata` | `rir.render.crossover.hybrid_crossover_with_metadata` |
| `_align_high_band_direct` | `rir.render.crossover.align_high_band_direct` |
| `_clip_rir_before_physical_arrival` | `rir.render.crossover.clip_rir_before_physical_arrival` |
| `_solve_modal_ard` | `rir.render.low_frequency.pytard.solve_modal_ard` |
| `_calibrate_pytard_signal` | `rir.render.low_frequency.pytard.calibrate_pytard_signal` |
| `_apply_rt60_decay_envelope` | `rir.render.low_frequency.pytard.apply_rt60_decay_envelope` |
| `_pytard_green_delta_excitation` | `rir.render.low_frequency.pytard.pytard_green_delta_excitation` |
| `_coerce_rir_array`、`_pad_or_trim` | `rir.render.arrays.coerce_rir_array`、`…pad_or_trim` |
| `_material_modal_decay_rates` | `rir.render.low_frequency.modal_damping.material_modal_decay_rates` |

## 只存在於新版的東西

有兩樣東西只存在於新的一側：

- **`puresound.audio.rir.api`** ——一個穩定的 façade，re-export 了大多數
  呼叫端會用到的 37 個名稱。用起來方便，但它會載入整個 renderer stack
  （包括 `torch`）；如果在意匯入成本，請直接匯入某一層的模組。
- **`puresound.audio.rir.contracts`** —— `RIRArray`、`RenderContext`、
  `BackendCapabilities`、`validate_rir_metadata`、`resolve_sound_speed`，
  以及 dtype／layout 常數。這些把程式碼原本就在遵守的慣例正式定型
  （formalize）下來。`resolve_sound_speed` 特別解決了一個真實存在的
  歧義：causality 邊界用的是 **scene environment** 的聲速，而不是
  `HybridRIRConfig.sound_speed`；這兩者差距足以讓邊界移動一個 sample。

## Layering（分層）

依賴關係只會朝下：

```text
api  ->  render, calibration, bank  ->  path_events, scene, metrics
     ->  physics, contracts  ->  numpy / scipy
```

`test/test_rir_r0_import_boundaries.py` 會強制這個方向，並確保
`contracts` 不匯入任何專案內的東西。如果你要在 `puresound/audio/rir/`
底下新增模組，要把它放進那個測試 `LAYER_RANK` 裡列出的某一層。

## 這次遷移移除了什麼

- **35 個相容性 shim。** 它們原本的用途，是讓八個遷移階段可以逐一驗證；
  等最後一個階段落地之後，它們就只是純粹的重複——為每個名稱多提供一種
  拼法而已。`test_rir_r0_api_inventory.py` 會在任何一個又出現時直接失敗。
- **底線別名。** `render/hybrid.py` 不再 re-export `_solve_modal_ard`
  之類的東西；外部程式碼原本透過底線取用的十五個私有 helper，現在都在
  各自的新模組裡有了公開名稱。還留著一個跨 package 的私有匯入——M6.5
  validator 用到的 `_paired_t_confidence_interval`——這件事記錄在
  inventory 測試裡。
- **`render/hybrid.py` 的 re-export 介面。** 它原本為了 shim 而 export
  22 個名稱；現在只 export 一個，`generate_hybrid_rir`，這正是一個
  orchestration 模組該擁有的東西。

有兩個 dead-code 候選項是刻意*不*移除的：`PytARDWaveBackend` 與
`scene.sampling._sample_source_in_shell`。兩者都記錄在 inventory 測試的
`KNOWN_UNUSED` 裡；如果這個判斷在任一方向上不再成立——不管是有人開始用
它們了，還是有人把它們刪掉卻沒更新這筆紀錄——這個測試就會失敗。
