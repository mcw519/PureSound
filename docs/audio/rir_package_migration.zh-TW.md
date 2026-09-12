# RIR import 遷移

English: [rir_package_migration.md](rir_package_migration.md)

RIR 程式現在統一放在 `puresound.audio.rir`。舊的 flat modules 與 compatibility
shims 已移除。更新舊 script、notebook 或 branch 時可使用此頁。

## Import 對照

| 舊 module | 現行 module |
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

低頻與高頻 backends 分別位於
`puresound.audio.rir.render.low_frequency` 和
`puresound.audio.rir.render.high_frequency`。

## 公開入口

Render hybrid RIR：

```python
from puresound.audio.rir.render.hybrid import generate_hybrid_rir
```

只使用 contracts，不載入 renderer stack：

```python
from puresound.audio.rir.contracts import HybridRIRConfig
```

使用 path events：

```python
from puresound.audio.rir.path_events import PathEvent, render_path_events
```

`puresound.audio.rir.api` 會 re-export 常用名稱，但不是 compatibility boundary。
Library code 應直接從負責該功能的 layer import。

## Private helper 對照

舊 `hybrid_rir` 匯出的 helpers 已改成所屬模組中的 public names：

| 舊 helper | 現行位置 |
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

`test/test_rir/test_rir_import_boundaries.py` 會檢查此依賴方向，並避免 contracts 與
metadata reader 載入不必要的 renderer dependencies。
