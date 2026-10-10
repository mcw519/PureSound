# RIR 實作指南

English version: [rir_realism_algorithm.md](rir_realism_algorithm.md)

本頁把 RIR 生成管線對應到實作它的模組，並連到說明每個階段的頁面。指令見
[RIR generation 指南](../../../egs/rir_generation/README.zh-TW.md)；套件分層與
import 規則見 [RIR 套件佈局](rir_package_layout.zh-TW.md)。

## 管線

```text
RoomSceneV2（幾何、材質、環境、transducer）
  ├─ 低頻 backend（波動／模態）                         ─┐
  ├─ 高頻 backend（image source／PathEvent [+ FDN]）     ─┤
  └─ 因果截斷、能量匹配、LR4 crossover、輸出位準 ◄────────┘
       ├─ 可選：receiver 陣列、FOA、BRIR（空間 renderer）
       └─ WAV + JSON item → bank manifest → QC → release → 訓練 loader
```

| 階段 | 模組 | 頁面 |
|---|---|---|
| 資料契約、`HybridRIRConfig`、陣列排版 | `puresound.audio.rir.contracts` | [hybrid RIR](hybrid_rir.zh-TW.md) |
| Scene schema、材質目錄、scene 抽樣 | `puresound.audio.rir.scene` | [scene schema](rir_scene_v2.zh-TW.md) |
| 空氣吸收、阻抗、波動參考解 | `puresound.audio.rir.physics` | 見下；[阻抗先驗](impedance_priors.zh-TW.md)、[模態驗證](modal_validation.zh-TW.md) |
| 相干反射路徑 | `puresound.audio.rir.path_events` | 見下 |
| Backend、crossover、耦合、空間組裝 | `puresound.audio.rir.render` | [hybrid RIR](hybrid_rir.zh-TW.md)、[晚場耦合](rir_late_coupling.zh-TW.md)、[FDN](multiband_fdn.zh-TW.md)、[空間](spatial_rir.zh-TW.md) |
| 聲學指標與歸因 | `puresound.audio.rir.metrics` | [metrics](rir_metrics.zh-TW.md)、[attribution](rir_attribution.zh-TW.md) |
| 量測房間校準 | `puresound.audio.rir.calibration` | [量測 campaign](rir_measurement_campaign.zh-TW.md) |
| Bank 格式、QC、release、loader | `puresound.audio.rir.bank` | [bank 格式](rir_bank_v2.zh-TW.md)、[loader](rir_bank.zh-TW.md) |

訓練只 import `bank.loader`；其他部分都在建 bank 時離線執行。

## Scene

`RoomSceneV2` 存的是成因，RT60 由它們導出：七個 octave 的材質頻譜
（125 Hz – 8 kHz）、面積加權的 patch、由溫度與濕度推得的聲速，以及逐 octave 的
Sabine 預測。Sabine 估計假設擴散聲場且只計入邊界；吸收集中在單一表面或家具主導
時最不可靠。見 [scene schema](rir_scene_v2.zh-TW.md)。

## 物理

- **空氣吸收**（`physics/propagation.py`）。依溫度、濕度與氣壓算出 ISO 9613-1
  衰減（dB/m）。`apply_air_absorption` 以直達距離的因果最小相位 FIR
  濾波一個 channel（policy `puresound.iso9613_1.minimum_phase_direct.v2`）。
  `num_taps`（預設 129）決定線性相位原型的長度；其最小相位的一半長度為
  `(num_taps + 1) // 2` tap，預設為 65。
  `air_adjusted_rt60_s` 把材質衰減與沿著以 `c` 增長之路徑的損失合併：
  `60 / (60 / RT60 + a(f) · c)`。
- **阻抗**（`physics/impedance/`）。被動導納模型、保留相位的先驗、阻抗管換算、
  量測匯入與擬合、矩形房間阻抗模態。見 [阻抗先驗](impedance_priors.zh-TW.md)、
  [阻抗量測](impedance_measurements.zh-TW.md) 與
  [阻抗管規程](impedance_tube_protocol.zh-TW.md)。
- **波動參考解**（`physics/wave/`）。用來驗證模態 backend 的小型 3-D FDTD
  求解器、低頻共振指標，以及兩個頻段共用的自由場 `1/r` source convention。見
  [模態驗證](modal_validation.zh-TW.md)。

## Path event

`PathEvent` 是一次物理到達——距離、fractional delay、依序經過的表面、複數增益
頻譜——在波形組裝前都保持可檢視；延遲與增益分開存放，傳播相位不會被計算兩次。

| 步驟 | 模組 |
|---|---|
| 有序的 shoebox image-source 格點、依入射角的複數邊界增益 | `path_events/generator.py` |
| 對垂直柱體物件的可見性 | `path_events/geometry.py`（`puresound.closed_vertical_prism_segment_visibility.v1`） |
| 可選的連續遮擋：每個物件對每段路徑視為 Fresnel–Kirchhoff 屏風（`occlusion_model="fresnel_kirchhoff"`） | `path_events/occlusion.py`（`puresound.fresnel_kirchhoff_screen.v1`） |
| 物件穿透、有界的刀緣繞射、擴散散射切分 `(1 − s)`／`s / N` | `path_events/interactions.py` |
| 一階聲源／receiver 指向性；實測、依頻率變化的講者指向性 `speech_human` 以頻帶增益表示 | `path_events/directivity.py` |
| 頻帶增益（`PathEvent.band_gain`）以最小相位 FIR 實現 | `path_events/band_filter.py` |
| 以因果 forward-Lagrange fractional delay（`floor(delay · fs)` 之前嚴格為零）與被動數位邊界濾波渲染；可選窗 sinc 延遲（`path_events/fractional_delay.py`） | `path_events/renderer.py`（`puresound.causal_forward_lagrange.v1`） |

邊界濾波來自 `material_absorption_relaxation_models`：依每個材質低頻與高頻吸收
擬合的被動、因果單極點導納。只靠吸收係數無法決定反射相位，所以這是宣告過的相位
先驗，不是量測所得的阻抗。這個架構的已知限制與可選修正列在
[PathEvent 架構的已知限制](path_event_audit.zh-TW.md)；移動聲源由
[`render/dynamic.py`](dynamic_scene.zh-TW.md) 渲染。

## 低頻 backend

`render/low_frequency/`：

| Backend | 模型 |
|---|---|
| `pytard.py`（`GpuARDPytARDBackend`、CuPy 版） | pytARD adaptive rectangular decomposition 求解，以內部取樣率（預設 16 kHz）與 Green-delta 聲源計算後重採樣到輸出取樣率。其原始振幅會被 peak-normalize 並重新套上 `1/r`（`calibrate_pytard_signal`），所以由 crossover 能量匹配決定它的位準。無損求解結果會從直達到達起套一個寬頻 RT60 包絡，或在啟用材質阻尼時改為逐模態衰減 |
| `analytic_modal.py` | 具互易聲源／receiver 耦合的矩形房間模態；測試用的快速替代 |
| `impedance_modal.py` | 針對被動有理牆面導納的可分離複數阻抗模態，residue 以 FDTD 校準 |
| `modal_damping.py` | 由邊界參與度推得的逐模態振幅衰減率，`γ = s · c · Σ_axes (α₋ + α₊) / N_axis / 8`，模態索引為 0 時 `N_axis = L`、其餘為 `L/2`，上限為 0.95 ω；遞迴使用阻尼頻率 `√(ω² − γ²)` 與極點半徑 `exp(−γ Δt)` |

材質阻尼律尚未經量測驗證，是需明確選用的模型；見 [modal damping](modal_damping.zh-TW.md)。

## 高頻 backend

| Backend | 模型 |
|---|---|
| `pyroomacoustics.py` | Image source 加 ray tracing，使用每個邊界的等效吸收與散射頻譜；移除固定的濾波延遲；家具事後以 `obstacle_occlusion_recovery_ms` 內的衰減斜坡與一個散射 tap 套用（`obstacles.py`）。只有 `set_rng_seed` 後才可重現 |
| `path_event.py` | 階數到 12 的相干 PathEvent，含交互作用、指向性與空氣吸收；物件影響個別路徑，而不是整個 channel |
| `fdn.py` | PathEvent 早場耦合 multiband FDN 晚場（[晚場耦合](rir_late_coupling.zh-TW.md)） |

## 組裝

`render/crossover.py` 在物理到達處截斷各頻段、對齊 Pyroomacoustics 的直達路徑、
在 crossover 附近把低頻段匹配到高頻段，並套用因果四階 Linkwitz–Riley crossover；
`render/hybrid.py::generate_hybrid_rir` 負責串接並套用尾端淡出與輸出位準
（`calibrated` 或 `peak_normalized`）。見 [hybrid RIR](hybrid_rir.zh-TW.md)。同步陣列、
FOA 與 BRIR 在 `render/spatial.py`、`render/spatial_late_field.py` 與
`render/binaural.py`（[空間](spatial_rir.zh-TW.md)）。

## Policy 字串

像 `puresound.iso9613_1.minimum_phase_direct.v2` 這樣的字串會蓋在 item metadata 中，
命名一種可觀察的行為。改變某階段的計算內容就必須換新字串，讓 bank 記錄產生它的
是哪一種行為。

| Policy | 位置 |
|---|---|
| `rir_scene.v2`、`puresound-materials.v1` | Scene schema 與材質目錄 |
| `puresound.iso9613_1.minimum_phase_direct.v2` | 空氣吸收 |
| `puresound.causal_forward_lagrange.v1` | PathEvent fractional delay |
| `puresound.fresnel_kirchhoff_screen.v1` | 可選的連續物件遮擋 |
| `puresound.dynamic_geometric_fdn.v2` | 移動聲源渲染器 |
| `puresound.pytard.green_delta.v1` | pytARD 激發 |
| `puresound.multiband_fdn.v2`、`puresound.path_event_fdn_coupling.v1` | FDN 與耦合 |
| `puresound.spatial_room_rir.v1`、`puresound.spatial_late_field.v1`、`puresound.ambisonic_binaural_decoder.v1` | 空間 renderer |
| `puresound.rir_bank.v2`、`puresound.m6_split.sha256_acoustic_space.v1`、`puresound.rir_bank_qc.physical.v1`、`puresound.rir_bank_release.v1` | Bank 格式 |
| `puresound.measured_time_origin.iso3382_onset_to_geometric_arrival.v1` | 量測 RIR 匯入 |
