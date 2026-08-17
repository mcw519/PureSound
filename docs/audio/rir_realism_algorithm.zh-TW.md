# RIR 演算法與代碼對應

English version: `rir_realism_algorithm.md`

本文件是 `puresound.audio.rir` 的**演算法 ↔ 代碼**參考：每一段物理與訊號處理，
對應到實作它的模組、函式與 policy 字串。它描述系統「是什麼、在哪裡」；
實驗結果、審查結論與工作計畫在 [`RIR_EXP_LOG.md`](../../RIR_EXP_LOG.md)，
使用方式在 [`egs/rir_generation/README.md`](../../egs/rir_generation/README.md)。

輸出契約中帶版本的 policy 字串（如 `puresound.iso9613_1.minimum_phase_direct.v2`）
是行為的身份證：實作變更會同步變更 policy 版本，讓 bank metadata 能區分
不同行為產生的資料。

---

## 1. 系統概觀

一條合成 RIR 的生成路徑：

```
場景抽樣（材質先行）
  └─ scene/            RoomSceneV2：幾何 + 表面材質 + 環境 + 收發器
低頻帶（20–1000 Hz）        高頻帶（1000 Hz–Nyquist）
  └─ render/low_frequency/    └─ render/high_frequency/
     ARD/DCT 波動解               pyroomacoustics｜PathEvents(M3)｜PathEvents+FDN(M4)
     + per-mode 材質阻尼
        └────────── render/crossover.py ──────────┘
                causal Linkwitz–Riley 4 階 @ 1 kHz
                + RMS 能量匹配 [700, 1300] Hz
  └─ render/spatial.py         （可選）FOA / 同步陣列 / 雙耳
  └─ bank/                     M6 契約：manifest → QC → release → 評估 → 產線決策
```

套件佈局：

| 子套件 | 職責 |
|---|---|
| `puresound/audio/rir/contracts.py` | 共用契約：`HybridRIRConfig`、軸序/資料型別常數、crossover 增益夾限 |
| `puresound/audio/rir/scene/` | 場景 schema、材質目錄、幾何、抽樣 |
| `puresound/audio/rir/physics/` | 傳播（空氣吸收）、阻抗、FDTD 參考解 |
| `puresound/audio/rir/path_events/` | 波路徑事件：生成、幾何、邊界互動、指向性、渲染 |
| `puresound/audio/rir/render/` | 低/高頻 backend、crossover、FDN、空間渲染、orchestration |
| `puresound/audio/rir/metrics/` | 聲學量測：衰減、清晰度、頻譜、echo density、空間 |
| `puresound/audio/rir/calibration/` | M5 量測 campaign 契約與反演校準 |
| `puresound/audio/rir/bank/` | M6 訓練 bank：契約、QC、release、評估、產線決策、實測 ingest |

入口：`puresound.audio.rir.api` 是方便用的集合，不是穩定性邊界（見
`rir_package_migration.zh-TW.md`）——請直接匯入你要的那一層。
`render/hybrid.py` 只有一個公開函式 `generate_hybrid_rir`（orchestration）。
`render/backend.py` 定義 backend protocol 與 `BackendCapabilities`
（含決定性宣告）。

---

## 2. 場景：材質先行（scene/）

核心原則：**場景記錄聲學成因（幾何、材質、環境），聲學結果（RT60）是推導量。**

| 概念 | 代碼 |
|---|---|
| v2 場景 schema | `scene/schema.py` — `RoomSceneV2`：`dimensions_m`、`surfaces`（含 patch）、`materials`、`environment`、`sources`/`receivers`（`TransducerConfig` + `Pose`）、`objects` |
| 材質吸收/散射頻譜 | `MaterialSpectrum`：7 個 octave 中心 `[125…8000]` Hz 的係數；`.at(f)` 內插、`.to_pra_dict()` 轉 pyroomacoustics 格式 |
| 環境 → 聲速 | `EnvironmentConfig`：未給 `sound_speed_m_s` 時用 `331.3 + 0.606·T + 0.0124·RH` |
| 表面 patch 混合 | `RoomSceneV2.effective_boundary_materials()`：同一面牆的 patch 依面積加權混出六面等效材質 |
| Sabine RT60 預測 | `RoomSceneV2.predicted_octave_rt60_s()`：`RT60(f) = (24·ln10 / c) · V / Σᵢ Sᵢ·αᵢ(f)`，夾限 [0.02, 20] s |
| 相容欄位 `rt60` | `RoomSceneV2.rt60` property = 500/1000 Hz 預測的中位數；metadata 標 `rt60_origin: surface_material_sabine_prediction` — **推導量，不是抽樣輸入** |
| 材質目錄與房型 | `scene/materials.py` — `ROOM_TYPE_RECIPES`（教室/客廳/會議室/辦公室）、`sample_materialized_shoebox()`：一個房間 latent 產生相關的牆/地/天花板選材、patch 與家具 |
| v1 → v2 升級 | `scene/sampling.py` — `upgrade_hybrid_scene_to_v2()`：把 v1 幾何物化為材質場景；**v1 的 `rt60` 刻意不複製**（v2 的 rt60 由材質推導） |
| 幾何工具 | `scene/geometry.py`（多邊形/房間幾何）、`schema.shoebox_surface_areas()` |

已知限制（證據見 exp log）：

- Sabine 預測假設擴散場均勻取樣所有表面；當吸收集中在單一面（例如吸音天花板
  + 硬牆）時會系統性低估 RT60。它適合當**房間描述**，
  不適合直接當**渲染目標**（`RIR_EXP_LOG.md` §6.6）。
- Sabine 加總只含邊界表面；`objects`（家具）不在其中。
- 預設路徑（v1→v2 升級）上 `HybridRIRConfig.rt60_range` 不約束產出的
  推導 RT60 分佈（`RIR_EXP_LOG.md` §6.6.5）。

---

## 3. 低頻帶：波動解 + 材質模態阻尼（render/low_frequency/）

| 概念 | 代碼 |
|---|---|
| ARD/DCT 波動解 | `low_frequency/pytard.py` — `GpuARDPytARDBackend`（CPU）/ `GpuARDPytARDCuPyBackend`（GPU）：矩形房的 modal/DCT 格點 `omega[z,y,x]`，時間步進用逐模態 recurrence |
| 激勵 | policy `puresound.pytard.green_delta.v1`：單樣本 delta（`nonzero_sample_count: 1`），solver 帶限 1.2 kHz；取代舊 bipolar FIR（其固定 comb 特徵已除役） |
| 材質模態阻尼 | `low_frequency/modal_damping.py` — `material_modal_decay_rates()`：每個模態 `(nx,ny,nz)` 依其對六面牆的**表面參與度**加權材質吸收，得衰減率 γ（上限 `0.95·ω` 保穩定）；model 字串 `surface_participation_sabine` |
| 精確阻尼 recurrence | `pytard.py`：`damped_omega = √(ω²−γ²)`、`pole_radius = exp(−γ·dt)` 的阻尼餘弦 recurrence，逐模態、無全域包絡（`global_rt60_envelope_applied: false`） |
| 因果性 | policy `zero_samples_before_floor_distance_over_sound_speed`：每 channel 在幾何到達樣本前清零（有限 voxel 解的數值前導） |
| 舊全域 RT60 包絡 | `apply_rt60_decay_envelope()`：只在**非**材質阻尼模式當 fallback（`material_modal_damping=False`） |
| 阻抗模態 backend | `low_frequency/impedance_modal.py` — `ImpedanceModalLowFrequencyBackend` + `RectangularImpedanceBoundaryConfig`：複數阻抗邊界的模態解（M2 實驗線） |
| 解析模態 backend | `low_frequency/analytic_modal.py`：無耗散解析參考 |

注意：`material_modal_damping_metadata(max_modes=128)` 是 **metadata 摘要**——
只記錄最低的 128 個模態；solver 本身對整個 DCT 格點施加阻尼。
讀 metadata 判斷頻率覆蓋會得到錯誤結論（此陷阱記錄於 `RIR_EXP_LOG.md` §6.6.1）。

M2 的表面參與損失律**尚未通過出口 gate**（模組 docstring 與
`RIR_EXP_LOG.md` 皆註明）：它是基礎設施，不是已被接受的產線模型。

---

## 4. 高頻帶：三個 backend（render/high_frequency/）

### 4.1 `pyroomacoustics.py` — 幾何聲學（ISM + ray tracing）

- v2 場景走 `pra.Material(absorption.to_pra_dict(), scattering.to_pra_dict())`：
  **完整頻譜**（7 band）逐面傳入，非單一寬帶係數。
- v1 場景 fallback：`pra.inverse_sabine(scene.rt60, room_dim)` 反推吸收與 ISM 階數。
- 空氣吸收用 pra 內建 `set_air_absorption()`（**不是**本套件的 ISO 9613-1 實作）。
- 決定性警告（docstring 記載）：libroom 的 ray tracing 有 process-global RNG，
  per-task 種子控制不到，**同 seed 不保證 byte 重現**。

### 4.2 `path_event.py` — M3 同調 PathEvents

| 概念 | 代碼 |
|---|---|
| 波路徑事件集 | `path_events/schema.py`（`PathEventSet`）、`path_events/generator.py`（鏡像源展開） |
| 幾何核 | `path_events/geometry.py`：fold/unfold 鏡像距離（互檢至 1e-11）、AABB 粗篩 `apply_scene_object_visibility()`（先便宜的軸對齊包圍盒測試，再精確稜柱測試；不變量：粗篩絕不拒絕真相交） |
| 邊界濾波 | `path_events/renderer.py` — `boundary_filter_for(model, incidence_cosine)`：由材質導納模型（`physics/impedance/admittance.py`，Cayley 變換）構造被動、極點 <1 的反射濾波器；以 `(model, cosine, fs)` 記憶化 |
| 指向性 | `path_events/directivity.py`：`speech_cardioid` 等一階壓力樣式，逐路徑套用 |
| 空氣吸收 | `physics/propagation.py` — `apply_air_absorption()`：**ISO 9613-1** 最小相位濾波，policy `puresound.iso9613_1.minimum_phase_direct.v2`（v2 修正濕度單位：ISO 全程用百分比；v1 的 h 小了 100 倍，見 `RIR_EXP_LOG.md` §4.3b） |
| 因果性 | 單邊 Lagrange 分數延遲核——**構造性**滿足因果，不靠事後清零 |
| 家具遮蔽 | `high_frequency/obstacles.py` — model 字串 `direct_early_occlusion_with_diffuse_recovery`：只衰減 direct 窗並線性恢復，晚場不動（壓 DRR、不壓殘響） |

### 4.3 `fdn.py` — M4 PathEvents 早場 + 多帶 FDN 晚場

| 概念 | 代碼 |
|---|---|
| backend | `PathEventFDNHighFrequencyBackend`（繼承 M3；早場同 4.2） |
| 晚場目標 | `_target_rt60_s_by_hz()`：取 `scene.predicted_octave_rt60_s()` 再做空氣吸收修正（`air_adjusted_rt60_s`，policy `FDN_RT60_adds_sound_speed_times_atmospheric_loss`）；下限 `max(500 Hz, crossover/2)` |
| 多帶 FDN | `render/multiband_fdn.py` — `render_multiband_fdn()`：被動確定性 FDN；filterbank 是 cascaded binary split，**端點完備**（最高帶 highpass 到 Nyquist） |
| 耦合 | `render/coupling.py` — `couple_path_event_rir_with_fdn()`，policy `PATH_EVENT_FDN_COUPLING_POLICY`：在 direct 後 `mixing_time_s`（預設 24 ms）以 `equal_power_transition_weights()` 交叉淡接，transition 前 sample-exact 保留早場 |
| 晚場能量錨定 | `coupling.py` — `extrapolated_path_tail_energy_target()`：以材質 RT60 衰減律把有限 PathEvent tail 的能量外推積分到 render 邊界，FDN 增益錨在外推值（不鎖在截斷 tail 上）；前提 = 材質衰減律，`rt60_range` 上限 >1.5 s 時外推佔比需重新評估 |
| 種子 | `_channel_seed()`：`blake2b(fdn_seed, scene_id, channel)` — 逐 channel 確定性 |

metadata 中 `opt_in: true / production_default_changed: false` 描述的是
`generate_hybrid_rir` 那一層（M4/M5 出口 gate 錨定該層預設為 pyroomacoustics）；
M6 wrapper `generate_m6_bank.py` 自 2026-08-04 起**預設選用本 backend**，
且每次都顯式傳 `--high-backend`。選型依據（實測衰減形狀對照）見
`RIR_EXP_LOG.md` §6.6.6。

已知限制：FDN 晚場**照目標實現**（實測命中 0.94–1.00×），因此目標的誤差
就是輸出的誤差——Sabine 目標無法回應吸收的空間分佈（`RIR_EXP_LOG.md` §6.6）。

---

## 5. Crossover 與組裝（render/crossover.py、render/hybrid.py）

| 概念 | 代碼 |
|---|---|
| 分頻 | `hybrid_crossover()` / `hybrid_crossover_with_metadata()`：**causal** Linkwitz–Riley 4 階 @ `HybridRIRConfig.crossover_hz`（預設 1000）；因果濾波保住兩帶的前導零 |
| 能量匹配 | `match_low_band_to_high_band()`：在 `effective_crossover_match_band()`（預設 [700, 1300] Hz）做 RMS 匹配，回傳 `(scaled, gain, raw_gain)`；增益夾限 `HybridRIRConfig.crossover_match_gain_range = (1e-4, 8.0)`，metadata 記 `low_band_gain_requested_by_channel` 與 `low_band_gain_clipped_channels` |
| 高帶對齊 | `align_high_band_direct()`；對齊後逐 channel `clip_rir_before_physical_arrival()` 清零 |
| 尾端淡出 | raised-cosine² 20 ms（`tail_fade` metadata） |
| 因果契約 | 全鏈成立：低帶 clip→causal LP4 保零；高帶對齊清零→causal HP4 保零；PathEvents 構造性；FDN 耦合 transition 前 sample-exact（`RIR_EXP_LOG.md` §2.2 有實測） |
| Orchestration | `render/hybrid.py` — `generate_hybrid_rir()`：唯一公開入口，組合 scene→兩帶→crossover→輸出校準 |
| 輸出位準 | `output_mode`：`calibrated`（物理 SPL 語意，peak 可 >1）或 `peak_normalized` |
| 陣列殼形 | `render/arrays.py`：render 層共用的 RIR 陣列形狀（軸序 `contracts.RIR_AXIS_ORDER`） |

已知根因（未修，記錄於 exp log）：pytARD 低帶輸出**逐 item peak 正規化**
（`signal/peak · target_peak · 1/distance`），絕對位準不物理；
RMS 匹配是把低帶拉回高帶位準的機制，夾限 (1e-4, 8.0) 是為此而設。

---

## 6. 空間渲染（render/spatial*.py、binaural.py）

| 概念 | 代碼 |
|---|---|
| FOA / 房間空間 RIR | `render/spatial.py` — `render_room_scene_spatial_rir()`，policy `SPATIAL_ROOM_RIR_POLICY`；一階 Ambisonics 用 **SN3D/ACN** |
| 擴散場等向性 | 位準求解用**單一共享增益**（`one_shared_array_gain_preserves_spatial_ratios`）：逐 channel 求解會破壞 Y/Z/X 對 W 的等向比 `E[Y²]=E[W²]/3`（−4.77 dB）；常駐 validator `egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_foa_diffuse_isotropy.py` |
| 同步陣列晚場 | `render/spatial_late_field.py`（M4.5） |
| 雙耳 | `render/binaural.py`（M4.6，FOA→BRIR，可選） |

---

## 7. 聲學量測（metrics/）

單一入口 `analyze_rir(signal, fs, ...)` 回傳整包指標；個別函式可獨用。

| 指標 | 函式 | 備註 |
|---|---|---|
| Schroeder 衰減 | `schroeder_decay_db` / `noise_compensated_schroeder_decay_db` | 後者用噪音地板補償 |
| T20/T30 | `estimate_decay_time` → `DecayEstimate`（含 fit R²） | 判讀規則：fit R² 門檻先於數值 |
| C50/C80 | `clarity_db` | |
| DRR | `compute_drr_db`、`direct_sample` | direct 窗預設 2.5 ms |
| 頻譜斜率 | `spectral_tilt_db_per_octave` | 對實測參照的效度限制見 `RIR_EXP_LOG.md` §4.4 |
| 噪音地板 | `estimate_noise_floor_lundeby` → `NoiseFloorEstimate` | Lundeby 迭代；需 ≥15 dB 動態範圍 |
| Echo density / mixing time | `analyze_echo_density`、`abel_normalized_echo_density_profile`、`estimate_abel_mixing_time`（policy `ABEL_ECHO_DENSITY_POLICY`） | |
| Octave 分帶 | `octave_band_rir`、`valid_octave_centers`、`DEFAULT_OCTAVE_CENTERS_HZ` | |
| 多帶晚場 | `analyze_multiband_late_field`（policy `MULTIBAND_LATE_FIELD_POLICY`） | M4.2 契約 |
| 空間 | `analyze_binaural_iacc`（`IACC_POLICY`）、`analyze_array_spatial_coherence`、`diffuse_field_coherence`（`DIFFUSE_FIELD_COHERENCE_POLICY`） | 只對同步接收陣列有意義 |

低頻專用：`physics/wave/low_frequency.py`（模態峰值/Q 估計）、
`physics/wave/fdtd.py`（獨立 3D FDTD 參考解，供模態驗證交叉比對）。

---

## 8. 實測 RIR ingest（bank/measured_ingest.py）

把公開語料的實測 RIR 整備成能通過 M6 QC 的 `measured` variant。
核心問題：M6 定義 `t=0` 為聲源發聲時刻，公開語料以自身直達音為原點——
ingest 把語料移除的傳播延遲放回去，房間響應本身不動。

| 概念 | 代碼 |
|---|---|
| 對位 policy | `MEASURED_TIME_ORIGIN_POLICY = puresound.measured_time_origin.iso3382_onset_to_geometric_arrival.v1` |
| Onset 判準 | ISO 3382-1：**從頭往前**找第一次越過 peak−20 dB（且高於噪音 +20 dB）；判準以 BRUDEX 為校正標準選定（該語料自帶傳播延遲，正確判準必須對它要求零平移） |
| 對位 | `align_measured_channel()`：平移使 onset 落在 `floor(d/c·fs) + fade`，前段靜音，4 樣本 raised-cosine 淡入只碰次門檻樣本 |
| 拒收準則 | `earlier_arrival`（與 onset 有**間隔**的更早到達——位準判不了，pre-onset 本來就有上升緣）、`removed_energy`（>1% 能量）、`implausible_shift`（延遲 >100 ms 或前移 >250 ms） |
| 聲速假設 | 語料不發佈溫濕度 → `ASSUMED_MEASURED_ENVIRONMENT`（20 °C/50 % → 344.04 m/s）寫進每個 item 的 scene 並標 `provenance: assumed`；QC 用同一數字重算到達 |
| 打包 | `build_measured_m6_bank()`：每語料一個 renderer profile（量測鏈即渲染器）、`signal_variant=measured`、`level_policy=native_measured`；被拒 item 記錄於 ingest report，不入 manifest |

CLI：`egs/rir_generation/phases/m6_bank/scripts/ingest_measured_m6_variant.py`。
選型與驗證數據（44–74 dB 高於噪音地板的 pre-arrival、五語料實跑結果）見
`RIR_EXP_LOG.md` §5。

---

## 9. 校準（calibration/）

| 概念 | 代碼 |
|---|---|
| M5 量測 campaign 契約 | `calibration/measured_campaign.py`：受控房間、repeated ESS、房間/收發器/環境/資產全 sha 定址、room-disjoint split（`RIRMeasurementCampaign` 等 dataclass + `audit_measurement_campaign`） |
| 校準執行 | `calibration/measured_runner.py` |
| 多目標損失 | `calibration/loss.py`：多解析度 STFT + direct-relative EDC + pre-arrival 懲罰 |
| M4 反演 | `calibration/inverse_m4.py`：`M4_PROFILE_CONVERGENCE_POLICY = puresound.m4_profile_convergence.stable_minimum.v1` — 收斂定義為「擬合達到穩定極小」（相對容差 1e-3、自 `result.x` 重啟驗證），不是 scipy 的 success flag |
| M5 反演 / 殘差 | `calibration/inverse_m5.py`、`calibration/residual.py`（受約束殘差模型）、`calibration/synthetic_recovery.py`（合成恢復自檢） |
| 阻抗量測 | `physics/impedance/`：`measurements.py`、`tube.py`（阻抗管協定）、`fitting.py`、`priors.py`、`modes.py`、`residues.py` |

---

## 10. M6 訓練 bank（bank/）

六層管線，每層 fail-closed，全鏈 content-addressed（單一
`canonical_json_sha256`：sort_keys + 固定分隔符 + `allow_nan=False`）。

| 層 | 代碼 | 要點 |
|---|---|---|
| 契約 | `bank/schema.py` | `RIRBankManifest`/`RIRBankItem`（WAV+JSON 同名對、逐檔 sha）；`BankSplitPolicy`：以 `sha256(policy_id, seed, acoustic_space_id)` 決定 split——room 與 acoustic-space disjoint、確定性；`BankGeneratorProvenance`（code revision、config sha、task plan sha）；`BankRendererProfile`（evidence tier：`development/empirical_candidate/production_approved`）；libsndfile `PEAK` chunk timestamp 歸零（`canonicalize_float_wav_header`）讓 float WAV byte 可重現 |
| 生成 | `egs/rir_generation/generate_hybrid_rir.py` | task plan 先於執行、逐 item 種子、resume 綁 code revision（換版重做不混血）、`generation_run` 記 `elapsed_seconds`/`items_failed`（worker 例外中止於 manifest 之前，故 failed=0 是設計事實） |
| QC | `bank/qc.py` | `RIRBankQCPolicy`（版本化門檻：pre-arrival ≤1e-7 相對 peak、到達誤差 ≤1 ms、tail 能量、T20 fit R²≥0.7 覆蓋、octave 衰減覆蓋、echo density、有限性/形狀/身份）；`run_rir_bank_qc` 產 pass-only candidate index + quarantine index；`audit_rir_bank_qc_release` **重算**全部 hash 與 membership |
| Release | `bank/release.py` | `build_m6_variant_release`：`synthetic_calibrated`/`synthetic_peak_normalized` variant + `real_native`/`mixed_calibrated_real` recipe（無 measured bank 時誠實 blocked）；lineage 逐樣本驗證（child = parent × common gain）；`prune_bank_to_qc_passed`：release variant 必須全 pass，剪枝副本入 release、未剪枝 bank 留作紀錄 |
| 評估 | `bank/evaluation.py` | `compare_release_distributions`（variant 尺度不變性 + synthetic↔measured normalized Wasserstein）；`validate_throughput_report` / `validate_listening_report`（empirical tier 會**重算** estimate 與 paired-t CI、<20 人不算 empirical）/ `validate_downstream_report`（CI 下界重算）；`evaluate_m6_release` 分 implementation exit 與 empirical exit |
| 產線決策 | `bank/production.py` | 13 項 `PRODUCTION_DECISION_CHECK_NAMES`；`audit_m6_production_evidence`（bundle、三角色簽核、逐 profile 核准，全部對 release/evaluation hash 綁定）；`validate_m6_production_certificate` **重算** decision components，證書不可偽造 |
| 證據產生 | `bank/evidence.py`、`bank/listening.py` | 核准/簽核/bundle/throughput 的**產生器**（全 fail-closed：無具名核准者、無計時、檔案不存在都拒絕）；聽測 assignment 設計（room-disjoint、盲化標籤、hidden reference/degraded anchor）與回應 ingest；乾跑走 `contract_fixture` tier + `explicitly_not_human_responses`，**不產生回應資料** |
| 讀取 | `bank/loader.py`、`bank/storage.py` | `PreGeneratedReleaseBank`：split 必填、`split == usage_role` 交叉檢查、manifest 缺席 fail-closed；訓練側 provenance 傳到每個 sample（`rir_release_sha256` 等） |

注意兩套 `EVIDENCE_TIERS` 字彙不同：renderer profile 用
`(development, empirical_candidate, production_approved)`（`bank/schema.py`），
聽測/評估契約用 `(contract_fixture, empirical)`（`bank/evaluation.py`）。

核准必須在 QC **之前**蓋進 manifest（QC summary 綁 manifest hash），
所以 release 是兩趟流程；驅動 CLI 為
`egs/rir_generation/phases/m6_bank/scripts/build_m6_evidence.py`
（evaluate → approve → attest）。

---

## 11. 不變量與測試對應

| 不變量 | 驗證位置 |
|---|---|
| AABB 粗篩不拒絕真相交 | `test/test_rir_path_events.py`（保守性不變量） |
| 邊界濾波被動、極點 <1 | path_events 測試（Cayley passivity） |
| FOA 擴散等向 ±1 dB | `validate_foa_diffuse_isotropy.py`（含對抗性反例） |
| 因果契約全鏈 | M6 item QC `prearrival_energy` gate + crossover 測試 |
| 同參數 fresh run byte 重現 | `RIR_EXP_LOG.md` §2.1（實測記錄）；`validate_m6_reproducible_generation.py` |
| ISO 9613-1 數值正確 | `test/test_rir_air_absorption.py`：對標準表值 + 獨立轉寫雙重驗證 |
| BRUDEX 零平移（實測對位） | `test/test_rir_measured_ingest.py`（含前向掃描回歸守衛） |
| 聽測統計可重算 | `test/test_rir_m6_evidence.py`（producer 產物過 validator 的重算） |
| API 表面凍結 | `test/test_rir_r0_api_inventory.py`（`__all__`、跨套件私有 import、死碼宣告） |
| 匯入分層 | `test/test_rir_r0_import_boundaries.py` |
| M6 憑證不可偽造 | `test/test_m6_production_decision_validator.py` |

---

## 12. 已知限制與未結案（指向 exp log）

| 主題 | 位置 |
|---|---|
| Sabine 當 FDN 晚場目標的效度 | `RIR_EXP_LOG.md` §6.6 |
| 低頻帶偏快（0.62–0.79×，兩臂共用） | §6.6.7 |
| pyroomacoustics 高頻衰減過長（~2.2×） | §6.6.6 |
| `rt60_range` 不約束 v1→v2 抽樣 | §6.6.5 |
| pytARD 低帶位準非物理（peak 正規化根因） | §4.2 |
| 實測參照對 tilt/噪音地板不可用 | §4.4、§4.5 |
| M6 產線決策剩餘 blocker（真人聽測、下游訓練） | §6 |
