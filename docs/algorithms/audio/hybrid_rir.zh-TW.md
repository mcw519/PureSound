# Hybrid RIR 渲染 — `puresound.audio.rir.render.hybrid`

English version: [hybrid_rir.md](hybrid_rir.md)

`generate_hybrid_rir` 為每個聲源渲染一支房間脈衝響應：低頻段用波動／模態
backend 求解，高頻段用幾何 backend，最後在一個因果（causal）crossover
把兩段接起來。各 backend 在套件中的位置見[實作指南](rir_realism_algorithm.zh-TW.md)。

## 為什麼分兩段

低頻時房間表現為一組離散模態；幾何聲學（image source、ray）無法表示它們，
波動求解器可以。高頻時模態密度夠高，幾何聲學已經準確，而波動求解器太貴：
網格必須解析波長，成本大約隨最高頻率的三次方成長。Hybrid renderer 只在
每種方法有效且負擔得起的頻段使用它。

## 入口

```python
from puresound.audio.rir.render.hybrid import generate_hybrid_rir

rir, metadata = generate_hybrid_rir(
    config,               # HybridRIRConfig
    scene=None,           # HybridRIRScene | RoomSceneV2；None 時依 config 抽樣
    low_backend=None,     # 預設 GpuARDPytARDBackend()（pytARD，CPU）
    high_backend=None,    # 預設 PyroomacousticsHighFrequencyBackend()
    seed=None,            # scene 為 None 時抽樣 scene 用的 seed
)
# rir：torch.float32 [num_sources, num_samples]；metadata：JSON-safe dict
```

任何具備 `simulate(scene, config) -> np.ndarray [num_sources, samples]` 的物件
都滿足 `RIRBackend` protocol（`render/backend.py`）。`RoomSceneV2` 必須剛好有
`config.num_sources` 個聲源與一個 receiver；receiver array 走
[空間 renderer](spatial_rir.zh-TW.md)。

## 設定

`HybridRIRConfig`（`puresound.audio.rir.contracts`）同時放 scene 抽樣範圍與
渲染設定。渲染相關欄位：

| 欄位 | 預設 | 意義 |
|---|---|---|
| `sample_rate` | 48000 | 輸出取樣率，Hz |
| `duration` | 1.5 | 輸出長度，s |
| `low_fmin_hz`、`low_fmax_hz` | 20、1000 | 低頻 backend 負責的頻段，Hz |
| `crossover_hz` | 1000 | Linkwitz–Riley crossover 頻率，Hz |
| `match_crossover_energy` | `True` | 在 crossover 附近把低頻段縮放到高頻段的位準 |
| `crossover_match_band_hz` | `None` | 能量匹配頻段；`None` 表示 `[0.7, 1.3] × crossover_hz` |
| `crossover_match_target_db` | 0.0 | 匹配頻段內低／高頻 RMS 的目標比值，dB |
| `crossover_match_gain_range` | (1e-4, 8.0) | 低頻匹配增益的上下限 |
| `preserve_source_convention_at_crossover` | `True` | 低頻段已與高頻段共用 source convention 時跳過匹配 |
| `tail_fade_ms` | 20 | 響應尾端的淡出長度，ms |
| `output_mode` | `peak_normalized` | `peak_normalized` 或 `calibrated` |
| `normalize_peak` | 0.98 | `peak_normalized` 的峰值目標 |
| `calibrated_reference_source_spl_db` | 94 | transducer 增益的參考聲源位準，dB SPL |
| `record_realized_metrics` | `False` | 逐 channel 儲存 `analyze_rir` 結果 |
| `num_near_sources`、`num_far_sources` | 2、3 | channel 數為兩者之和 |

`RoomSceneV2` 的聲速取自 scene 的 environment，而不是
`HybridRIRConfig.sound_speed`。

## 演算法

1. **低頻段。** 低頻 backend 渲染每個聲源 channel。`floor(d / c · fs)` 之前的
   sample 設為零（`clip_rir_before_physical_arrival`）：有限的模態或 voxel
   求解會留下一點數值前驅，那不是物理路徑。
2. **高頻段。** 高頻 backend 渲染每個 channel。Pyroomacoustics 的輸出會先移除
   它固定的 fractional-delay 偏移，讓直達聲落在 `d / c`
   （`align_high_band_direct`），再以同樣方式截掉到達前的部分。PathEvent
   backend 本身就是因果的。
3. **Transducer 增益**（僅 `RoomSceneV2`）。兩個頻段逐 channel 乘上
   `10^((L_src − L_ref + G_rx) / 20)`，其中 `L_src` 是聲源的
   `power_db_spl_at_1m`，`L_ref` 是 `calibrated_reference_source_spl_db`，
   `G_rx` 是 receiver 的 `calibration_gain_db`。
4. **能量匹配。** 兩個頻段在匹配頻段內做帶通（二階 Butterworth），低頻段乘上
   `g = clip(RMS_high · 10^(target_dB / 20) / RMS_low, gain_range)`。pytARD
   低頻段經其自身校準後是 peak-normalized 的，所以這一步才是它相對高頻段的
   位準來源；增益撞到上下限的 channel 會列在 metadata。當低頻 backend 回報
   `direct_path_source_convention_matched` 且
   `preserve_source_convention_at_crossover` 開著時跳過匹配。Metadata 同時記錄
   「是否要求匹配」與「是否真的匹配」。
5. **Crossover。** 四階 Linkwitz–Riley（二階 Butterworth 套兩次）以因果方式
   （`sosfilt`）分別濾兩個頻段後相加。低通與高通共用同一相位響應，所以獨立
   渲染的兩段在時間上保持對齊、相加後幅度平坦；因果濾波避免直達聲前出現
   pre-ringing。
6. **尾端淡出。** 最後 `tail_fade_ms` 乘上一個結束於零的 raised-cosine 平方窗。
7. **輸出位準。** `peak_normalized` 把整個 item 縮放到
   `max |h| = normalize_peak`。`calibrated` 不做正規化：振幅承載聲源位準、
   距離、receiver 校準與跨房間位準，峰值可能超過 1.0。

## Backend

| CLI 名稱（`generate_hybrid_rir.py`） | 類別 | 說明 |
|---|---|---|
| `pytard` | `GpuARDPytARDBackend` | CPU 上的 pytARD DCT 網格波動求解；寬頻 RT60 包絡 |
| `pytard-material` | 同上，`material_modal_damping=True` | 由表面材質導出的逐模態阻尼（[modal damping](modal_damping.zh-TW.md)）；需要 material-first scene |
| `pytard-cupy`、`pytard-cupy-material` | `GpuARDPytARDCuPyBackend` | 同一求解器透過 CuPy 在 GPU 上執行 |
| `analytic`、`analytic-material` | `AnalyticModalLowFrequencyBackend` | 矩形房間模態；供測試與 smoke run 的快速替代 |
| `analytic-impedance` | `ImpedanceModalLowFrequencyBackend` | 複數阻抗模態；需要明確的邊界 JSON（[阻抗量測](impedance_measurements.zh-TW.md)） |
| `pyroomacoustics` | `PyroomacousticsHighFrequencyBackend` | Image source 到 `max_order`（12）加 ray tracing（20000 條 ray） |
| `path-events-m3` | `PathEventHighFrequencyBackend` | 相干 image-source PathEvent，含邊界濾波、指向性、ISO 9613-1 空氣吸收與物件遮擋 |
| `path-events-m4` | `PathEventFDNHighFrequencyBackend` | PathEvent 早場加 multiband FDN 晚場（[晚場耦合](rir_late_coupling.zh-TW.md)、[FDN](multiband_fdn.zh-TW.md)） |

pytARD 是選用的 AGPL 相依套件，不隨 PureSound 散佈；請用
`PURESOUND_PYTARD_ROOT` 指向一份 checkout。PathEvent backend 需要 `RoomSceneV2`。

## 因果性契約

每個階段都讓 channel 在 `floor(d / c · fs)` 之前嚴格為零，並保留到達那個
sample 本身：低頻段在因果低通之前截斷，高頻段在直達對齊之後截斷，PathEvent
使用單邊 fractional-delay kernel，FDN 耦合不動 transition 之前的任何 sample。
`RIRArray.violates_causality` 從使用端檢查同一條邊界，bank QC 以
`prearrival_energy` 讓 item 失敗。`resolve_sound_speed(metadata)` 回傳一個已存
item 定義這條邊界所用的聲速（先取 scene environment，config 為備援）。

## 決定性

PathEvent 與 FDN backend 在輸入固定時是決定性的。Pyroomacoustics 的 ray
tracing 會從 libroom 的 process 全域亂數產生器與套件內的 NumPy 產生器取數；
只有在渲染前用 `PyroomacousticsHighFrequencyBackend.set_rng_seed` 同時固定兩者，
輸出才能逐位元組重現。`generate_hybrid_rir.py` 在寫 bank manifest 時會逐 task
這麼做。`BackendCapabilities`（`contracts.py`）是 backend 宣告「固定 seed 下
是否可重現」的地方。

## Metadata

回傳的 dict 包含 `config`、`scene`（含 `channel_map`：每個 channel 的 label、
位置與距離）、`obstacle_effects`、`bands.low` 與 `bands.high`（backend、頻段、
邊界模型、激發、晚場、空氣吸收）、`output_calibration`、`crossover`，以及要求時
的 `realized_acoustics`。資料寫出端再加上 `sample_id`、`room_id`、`room_index`、
`rir_index` 與檔名；`validate_rir_metadata` 檢查必要欄位
（`RIR_METADATA_REQUIRED_KEYS`）與 channel map。

## 工具

| 工具 | 預設高頻 backend | 用途 |
|---|---|---|
| `egs/rir_generation/generate_hybrid_rir.py` | `pyroomacoustics` | 低階產生器：每個 item 一組 WAV/JSON，可選擇輸出 bank manifest（`--emit-m6-manifest`） |
| `egs/rir_generation/generate_m6_bank.py` | `path-events-m4` | 一個指令完成生成、QC 與 release 打包（[bank 格式](rir_bank_v2.zh-TW.md)） |

低階預設維持 `pyroomacoustics`，讓既有指令產生相同的資料；bank wrapper 一律
明確傳入 backend。指令見 [RIR generation 指南](../../../egs/rir_generation/README.zh-TW.md)。

生成的 item 是 32-bit float WAV 加一個同名 JSON sidecar。Sidecar 是資料的一部分
（channel map、距離、level policy），不可與 WAV 分開。訓練端透過
[bank loader](rir_bank.zh-TW.md) 讀取 item，而不是直接卷積檔案。
