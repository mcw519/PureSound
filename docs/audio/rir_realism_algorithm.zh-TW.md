# RIR 實作指南

English: [rir_realism_algorithm.md](rir_realism_algorithm.md)

本頁整理 RIR 管線與程式位置。可直接執行的範例請看
[RIR generation](../../egs/rir_generation/README.md)。

像 `puresound.iso9613_1.minimum_phase_direct.v2` 這類 policy 字串代表可觀察
行為的版本。行為變更時必須更新版本，才能追溯不同 RIR bank 的產生方式。

## 整體流程

```text
RoomSceneV2
  ├─ 低頻 renderer
  ├─ 高頻 renderer
  └─ crossover 與音量校正
       └─ 可選 FOA、陣列或 binaural 輸出
            └─ bank manifest、QC 與 release
```

| 範圍 | Package |
|---|---|
| 契約與陣列格式 | `puresound.audio.rir.contracts` |
| 場景 schema 與抽樣 | `puresound.audio.rir.scene` |
| 傳播與阻抗 | `puresound.audio.rir.physics` |
| 相干反射路徑 | `puresound.audio.rir.path_events` |
| Renderer 與組合 | `puresound.audio.rir.render` |
| 聲學指標 | `puresound.audio.rir.metrics` |
| 量測校正 | `puresound.audio.rir.calibration` |
| 訓練 bank 封裝 | `puresound.audio.rir.bank` |

請直接 import 所需的分層模組。`puresound.audio.rir.api` 只是方便入口，不是
長期相容性邊界。

## 場景模型

`RoomSceneV2` 記錄幾何、表面材料、環境、聲源、接收器與可選物件。RT60 是
由這些條件推導的結果，不是獨立抽樣參數。

主要行為：

- 材料吸收與散射使用 125 Hz 到 8 kHz 的七個倍頻帶。
- 同一表面的 patch 依面積合成有效材料。
- 未指定音速時，由溫度與濕度推導。
- `predicted_octave_rt60_s()` 使用 Sabine 估計。
- v1 場景升級為 v2 時，不會沿用原本的 RT60。

Sabine 估計假設擴散音場，而且只計算邊界表面。它適合當場景描述；當吸收集中
在少數表面或家具影響很大時，不宜直接當成精確 render target。

## 低頻渲染

實作位於 `render/low_frequency/`：

| Backend | 用途 |
|---|---|
| `analytic_modal.py` | 無損參考 |
| `impedance_modal.py` | 實驗性的複數阻抗模態 |
| `pytard.py` | DCT grid 波動求解，可選材料模態阻尼 |

材料模態阻尼依每個 mode 在六個牆面的參與程度計算 decay rate。遞迴使用阻尼後
的模態頻率與 `exp(-gamma * dt)` pole radius，幾何到達時間之前的 samples
會清零。

pytARD 輸出沒有絕對物理音量。Hybrid renderer 透過 crossover 頻帶的能量匹配
對齊高頻結果。阻抗模態目前仍屬研究用途。

## 高頻渲染

### Pyroomacoustics

`render/high_frequency/pyroomacoustics.py` 使用 image source 與 ray tracing。
V2 場景會把完整材料頻譜傳入 pyroomacoustics。

libroom 的 ray tracing 使用 process-global 隨機狀態，因此相同 seed 不保證
逐 byte 相同。

### PathEvents

`render/high_frequency/path_event.py` 建立相干 image-source paths，並對每條
路徑套用：

- 幾何延遲與衰減；
- 被動邊界濾波器；
- 聲源與接收器指向性；
- ISO 9613-1 空氣吸收；
- 可選的物件遮蔽。

Fractional delay 在建構上就是 causal。物件遮蔽只影響 direct/early window，
不會把整段 late field 一起壓低。

### PathEvents + FDN

`render/high_frequency/fdn.py` 保留 PathEvents early field，再加入 deterministic
multiband FDN。目標 decay 來自場景的倍頻帶 RT60 估計，並加入空氣損耗修正。

Early 與 late field 在 direct arrival 之後做 equal-power transition。每個
channel 的 seed 由 FDN seed、scene ID 與 channel 推導。

此 renderer 會準確追隨 target；Sabine target 本身的誤差也會反映在輸出。

## Crossover 與輸出

`render/crossover.py` 使用 causal 四階 Linkwitz–Riley crossover 合併兩個
頻帶，預設為 1 kHz。低頻 RMS 會在指定頻帶對齊高頻，gain 有上下限。

`render/hybrid.py::generate_hybrid_rir()` 依序：

1. 渲染低頻與高頻；
2. 對齊並清除物理到達時間之前的 samples；
3. 套用 crossover 與 tail fade；
4. 套用 calibrated 或 peak-normalized 輸出政策。

需要保留實際音量關係時請用 calibrated；其 peak 可能大於 1。只有當資料契約
明確要求時，才使用 peak-normalized。

## 空間輸出

| 輸出 | 實作 |
|---|---|
| FOA 與房間陣列 | `render/spatial.py` |
| 同步陣列 late field | `render/spatial_late_field.py` |
| FOA 轉 binaural RIR | `render/binaural.py` |

FOA 使用 ACN channel order 與 SN3D normalization。陣列 channels 共用一個
late-field gain，以保留空間比例。Spatial coherence 與 IACC 只適用於同步接收器。

## 聲學指標

`metrics.analyze_rir()` 產生完整報告，各模組也可個別使用：

- Schroeder decay、T20/T30 與 fit quality；
- C50、C80 與 DRR；
- spectral tilt 與倍頻帶分析；
- Lundeby noise-floor estimation；
- echo density 與 mixing time；
- binaural IACC 與 array coherence。

使用 decay estimate 前，要同時確認 fit quality 與可用 dynamic range。

## 實測 RIR 匯入

`bank/measured_ingest.py` 把公開實測 RIR 轉成 bank 契約。許多 corpus 以 direct
sound 當時間零點，PureSound 則以 emission time 為零點。匯入時會補回幾何傳播
延遲；若 corpus 未提供環境資料，會記錄採用的假設。

若對齊會暴露更早的獨立 arrival、丟棄過多能量，或需要不合理的時間位移，該筆
資料會被拒絕。

CLI：

```bash
python egs/rir_generation/phases/m6_bank/scripts/ingest_measured_m6_variant.py --help
```

## 校正與 bank

量測與 inverse calibration 契約位於 `puresound.audio.rir.calibration`，詳見
[量測 campaign](rir_measurement_campaign.zh-TW.md)。

Bank manifest、split index、QC、release recipe 與 production evidence 位於
`puresound.audio.rir.bank`，詳見 [RIR bank v2](rir_bank_v2.zh-TW.md)。
