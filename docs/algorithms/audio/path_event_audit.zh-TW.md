# PathEvent 架構的已知限制

English: [path_event_audit.md](path_event_audit.md)

靜態渲染器與[移動聲源渲染器](dynamic_scene.zh-TW.md)共同依賴的 RIR 程式有以下限制。
每項都附量測、影響範圍，以及（若有的話）可選的修正。共用預設維持未修正的行為：
靜態渲染器與場景取樣會餵給訓練用的 RIR bank，改預設就會改變合成分佈。要不要把
修正用在訓練上，是另一個需要重新訓練比較的決定。

## 1. 三階 Lagrange 延遲的高頻誤差隨小數部分變化

`path_events/renderer.py`，預設
`render_path_events(fractional_delay="lagrange", fractional_delay_order=3)`。
前向 kernel 保持嚴格的因果支撐（`floor(delay · fs)` 之前全為零），代價是振幅會
隨每條路徑延遲的小數部分改變：

| | 4 kHz | 6 kHz | 7 kHz |
| --- | --- | --- | --- |
| 各小數部分間的起伏 | −0.1 … +0.7 dB | −3.2 … +1.3 dB | −8.4 … +1.4 dB |

靜態 RIR 的每條路徑都帶有各自近似隨機的高頻誤差，6 kHz 以上可達數 dB。

- **影響範圍：** 所有以 PathEvent 為基礎的 RIR（path-event 與 FDN 高頻 backend、
  空間渲染器）。
- **已提供：** `fractional_delay="windowed_sinc"`（32 taps Kaiser sinc，7 kHz 以內
  ±0.01 dB；代價是約 1 ms 的預振鈴，不再是嚴格支撐，因此依賴「抵達 bin 之前精確
  為零」的 bank QC 規則必須容忍預振鈴）。

## 2. `speech_cardioid` 是理想 cardioid

`path_events/directivity.py` 與 `render/high_frequency/pyroomacoustics.py`
（`CardioidFamily(p=0.5)`）。這個指向性不分頻率，講者正後方是零點（各頻率都是
−∞ dB）。實測人聲（Monson 等 2012）背後在 125 Hz 是 −3.6 dB、1 kHz −6 dB、
4–8 kHz −19 到 −26 dB。

- **影響範圍：** 訓練的場景取樣使用它（`scene/sampling.py`），且講者朝向（yaw）
  在 ±180° 內均勻取樣，所以會取到背對麥克風的講者。
- **已提供：** `directivity_id="speech_human"`（實測倍頻帶表，以最小相位頻帶濾波
  實現）。僅 PathEvent backend 支援；Pyroomacoustics backend 會拒絕。

## 3. 離散繞射在陰影邊界處跳變

`path_events/interactions.py`（`include_scene_interactions=True`）：

1. 只有直達線段被擋住時才產生繞射路徑。在陰影邊界上，直達路徑（增益 1）消失、
   邊緣路徑以 `0.5 · sqrt(1 − α − τ) ≤ 0.5` 出現：至少 6 dB 的跳變，移動聲源聽得
   出來。
2. 只考慮垂直邊；矮物件（沙發、桌上隔板）的上緣永遠不會成為繞射路徑。
3. 邊緣係數 `0.5 / sqrt(1 + v²)` 只在一個參考頻率（1 kHz）計算，繞射不隨頻率
   改變。
4. 穿過物件的反射路徑被隱藏，而不是衰減。

- **已提供：** `occlusion_model="fresnel_kirchhoff"`——對每條路徑的每一段套用連續、
  依頻率變化的 Fresnel–Kirchhoff 屏風模型，包含上緣與側緣（與
  `include_scene_interactions` 互斥）。

## 4. 渲染器只能為牆面實現依頻率變化的增益

`render_path_events` 只能透過邊界導納濾波接受依頻率變化的增益頻譜
（否則 `ComplexPathGainSpectrum.constant_real_value` 會丟出例外）。這就是第 2 項
與第 3.3 項不分頻率的原因。

- **已提供：** `PathEvent.band_gain`（`PathBandGain`），逐頻率的實數振幅，以最小
  相位 FIR 實現（`path_events/band_filter.py`）；預設不存在，既有事件的渲染結果
  不變。

## 5. 低反射階數時耦合目標會消失

`render/coupling.py`，`extrapolated_path_tail_energy_target`。晚場能量由 PathEvent
尾段最後一個有效窗外推而來。`max_order=0` 時轉換點之後沒有尾段，FDN 增益為 0
（沒有晚場）；一階時，估計只靠少數幾條反射。目標來自太少路徑時，耦合不會發出警告。

- **影響範圍：** 以低階路徑場呼叫 `couple_path_event_rir_with_fdn` 的地方。
- **已處理：** 移動聲源渲染器一律由二階路徑場推導晚場（`LATE_FIELD_ORDER`）。
