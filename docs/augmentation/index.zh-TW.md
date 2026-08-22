# Data Augmentation DSP 手冊

English version: [`index.md`](index.md)

本手冊整理 augmentation pipeline 中使用的訊號處理手法，是一份跨模組的教學
文件。每個手法涵蓋四件事：它對訊號做了什麼、背後的數學模型與假設、參數的
物理意義，以及它在 pipeline 中的工程約束。

本手冊與其他文件的分工：

* `docs/audio/`、`docs/task/` 是逐模組的 API reference，回答「這個函式怎麼
  呼叫」；本手冊回答「這個手法為什麼這樣設計」。
* `docs/audio/rir_*.md` 系列是 RIR 生成的深入文件；本手冊第 2 章提供其
  導讀地圖。
* 實驗結論與量測數字不收錄於本手冊，統一存放在
  `egs/voice_isolate/benchmarks/`。手法的設計動機若出自量測，文中僅給出處。

## 章節結構

每一章分成兩節：

* **演算法面**：訊號模型與公式推導、手法操縱的物理或感知線索、參數的物理
  意義與合理範圍、模型成立的假設與失效條件。
* **工程面**：實作位置（模組與函式名）、config 對照、RNG 與決定性行為、
  與其他 stage 的順序約束、已知陷阱。

| 章 | 檔案 | 內容 |
|---|---|---|
| 2 | [房間聲學（RIR 軸）](room_acoustics.zh-TW.md) | RIR 的物理結構、LTI 卷積模型、DRR 與臨界距離、影像源法、場景幾何取樣、hybrid 生成文件地圖 |
| 3 | [距離與時間線索](distance_cues.zh-TW.md) | 各距離線索的物理推導、哪些線索被管線移除、DRR contrast、direct smear、distance_level 混音 |
| 4 | [位準與動態](level_dynamics.zh-TW.md) | 位準度量、SNR/SIR 混音推導、增益失真、削波、fade、動態範圍壓縮器 |
| 5 | [頻譜與通道濾波](spectral_channel.zh-TW.md) | biquad 與 RBJ 設計公式、隨機二階 IIR、重取樣理論、變速、media 上色、`apply_linear` |
| 6 | [裝置鏈](device_chain.zh-TW.md) | 類比組與傳輸組的分界、target 跟隨規則、A/D 邊界、codec、packet loss |
| 7 | [場景構成](scene_construction.zh-TW.md) | Row 類型、overlap gating、turn-taking、混音模式、回聲殘餘、噪音三源 |
| 8 | [工程契約](engineering_contract.zh-TW.md) | RNG 決定性契約、合成順序總表、config 對照、分佈版本與可比性、測試防線 |

## 訊號流總覽

一列訓練資料由 `NoiseSuppressionDataset.__getitem__` 合成（voice isolation
任務經 row-type hooks 擴充）。下圖列出從乾語音到模型輸入的每一站，以及
各站所屬章節：

```
乾語音載入（RMS rescale 到 audio_gain_normalized_to）        [ch8 工程契約]
  │
  ├─ 裁切/對齊 align_audio_list（隨機 offset、避開全靜音窗）
  ├─ Row 計畫 _plan_row（target_absent / realnear / realfar）  [ch7 場景構成]
  │
  ├─ 前景通道：source-level RIR（full→混音、early→目標）       [ch2 房間聲學]
  ├─ 干擾者：取樣 → media 上色 → 各自的遠場 RIR 通道          [ch7 / ch2]
  │    └─ DRR contrast、direct smear 在 RIR 進 cache 前套用    [ch3 距離線索]
  ├─ Overlap gating：Bernoulli 或 turn-taking                  [ch7 場景構成]
  ├─ 前景×干擾混音：hard SIR 或 mix_mode                       [ch7 / ch3]
  ├─ target-absent 減除、echo playback（ERLE）                 [ch7 場景構成]
  ├─ avoid_audio_clipping（成對 peak 縮放）
  ├─ Speed perturbation（成對）                                 [ch5 頻譜與通道]
  ├─ Whole-mix folder RIR（非 source-level 列）                 [ch2 房間聲學]
  │
  ├─ Noise stage：錄音噪音(SNR/房間化) → 白噪 → 絕對底噪       [ch7 場景構成]
  ├─ VAD 參照快照（在失真鏈之前）                               [ch7 / ch8]
  │
  └─ Device chain：SRC → 2nd IIR → HPF → volume/clipping        [ch6 裝置鏈]
       → compressor → A/D 邊界 → codec → packet loss            [ch4 / ch5 / ch6]
```

## 記號約定

* 訊號以離散時間 `x[n]` 表示，取樣率 `fs`（訓練管線預設 16 kHz）；連續時間
  推導時用 `x(t)`。
* dB 一律註明基準：能量比 `10·log10(E1/E0)`、振幅比 `20·log10(a1/a0)`、
  絕對位準 dBFS（數位滿刻度 = 1.0，只在 A/D 邊界之後有意義，見 ch6）。
* `U(a, b)` 為均勻分佈、`N(mu, sigma^2)` 為常態分佈。
* SNR 指語音對非語音噪音的能量比，SIR 指前景語者對干擾語者，兩者的混音
  數學相同（ch4）。
* 「前景」= 要保留的近場目標語者；「干擾者」= 要抑制的其他語者；「目標」
  （target）= 模型的訓練參考訊號；「混音」（mixture）= 模型的輸入。
