# 資料增強

English: [index.md](index.md)

本節說明 PureSound 如何合成訓練混音。函式簽章放在 API reference；這裡只整理
訊號行為、處理順序、設定方式與可重現性。

## 主題

| 文件 | 內容 |
| --- | --- |
| [房間聲學](room_acoustics.zh-TW.md) | RIR、convolution、DRR 與房間模擬 |
| [距離線索](distance_cues.zh-TW.md) | 位準、DRR、時間與 direct arrival |
| [位準與動態](level_dynamics.zh-TW.md) | SNR/SIR、gain、clipping、fade 與 compressor |
| [頻譜與通道效果](spectral_channel.zh-TW.md) | Filters、重採樣、變速與 channel coloring |
| [裝置鏈](device_chain.zh-TW.md) | 類比效果、A/D、codec 與 packet loss |
| [場景合成](scene_construction.zh-TW.md) | Row 類型、overlap、turn-taking、混音、echo 與 noise |
| [工程契約](engineering_contract.zh-TW.md) | Stage 順序、RNG、config 對照與測試 |

## 處理順序

每筆訓練資料依序經過：

1. 載入、正規化、裁切並對齊乾淨音源。
2. 決定 row 類型。
3. 套用各音源的 room response。
4. 混合 foreground 與 interferers。
5. 視設定套用整體混音 RIR。
6. 加入錄音噪音、白噪音與 absolute noise floor。
7. 產生 VAD labels。
8. 套用 device chain：重採樣、filter、gain、clipping、compression、codec、
   packet loss。

實際順序以 `NoiseSuppressionDataset.__getitem__` 為準。Voice isolation 另外加入
task-specific row planning 與 labels。

## 名詞

- **Foreground：**要保留的語者。
- **Interferer：**要抑制的其他語者。
- **Target：**訓練 reference。
- **Mixture：**模型輸入。
- **SNR：**語音對噪音的能量比。
- **SIR：**foreground 對 interferer 的能量比。
- **dBFS：**相對於 digital full scale 的位準。
