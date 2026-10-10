# 資料增強

English: [index.md](index.md)

一筆訓練資料如何合成：每個階段對訊號做什麼、它的參數與 YAML key、各階段的執行
順序，以及讓結果可重現的規則。運算子本身（簽章與算式）記載於
[Audio](../audio/index.zh-TW.md)。

## 頁面

| 頁面 | 內容 |
| --- | --- |
| [房間聲學](room_acoustics.zh-TW.md) | LTI 房間模型、`wav_apply_rir` 對訓練對的處理（截窗、峰值正規化、延遲對齊）、DRR，以及 RIR 來源（bank、模擬器、資料夾） |
| [距離線索](distance_cues.zh-TW.md) | 哪些距離線索在合成後仍保留，以及操縱它們的 knob：DRR contrast、direct-arrival smear、`distance_level` 混音 |
| [位準與動態](level_dynamics.zh-TW.md) | 位準度量、SNR/SIR 混音、峰值保護、增益與削波失真、淡入淡出、壓縮器 |
| [頻譜與通道效果](spectral_channel.zh-TW.md) | Biquad、隨機換能器 IIR、重取樣 backend、speed 與 pitch、media coloring、`apply_linear` |
| [裝置鏈](device_chain.zh-TW.md) | 收音與傳輸鏈：重取樣、換能器響應、HPF、增益與削波、壓縮、A/D 增益調度、codec、封包遺失 |
| [場景合成](scene_construction.zh-TW.md) | Row 類型（target-absent、real-far、real-near、session）、干擾者、overlap gating 與 turn taking、SIR 混音與 `mix_mode`、回音、開頭環境段、三個噪音來源 |
| [工程契約](engineering_contract.zh-TW.md) | RNG 契約、合成順序、設定與程式對應、分佈斷點、測試、新增 knob |

## 每筆資料的順序

`NoiseSuppressionDataset.__getitem__`（`puresound/task/ns.py`）的一筆資料依序執行
下列步驟；`VoiceIsolationDataset`（`puresound/task/voice_isolation.py`）透過 hook
換上自己的 row 類型與混音方式。含順序限制的權威步驟表在
[工程契約 §2](engineering_contract.zh-TW.md)。

1. 載入前景（重新取樣到 `dataset.target_sample_rate`，設定了
   `dataset.gain_normalized_to` 時調整 RMS 位準），裁到該筆長度。
2. 決定 row 類型。
3. 給前景通道：source-level RIR，混音用 `full`、目標用目標窗。
4. 抽干擾者、為媒體聲源染色、各自給 RIR（DRR contrast 與 direct smear 在取 RIR
   時作用）。
5. 控制干擾者與前景的重疊（Bernoulli 或 turn taking）。
6. 以 SIR 混合前景與干擾者。
7. target-absent 的 row 減去前景；加入殘餘播放回音。
8. 套用配對的峰值保護，再做 speed perturbation。
9. 沒有 source-level 殘響的 row 套用整段混音的 RIR。
10. 在開頭環境段把所有語音靜音。
11. 以 SNR 加入錄製噪音、白噪音，再加收音底噪。
12. 把目標存成 VAD 參考。
13. 執行裝置鏈：類比群組、A/D 轉換、codec 與封包遺失。
14. 裁到該筆長度、計算 VAD 標籤，輸出樣本與其來源資訊。

## 名詞

- **Foreground（前景）：** 模型要保留的近場說話者。
- **Interferer（干擾者）：** 模型要壓制的其他說話者。
- **Target（目標）：** 評分輸出時的訓練參考。
- **Mixture（混音）：** 模型輸入。
- **SNR：** 語音對噪音的能量比，以整段 clip 計。
- **SIR：** 前景對干擾者的能量比，以整段 clip 計。
- **dBFS：** 相對於數位滿刻度（1.0）的位準。
