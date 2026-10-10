# Audio

English: [index.md](index.md)

`puresound.audio`：合成 pipeline 所用的訊號運算子，以及 `puresound.audio.rir` 下的
RIR 生成堆疊。這些運算子如何組成訓練資料，見 [資料增強](../augmentation/index.zh-TW.md)。

## 基礎音訊

| 頁面 | 內容 |
| --- | --- |
| [I/O](io.zh-TW.md) | `AudioIO`：soundfile 讀寫、載入時重新取樣與 RMS 位準調整、隨機裁切 |
| [DSP](dsp.zh-TW.md) | `apply_linear`、確定性與隨機化的重新取樣、RBJ biquad、`ParametricEQ`、`compressor_gain` |
| [Spectrum](spectrum.zh-TW.md) | STFT/iSTFT 包裝，以及複數 ↔ 幅度／相位轉換 |
| [Volume](volume.zh-TW.md) | RMS、位準正規化、片段增益失真、淡入淡出、分位數截斷 |
| [Noise](noise.zh-TW.md) | 以 SNR/SIR 混入噪音床或說話者；白噪音 |
| [Impulse response](impulse_response.zh-TW.md) | `wav_apply_rir`（截窗、峰值正規化、對齊）、DRR、隨機換能器 IIR、直達聲 smear |
| [Room simulator](room_simulator.zh-TW.md) | 即時 image-source shoebox RIR，依角色擺放聲源 |
| [Augmentation API](augmentation.zh-TW.md) | `AudioEffectAugmentor`：噪音池、RIR 來源、DRR contrast、`apply_rir`、通道與 codec 運算子 |

## RIR 生成

可執行的工具與範例：[RIR generation recipe](../../../egs/rir_generation/README.zh-TW.md)。

| 頁面 | 內容 |
| --- | --- |
| [實作指南](rir_realism_algorithm.zh-TW.md) | 管線階段對應到模組與頁面；policy 字串 |
| [套件分層](rir_package_layout.zh-TW.md) | `puresound.audio.rir` 的分層、import 規則、公開入口 |
| [Scene schema](rir_scene_v2.zh-TW.md) | `RoomSceneV2`：material-first scene、材質目錄、scene 抽樣 |
| [Hybrid renderer](hybrid_rir.zh-TW.md) | 低／高頻段渲染、crossover、因果性與輸出位準 |
| [Multiband FDN](multiband_fdn.zh-TW.md) | 被動 multiband feedback-delay-network 晚場 |
| [早／晚場耦合](rir_late_coupling.zh-TW.md) | 以能量匹配把 path-event 早場接到 FDN 尾段 |
| [移動聲源渲染](dynamic_scene.zh-TW.md) | 靜態房間中依關鍵影格移動的講者：路徑濾波、連續傳播時間、遮擋、沿路徑的晚場 |
| [PathEvent 已知限制](path_event_audit.zh-TW.md) | 共用 PathEvent 架構的已知限制，附量測與可選修正 |
| [Spatial RIR](spatial_rir.zh-TW.md) | 同步的麥克風陣列、FOA 與 BRIR 渲染 |
| [RIR metrics](rir_metrics.zh-TW.md) | 衰減、清晰度、DRR、噪音底、回音密度、IACC 與相干性 |
| [Attribution](rir_attribution.zh-TW.md) | 供 renderer 稽核的精確直達／早期／晚期分解 |
| [RIR bank 格式](rir_bank_v2.zh-TW.md) | **權威** bank 契約：manifest、split、QC、release、量測匯入、升級 |
| [RIR bank loaders](rir_bank.zh-TW.md) | 訓練端 loader（room、release、union）與其 YAML；讀取上列格式 |

## 阻抗、低頻模態與校正

| 頁面 | 內容 |
| --- | --- |
| [阻抗 prior](impedance_priors.zh-TW.md) | 由 flow resistivity 推得的 Miki 多孔層 prior，擬合成單極點被動邊界 |
| [阻抗量測](impedance_measurements.zh-TW.md) | 複數阻抗資料契約、被動 multi-pole／resonant 模型、bilinear 離散化 |
| [阻抗管量測流程](impedance_tube_protocol.zh-TW.md) | 雙麥克風 H12 化簡、頻帶與被動性門檻、原始檔格式、量測程序 |
| [複數阻抗資料來源](complex_impedance_source_audit.zh-TW.md) | 公開複數阻抗資料的接受準則與保留的驗證資料集 |
| [模態阻尼](modal_damping.zh-TW.md) | 由牆面吸音推得的逐模態衰減、analytic probe 耦合、pytARD 阻尼遞迴 |
| [模態驗證](modal_validation.zh-TW.md) | FDTD reference、1-D／3-D 阻抗模態、residue 校正、evidence tiers |
| [量測活動](rir_measurement_campaign.zh-TW.md) | 實測房間資料契約、audit、房間不相交 split、校正 loss 與擬合 |
