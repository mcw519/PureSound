# Audio

English: [index.md](index.md)

`puresound.audio` 提供音訊 I/O、DSP、資料增強與房間聲學工具。

## 基礎音訊

| 文件 | 內容 |
| --- | --- |
| [I/O](io.zh-TW.md) | 音訊讀寫 |
| [DSP](dsp.zh-TW.md) | 重採樣、biquad 與 parametric EQ |
| [Spectrum](spectrum.zh-TW.md) | STFT 轉換 |
| [Volume](volume.zh-TW.md) | 位準量測與增益 |
| [Noise](noise.zh-TW.md) | 噪音混合 |
| [Augmentation API](augmentation.zh-TW.md) | `AudioEffectAugmentor` |
| [Impulse response](impulse_response.zh-TW.md) | RIR convolution 與 filter |
| [Room simulator](room_simulator.zh-TW.md) | Shoebox room 模擬 |

## RIR 生成

第一次使用請先看 [RIR 生成指南](../../egs/rir_generation/README.zh-TW.md)。

| 文件 | 內容 |
| --- | --- |
| [Hybrid renderer](hybrid_rir.zh-TW.md) | 低頻與高頻 backend 組合 |
| [Scene schema](rir_scene_v2.zh-TW.md) | 房間、材質、聲源與接收器 |
| [RIR metrics](rir_metrics.zh-TW.md) | 聲學量測 |
| [Attribution](rir_attribution.zh-TW.md) | Direct、early、late 分解 |
| [RIR bank](rir_bank.zh-TW.md) | 載入已生成的 room bank |
| [Production bank](rir_bank_v2.zh-TW.md) | 生成、QC、切分與 release bundle |
| [Spatial RIR](spatial_rir.zh-TW.md) | Arrays、FOA、BRIR 與 HRTF |
| [Multiband FDN](multiband_fdn.zh-TW.md) | Late-field reverberation |
| [Early/late coupling](rir_late_coupling.zh-TW.md) | Path event 與 FDN 的銜接 |
| [演算法對照](rir_realism_algorithm.zh-TW.md) | RIR 概念與程式模組對應 |

## 阻抗與低頻驗證

| 文件 | 內容 |
| --- | --- |
| [Impedance priors](impedance_priors.zh-TW.md) | 多孔材質參考值與 fitting |
| [Impedance measurements](impedance_measurements.zh-TW.md) | 資料格式、被動 fitting、FDTD 與 modes |
| [阻抗管流程](impedance_tube_protocol.zh-TW.md) | 量測與匯入 |
| [資料來源審核](complex_impedance_source_audit.zh-TW.md) | 公開資料接受標準 |
| [Modal damping](modal_damping.zh-TW.md) | 材質推導的低頻 damping |
| [Modal validation](modal_validation.zh-TW.md) | FDTD 與 modal 驗證 |
| [量測流程](rir_measurement_campaign.zh-TW.md) | Measured-RIR calibration |
