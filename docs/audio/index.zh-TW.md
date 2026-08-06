# puresound.audio

English version: [`index.md`](index.md)

Audio 處理子套件，提供 I/O、數位訊號處理（DSP）、頻譜分析、增強
（augmentation）與房間模擬的基礎元件。

## 子模組

| 模組 | 說明 |
|--------|-------------|
| [RIR generation README（繁體中文）](../../egs/rir_generation/README.zh-TW.md) | M0–M6 roadmap、最接近訓練資料的模擬設定，以及 M6 bank 使用方式 |
| [RIR realism：算法對照代碼的完整參考](rir_realism_algorithm.zh-TW.md) | 從物理原理、M2 阻抗、M3 PathEvent、M4 空間尾場、M5 反演，到 M6 bank release/evaluation 的完整說明 |
| [rir.calibration（M5 量測活動）](rir_measurement_campaign.zh-TW.md) | Repeated ESS 契約、group identifiability、fail-closed measured runner、同步 spatial calibration、受限 residual 與雙 exit |
| [M6 Production RIR Bank v2](rir_bank_v2.zh-TW.md) | Deterministic split、QC、variant recipes、evaluation，以及 immutable production certificate／BLOCKED promotion gate |
| [audio.io](io.zh-TW.md) | 音訊檔案讀寫 |
| [audio.dsp](dsp.zh-TW.md) | Resampling 與 biquad／parametric EQ 濾波 |
| [audio.spectrum](spectrum.zh-TW.md) | STFT 分析與合成工具 |
| [audio.volume](volume.zh-TW.md) | 振幅正規化與音量調整 |
| [audio.noise](noise.zh-TW.md) | 背景噪音混合 |
| [audio.augmentation](augmentation.zh-TW.md) | 可組合式音訊增強 pipeline |
| [audio.impulse_response](impulse_response.zh-TW.md) | RIR 卷積與 IIR 濾波 |
| [audio.impedance_priors](impedance_priors.zh-TW.md) | 版本化的 Miki／玻璃棉參考值、evidence tiers、fitting 與 FDTD modal 驗證 |
| [rir.physics.impedance（複數阻抗量測）](impedance_measurements.zh-TW.md) | SI／normalized 契約、真實 liner、被動 fitting、FDTD、1D 與 separable 3D 複數模態 |
| [正入射阻抗管量測流程](impedance_tube_protocol.zh-TW.md) | repeated H12、換麥校正、有效頻帶、coherence、不確定度與匯入 CLI |
| [複數阻抗公開資料審核](complex_impedance_source_audit.zh-TW.md) | complex phase、樣品條件與授權的接受／拒絕記錄 |
| [rir.metrics](rir_metrics.zh-TW.md) | Broadband 與 octave-band 的 RIR 聲學指標 |
| [rir.metrics.attribution](rir_attribution.zh-TW.md) | 精確可重建的 direct-anchor／early／later 分解，搭配互補的 raised-cosine 遮罩 |
| [rir.render.multiband_fdn](multiband_fdn.zh-TW.md) | Deterministic、內部 contractive 的 M4.3 multiband FDN core 與驗證契約 |
| [rir.render.coupling](rir_late_coupling.zh-TW.md) | Opt-in 的 M4.4 causal PathEvent-early／FDN-late 過渡與能量契約 |
| [rir.render.spatial（M4 空間 RIR 與 BRIR）](spatial_rir.zh-TW.md) | 同步 receiver array、ACN/SN3D FOA、directivity、HRTF FIR decoder 與 evidence boundary |
| [rir.scene](rir_scene_v2.zh-TW.md) | 版本化的 material-first scene、material catalog、environment 與 transducer calibration |
| [rir.path_events](rir_realism_algorithm.zh-TW.md#42-path_eventpy--m3-coherent-pathevents) | 版本化的有序路徑、精確 folded geometry、家具可見性、transmission、diffraction、可控 scattering、fractional delay 與 causal boundary rendering |
| [rir.render.low_frequency modal damping](modal_damping.zh-TW.md) | 實驗性的 M2 per-mode 低頻材質損耗與精確 damped recurrence |
| [audio low-frequency modal validation](modal_validation.zh-TW.md) | 獨立的 3D FDTD 參考、實驗性 wall-face pressure schemes、離散 oblique-boundary phase 與 modal 頻率／Q 指標 |
| [rir.bank](rir_bank.zh-TW.md) | Pre-generated room bank 與 M6.4 release-recipe serving（split、variant、origin provenance） |
| [audio.room_simulator](room_simulator.zh-TW.md) | 基於物理的 shoebox 房間模擬器 |
| [rir.render（hybrid）](hybrid_rir.zh-TW.md) | Hybrid wave／geometric 五聲道 RIR 資料集生成 |
