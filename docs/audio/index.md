# puresound.audio

Audio processing sub-package providing primitives for I/O, digital signal processing, spectrum analysis, augmentation, and room simulation.

## Sub-modules

| Module | Description |
|--------|-------------|
| [RIR generation README（繁體中文）](../../egs/rir_generation/README.zh-TW.md) | M0–M6 roadmap、最接近訓練資料的模擬設定，以及 M6 bank 使用方式 |
| [RIR 擬真算法（繁體中文）](rir_realism_algorithm_zh-TW.md) | 從物理原理、M2 阻抗、M3 PathEvent、M4 空間尾場、M5 反演，到 M6 bank release/evaluation 的完整中文說明 |
| [M5 受控房間 RIR 量測與反演（繁體中文）](rir_measurement_campaign_zh-TW.md) | repeated ESS 契約、group identifiability、fail-closed measured runner、同步 spatial calibration、受限 residual 與雙 exit |
| [M6 Production RIR Bank v2（繁體中文）](rir_bank_v2_zh-TW.md) | deterministic split、QC、variant recipes、evaluation，以及 immutable production certificate／BLOCKED promotion |
| [audio.io](io.md) | Audio file reading and writing |
| [audio.dsp](dsp.md) | Resampling and biquad/parametric EQ filtering |
| [audio.spectrum](spectrum.md) | STFT analysis and synthesis utilities |
| [audio.volume](volume.md) | Amplitude normalization and volume manipulation |
| [audio.noise](noise.md) | Background noise mixing |
| [audio.augmentation](augmentation.md) | Composable audio augmentation pipeline |
| [audio.impulse_response](impulse_response.md) | RIR convolution and IIR filtering |
| `audio.acoustic_impedance` | Phase-aware complex impedance plus passive causal relaxation and resonant RLC boundaries |
| [audio.impedance_priors](impedance_priors.md) | Versioned Miki/glass-wool references, evidence tiers, fitting, and FDTD modal validation |
| [複數阻抗量測與共振邊界](impedance_measurements.md) | SI／normalized 契約、真實 liner、被動 fitting、FDTD、1D 與 separable 3D 複數模態 |
| [正入射阻抗管量測流程（繁體中文）](impedance_tube_protocol_zh-TW.md) | repeated H12、換麥校正、有效頻帶、coherence、不確定度與匯入 CLI |
| [複數阻抗公開資料審核](complex_impedance_source_audit.md) | complex phase、樣品條件與授權的接受／拒絕記錄 |
| [audio.rir_metrics](rir_metrics.md) | Broadband and octave-band RIR acoustic metrics |
| [audio.multiband_fdn](multiband_fdn.md) | Deterministic internally contractive M4.3 multiband FDN core and validation contract |
| [audio.rir_late_coupling](rir_late_coupling.md) | Opt-in M4.4 causal PathEvent-early/FDN-late transition and energy contract |
| [M4 空間 RIR 與 BRIR](spatial_rir.md) | 同步 receiver array、ACN/SN3D FOA、directivity、HRTF FIR decoder 與 evidence boundary |
| [audio.rir_scene](rir_scene_v2.md) | Versioned material-first scene, material catalog, environment, and transducer calibration |
| `audio.rir_path_events` | Versioned ordered paths, exact folded geometry, furniture visibility, transmission, diffraction, controlled scattering, fractional delay, and causal boundary rendering |
| `audio.rir_attribution` | Exactly reconstructive direct-anchor/early/later decomposition with complementary raised-cosine masks |
| [audio.hybrid_rir modal damping](modal_damping.md) | Experimental M2 per-mode low-frequency material loss and exact damped recurrence |
| [audio low-frequency modal validation](modal_validation.md) | Independent 3D FDTD reference, experimental wall-face pressure schemes, discrete oblique-boundary phase, and modal frequency/Q metrics |
| [audio.rir_bank](rir_bank.md) | Pre-generated room bank 與 M6.4 release recipe serving（split、variant、origin provenance） |
| [audio.room_simulator](room_simulator.md) | Physics-based shoebox room simulator |
| [audio.hybrid_rir](hybrid_rir.md) | Hybrid wave/geometric 5-channel RIR dataset generation |
