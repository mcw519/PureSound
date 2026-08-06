# puresound.audio

繁體中文版本：[`index.zh-TW.md`](index.zh-TW.md)

Audio processing sub-package providing primitives for I/O, digital signal processing, spectrum analysis, augmentation, and room simulation.

## Sub-modules

| Module | Description |
|--------|-------------|
| [RIR generation README（繁體中文）](../../egs/rir_generation/README.zh-TW.md) | M0–M6 roadmap、最接近訓練資料的模擬設定，以及 M6 bank 使用方式 |
| [RIR realism: algorithm-to-code reference](rir_realism_algorithm.md) | Physical principles, M2 impedance, M3 PathEvent, M4 spatial late field, M5 inversion, through to M6 bank release/evaluation |
| [rir.calibration (M5 measurement campaign)](rir_measurement_campaign.md) | Repeated ESS contract, group identifiability, fail-closed measured runner, synchronized spatial calibration, constrained residual, and the dual exit |
| [M6 Production RIR Bank v2](rir_bank_v2.md) | Deterministic split, QC, variant recipes, evaluation, and the immutable production certificate / BLOCKED promotion gate |
| [audio.io](io.md) | Audio file reading and writing |
| [audio.dsp](dsp.md) | Resampling and biquad/parametric EQ filtering |
| [audio.spectrum](spectrum.md) | STFT analysis and synthesis utilities |
| [audio.volume](volume.md) | Amplitude normalization and volume manipulation |
| [audio.noise](noise.md) | Background noise mixing |
| [audio.augmentation](augmentation.md) | Composable audio augmentation pipeline |
| [audio.impulse_response](impulse_response.md) | RIR convolution and IIR filtering |
| [audio.impedance_priors](impedance_priors.md) | Versioned Miki/glass-wool references, evidence tiers, fitting, and FDTD modal validation |
| [rir.physics.impedance (complex impedance measurement)](impedance_measurements.md) | SI/normalized contract, real liners, passive fitting, FDTD, and 1D/separable-3D complex modes |
| [Normal-incidence impedance tube protocol](impedance_tube_protocol.md) | Repeated H12, mic-swap calibration, valid band, coherence, uncertainty, and the ingest CLI |
| [Complex impedance public-source audit](complex_impedance_source_audit.md) | Acceptance/rejection record for complex-phase sources, sample condition, and licensing |
| [rir.metrics](rir_metrics.md) | Broadband and octave-band RIR acoustic metrics |
| [rir.metrics.attribution](rir_attribution.md) | Exactly reconstructive direct-anchor/early/later decomposition with complementary raised-cosine masks |
| [rir.render.multiband_fdn](multiband_fdn.md) | Deterministic internally contractive M4.3 multiband FDN core and validation contract |
| [rir.render.coupling](rir_late_coupling.md) | Opt-in M4.4 causal PathEvent-early/FDN-late transition and energy contract |
| [rir.render.spatial (M4 spatial RIR and BRIR)](spatial_rir.md) | Synchronized receiver array, ACN/SN3D FOA, directivity, HRTF FIR decoder, and the evidence boundary |
| [rir.scene](rir_scene_v2.md) | Versioned material-first scene, material catalog, environment, and transducer calibration |
| [rir.path_events](rir_realism_algorithm.md#42-path_eventpy--m3-coherent-pathevents) | Versioned ordered paths, exact folded geometry, furniture visibility, transmission, diffraction, controlled scattering, fractional delay, and causal boundary rendering |
| [rir.render.low_frequency modal damping](modal_damping.md) | Experimental M2 per-mode low-frequency material loss and exact damped recurrence |
| [audio low-frequency modal validation](modal_validation.md) | Independent 3D FDTD reference, experimental wall-face pressure schemes, discrete oblique-boundary phase, and modal frequency/Q metrics |
| [rir.bank](rir_bank.md) | Pre-generated room bank and M6.4 release-recipe serving (split, variant, origin provenance) |
| [audio.room_simulator](room_simulator.md) | Physics-based shoebox room simulator |
| [rir.render（hybrid）](hybrid_rir.md) | Hybrid wave/geometric 5-channel RIR dataset generation |
