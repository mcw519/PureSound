# Audio

Traditional Chinese: [index.zh-TW.md](index.zh-TW.md)

The `puresound.audio` package contains audio I/O, DSP, augmentation, and room
acoustics tools.

## Core audio

| Document | Covers |
| --- | --- |
| [I/O](io.md) | Reading and writing audio |
| [DSP](dsp.md) | Resampling, biquads, and parametric EQ |
| [Spectrum](spectrum.md) | STFT conversion |
| [Volume](volume.md) | Level measurement and gain |
| [Noise](noise.md) | Noise mixing |
| [Augmentation API](augmentation.md) | `AudioEffectAugmentor` |
| [Impulse response](impulse_response.md) | RIR convolution and filtering |
| [Room simulator](room_simulator.md) | Shoebox-room simulation |

## RIR generation

Start with the [RIR generation guide](../../egs/rir_generation/README.md).

| Document | Covers |
| --- | --- |
| [Hybrid renderer](hybrid_rir.md) | Low/high-frequency backend composition |
| [Scene schema](rir_scene_v2.md) | Rooms, materials, sources, and receivers |
| [RIR metrics](rir_metrics.md) | Acoustic measurements |
| [Attribution](rir_attribution.md) | Direct, early, and late decomposition |
| [RIR bank](rir_bank.md) | Loading generated room banks |
| [Production bank](rir_bank_v2.md) | Generation, QC, splits, and release bundles |
| [Spatial RIR](spatial_rir.md) | Arrays, FOA, BRIR, and HRTF decoding |
| [Multiband FDN](multiband_fdn.md) | Late-field reverberation |
| [Early/late coupling](rir_late_coupling.md) | Path-event and FDN transition |
| [Algorithm map](rir_realism_algorithm.md) | RIR concepts mapped to source modules |

## Impedance and low-frequency validation

| Document | Covers |
| --- | --- |
| [Impedance priors](impedance_priors.md) | Porous-layer references and fitting |
| [Impedance measurements](impedance_measurements.md) | Data contract, passive fitting, FDTD, and modes |
| [Impedance-tube protocol](impedance_tube_protocol.md) | Measurement and ingestion |
| [Source audit](complex_impedance_source_audit.md) | Public-source acceptance criteria |
| [Modal damping](modal_damping.md) | Material-derived low-frequency damping |
| [Modal validation](modal_validation.md) | FDTD and modal checks |
| [Measurement campaign](rir_measurement_campaign.md) | Measured-RIR calibration workflow |
