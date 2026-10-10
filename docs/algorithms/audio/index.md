# Audio

Traditional Chinese: [index.zh-TW.md](index.zh-TW.md)

`puresound.audio`: the signal operators the synthesis pipeline is built from,
and the RIR generation stack under `puresound.audio.rir`. How these operators
are combined into training rows is in [data augmentation](../augmentation/index.md).

## Core audio

| Page | Covers |
| --- | --- |
| [I/O](io.md) | `AudioIO`: soundfile read/write, resampling and RMS levelling on load, random crops |
| [DSP](dsp.md) | `apply_linear`, deterministic and randomised resampling, RBJ biquads, `ParametricEQ`, `compressor_gain` |
| [Spectrum](spectrum.md) | STFT/iSTFT wrappers and complex ↔ magnitude/phase conversion |
| [Volume](volume.md) | RMS, level normalisation, segment gain distortion, fades, quantile clipping |
| [Noise](noise.md) | Mixing a noise bed or a talker at an SNR/SIR; white noise |
| [Impulse response](impulse_response.md) | `wav_apply_rir` (window, peak normalisation, alignment), DRR, random transducer IIR, direct-arrival smear |
| [Room simulator](room_simulator.md) | On-the-fly image-source shoebox RIRs with role-based source placement |
| [Augmentation API](augmentation.md) | `AudioEffectAugmentor`: noise pools, RIR sources, DRR contrast, `apply_rir`, channel and codec operators |

## RIR generation

Runnable tools and examples: [RIR generation recipe](../../../egs/rir_generation/README.md).

| Page | Covers |
| --- | --- |
| [Implementation guide](rir_realism_algorithm.md) | Pipeline stages mapped to modules and pages; policy strings |
| [Package layout](rir_package_layout.md) | Layers of `puresound.audio.rir`, import rules, public entry points |
| [Scene schema](rir_scene_v2.md) | `RoomSceneV2`: material-first scene, material catalog, scene sampling |
| [Hybrid renderer](hybrid_rir.md) | Low/high-band rendering, crossover, causality and output level |
| [Multiband FDN](multiband_fdn.md) | Passive multiband feedback-delay-network late field |
| [Early/late coupling](rir_late_coupling.md) | Joining the path-event early field to the FDN tail at matched energy |
| [Moving-source rendering](dynamic_scene.md) | Keyframed talkers in a static room: path filters, continuous travel time, occlusion, late field along the path |
| [PathEvent limitations](path_event_audit.md) | Known limitations of the shared PathEvent stack, with measurements and opt-in corrections |
| [Spatial RIR](spatial_rir.md) | Synchronised microphone-array, FOA and BRIR rendering |
| [RIR metrics](rir_metrics.md) | Decay, clarity, DRR, noise floor, echo density, IACC and coherence |
| [Attribution](rir_attribution.md) | Exact direct/early/late decomposition for renderer audits |
| [RIR bank format](rir_bank_v2.md) | **Authoritative** bank contract: manifest, splits, QC, release, measured ingest, promotion |
| [RIR bank loaders](rir_bank.md) | Training-side loaders (room, release, union) and their YAML; reads the format above |

## Impedance, low-frequency modes and calibration

| Page | Covers |
| --- | --- |
| [Impedance priors](impedance_priors.md) | Miki porous-layer priors from flow resistivity, fitted to a passive one-pole boundary |
| [Impedance measurements](impedance_measurements.md) | Complex-impedance data contract, passive multi-pole/resonant models, bilinear discretisation |
| [Impedance-tube protocol](impedance_tube_protocol.md) | Two-microphone H12 reduction, band and passivity gates, raw formats, measurement procedure |
| [Complex-impedance sources](complex_impedance_source_audit.md) | Acceptance criteria for public complex-impedance data and the retained validation dataset |
| [Modal damping](modal_damping.md) | Per-mode decay from wall absorption, analytic-probe coupling, the damped pytARD recurrence |
| [Modal validation](modal_validation.md) | FDTD reference, 1-D/3-D impedance modes, residue calibration, evidence tiers |
| [Measurement campaign](rir_measurement_campaign.md) | Measured-room data contract, audit, room-disjoint splits, calibration loss and fitting |
