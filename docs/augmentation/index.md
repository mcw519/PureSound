# Data augmentation

Traditional Chinese: [index.zh-TW.md](index.zh-TW.md)

This section explains how PureSound builds training mixtures. API signatures
remain in the module reference; these pages cover signal behavior, ordering,
configuration, and reproducibility.

## Topics

| Document | Covers |
| --- | --- |
| [Room acoustics](room_acoustics.md) | RIR structure, convolution, DRR, and room simulation |
| [Distance cues](distance_cues.md) | Level, DRR, timing, and direct-arrival changes |
| [Level and dynamics](level_dynamics.md) | SNR/SIR, gain, clipping, fades, and compression |
| [Spectral and channel effects](spectral_channel.md) | Filters, resampling, speed, and channel coloring |
| [Device chain](device_chain.md) | Analog effects, A/D boundary, codecs, and packet loss |
| [Scene construction](scene_construction.md) | Row types, overlap, turn-taking, mixing, echo, and noise |
| [Engineering contract](engineering_contract.md) | Stage order, RNG behavior, config mapping, and tests |

## Pipeline order

A training row follows this order:

1. Load, normalize, crop, and align clean sources.
2. Choose the row type.
3. Apply source-level room responses.
4. Mix foreground and interfering speakers.
5. Apply whole-mixture room response when configured.
6. Add recorded noise, white noise, and the absolute noise floor.
7. Capture VAD labels.
8. Apply the device chain: resampling, filters, gain, clipping, compression,
   codec, and packet loss.

The exact implementation is in `NoiseSuppressionDataset.__getitem__`.
Voice isolation adds task-specific row planning and labels.

## Terms

- **Foreground:** the talker to preserve.
- **Interferer:** another talker to suppress.
- **Target:** the training reference.
- **Mixture:** the model input.
- **SNR:** speech-to-noise energy ratio.
- **SIR:** foreground-to-interferer energy ratio.
- **dBFS:** digital level relative to full scale.
