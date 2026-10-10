# Data augmentation

Traditional Chinese: [index.zh-TW.md](index.zh-TW.md)

How a training row is synthesised: what each stage does to the signal, its
parameters and YAML keys, the order the stages run in, and the rules that keep
the result reproducible. The operators themselves (their signatures and
arithmetic) are documented under [Audio](../audio/index.md).

## Pages

| Page | Covers |
| --- | --- |
| [Room acoustics](room_acoustics.md) | The LTI room model, what `wav_apply_rir` does to a training pair (window, peak normalisation, delay alignment), DRR, and where RIRs come from (bank, simulator, folder) |
| [Distance cues](distance_cues.md) | Which distance cues survive synthesis, and the knobs that change them: DRR contrast, direct-arrival smear, `distance_level` mixing |
| [Level and dynamics](level_dynamics.md) | Level measures, SNR/SIR mixing, peak guards, gain and clipping distortion, fades, the compressor |
| [Spectral and channel effects](spectral_channel.md) | Biquads, the random transducer IIR, resampling backends, speed and pitch, media coloring, `apply_linear` |
| [Device chain](device_chain.md) | The capture and transmission chain: resampling, transducer response, HPF, gain and clipping, compressor, A/D gain staging, codec, packet loss |
| [Scene construction](scene_construction.md) | Row types (target-absent, real-far, real-near, session), interferers, overlap gating and turn taking, SIR mixing and `mix_mode`, echo, the ambient lead, the three noise sources |
| [Engineering contract](engineering_contract.md) | RNG contracts, synthesis order, config-to-code mapping, distribution breaks, tests, adding a knob |

## Row order

A row of `NoiseSuppressionDataset.__getitem__` (`puresound/task/ns.py`) runs
these steps; `VoiceIsolationDataset` (`puresound/task/voice_isolation.py`)
substitutes its own row types and mixing through hooks. The authoritative
step table, with the ordering constraints, is in
[Engineering contract §2](engineering_contract.md#2-synthesis-order).

1. Load the foreground (resampled to `dataset.target_sample_rate`, RMS-levelled
   when `dataset.gain_normalized_to` is set) and crop it to the row length.
2. Plan the row type.
3. Give the foreground its channel: source-level RIR, `full` for the mixture
   and the target window for the target.
4. Sample interferers, color media sources, give each its RIR (DRR contrast and
   direct smear act inside the RIR draw).
5. Gate the interferers' overlap with the foreground (Bernoulli or turn taking).
6. Mix foreground and interferers at an SIR.
7. Subtract the foreground on target-absent rows; add residual playback echo.
8. Apply the paired peak guard, then speed perturbation.
9. Apply a whole-mixture RIR on rows without source-level reverb.
10. Silence all speech over the row-initial ambient lead.
11. Add recorded noise at an SNR, white noise, then the capture floor.
12. Snapshot the target as the VAD reference.
13. Run the device chain: analogue group, A/D conversion, codec and packet loss.
14. Crop to the row length, compute VAD labels, emit the sample and its
    provenance.

## Terms

- **Foreground:** the near talker the model keeps.
- **Interferer:** another talker the model suppresses.
- **Target:** the training reference the output is scored against.
- **Mixture:** the model input.
- **SNR:** speech-to-noise energy ratio, over the whole clip.
- **SIR:** foreground-to-interferer energy ratio, over the whole clip.
- **dBFS:** digital level relative to full scale (1.0).
