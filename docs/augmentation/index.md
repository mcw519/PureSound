# Data Augmentation DSP Handbook

繁體中文版本：[`index.zh-TW.md`](index.zh-TW.md)

This handbook documents the signal processing techniques used in the
augmentation pipeline. It is a cross-module teaching document: for each
technique it covers what it does to the signal, the mathematical model and
assumptions behind it, the physical meaning of its parameters, and the
engineering constraints it operates under inside the pipeline.

How this handbook relates to the rest of the documentation:

* `docs/audio/` and `docs/task/` are per-module API references answering "how
  do I call this function"; this handbook answers "why is this technique built
  this way".
* The `docs/audio/rir_*.md` series covers RIR generation in depth; chapter 2
  here provides a reading map for it.
* Experimental conclusions and measurement numbers are **not** recorded here.
  They live in `egs/voice_isolate/benchmarks/`. Where a technique's design
  motivation came from a measurement, the text cites the source rather than
  restating the numbers.

## Chapter structure

Every chapter has two sections:

* **Algorithm** — signal model and derivations, the physical or perceptual cue
  the technique manipulates, the physical meaning and reasonable range of each
  parameter, and the assumptions the model rests on together with the
  conditions under which they fail.
* **Engineering** — where it is implemented (module and function names), config
  mapping, RNG and determinism behaviour, ordering constraints relative to
  other stages, and known pitfalls.

| Ch | File | Contents |
|---|---|---|
| 2 | [Room acoustics (the RIR axis)](room_acoustics.md) | Physical structure of an RIR, the LTI convolution model, DRR and critical distance, the image source method, scene geometry sampling, hybrid generation reading map |
| 3 | [Distance and timing cues](distance_cues.md) | Derivation of each distance cue, which cues the pipeline removes, DRR contrast, direct smear, `distance_level` mixing |
| 4 | [Level and dynamics](level_dynamics.md) | Level metrics, SNR/SIR mixing derivation, gain distortion, clipping, fades, the dynamic range compressor |
| 5 | [Spectral and channel filtering](spectral_channel.md) | Biquads and the RBJ design equations, random 2nd-order IIR, resampling theory, speed perturbation, media coloring, `apply_linear` |
| 6 | [Device chain](device_chain.md) | The analogue/transmission split, target-follows rules, the A/D boundary, codecs, packet loss |
| 7 | [Scene construction](scene_construction.md) | Row types, overlap gating, turn-taking, mixing modes, residual echo, the three noise sources |
| 8 | [Engineering contracts](engineering_contract.md) | RNG determinism contracts, the synthesis order table, config mapping, distribution versioning and comparability, the test net |

## Signal flow overview

One training row is synthesised by `NoiseSuppressionDataset.__getitem__` (the
voice isolation task extends it through row-type hooks). The diagram below
lists every station from dry speech to model input, with the chapter that
covers it:

```
Load clean speech (RMS rescale to audio_gain_normalized_to)   [ch8 contracts]
  │
  ├─ Crop/align via align_audio_list (random offset, avoid all-silent window)
  ├─ Row plan _plan_row (target_absent / realnear / realfar)   [ch7 scene]
  │
  ├─ Foreground channel: source-level RIR (full→mix, early→target)  [ch2 room]
  ├─ Interferers: sample → media coloring → own far-field RIR   [ch7 / ch2]
  │    └─ DRR contrast, direct smear applied before the RIR is cached [ch3]
  ├─ Overlap gating: Bernoulli or turn-taking                   [ch7 scene]
  ├─ Foreground × interferer mixing: hard SIR or mix_mode       [ch7 / ch3]
  ├─ target-absent subtraction, echo playback (ERLE)            [ch7 scene]
  ├─ avoid_audio_clipping (paired peak rescale)
  ├─ Speed perturbation (paired)                                [ch5 spectral]
  ├─ Whole-mix folder RIR (non source-level rows)               [ch2 room]
  │
  ├─ Noise stage: recorded noise (SNR / room-colored) → white → floor [ch7]
  ├─ VAD reference snapshot (before the distortion chain)        [ch7 / ch8]
  │
  └─ Device chain: SRC → 2nd IIR → HPF → volume/clipping         [ch6 device]
       → compressor → A/D boundary → codec → packet loss         [ch4 / ch5 / ch6]
```

## Notation

* Discrete-time signals are written `x[n]` with sample rate `fs` (16 kHz by
  default in the training pipeline); continuous-time derivations use `x(t)`.
* Every dB figure states its reference: energy ratios as `10·log10(E1/E0)`,
  amplitude ratios as `20·log10(a1/a0)`, absolute level as dBFS (digital full
  scale = 1.0, meaningful only after the A/D boundary — see ch6).
* `U(a, b)` is the uniform distribution, `N(mu, sigma^2)` the normal.
* SNR is the energy ratio of speech to non-speech noise, SIR that of the
  foreground talker to interfering talkers. The mixing mathematics is the same
  for both (ch4).
* "Foreground" is the near-field target talker to preserve; "interferer" is any
  other talker to suppress; "target" is the model's training reference signal;
  "mixture" is the model input.
