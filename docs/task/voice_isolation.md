# puresound.task.voice_isolation

Near-field foreground voice isolation dataset: keep the speaker within ~1 m,
suppress everything further away. Specializes the
[NoiseSuppressionDataset](ns.md) synthesis skeleton through its row-type hooks —
there is exactly one synthesis path; the specialization lives in overrides.

## Class: `VoiceIsolationDataset`

Adds, each behind its own config knob (absent/disabled blocks never consume
RNG, so plain noise-suppression recipes regenerate bit-identically):

| block | row type |
|---|---|
| `augmentation_realfar` | far interferers are **finished loudspeaker→air→mic recordings** from a pool manifest, inserted with no RIR applied (a convolved far channel only carries the LTI part of a capture chain). `prob` = share of items; `lone_far_prob` = share with no near speaker (target = silence, the absolute "lone far voice = suppress" example); `turn_taking_prob` = row-level rate. |
| `augmentation_realnear` | foreground is a **genuine close-mic recording** (target = itself), interferers preferentially from the same room and never the same speaker — real speech on the KEEP side of the same mixtures whose far side is real, so the boundary stays on proximity cues instead of capture-chain identity. |
| `augmentation_speech.mix_mode` | explicit foreground/interferer level relationships: `physical` sums without rescaling — in practice a near-0 dB ratio, because sources are RMS-normalized and RIRs peak-normalized per channel, so the surviving distance cues are DRR / tail shape / tilt, not level; rescale modes draw a mode-specific SIR range, and a `distance_level` mode reinstates the level cue explicitly (SIR = 20·log10(d_itf/d_fg) + jitter from the scene's actual geometry). Real-far rows skip it (a simulated-near vs real-far level ratio is not physical). |

Pool manifests are one JSON object per line
(`wav_path` / `distance_m` / `room` / `speaker`), built by
`egs/voice_isolate/scripts/build_real_recording_pool.py`.

### Emitted labels

`_emit_task_metadata` attaches per-sample scalars (NaN where undefined):
distances/DRRs of foreground and strongest interferer, `drr_gap`, `rt60`,
`n_interferers`, `target_absent`/`target_present`, `has_background_speech`,
`near_count`/`far_count`, `mix_mode` code, `realized_speech_sir`, `noise_snr`,
`overlap_fraction`, `turn_taking`. These feed auxiliary supervision
(e.g. [`DistHeadRegressionLoss`](../nnet/loss/dist.md)) and eval bucketing.

## Class: `VoiceIsolationRowPlan`

`RowPlan` extended with `use_realnear` / `use_realfar` and the real-near pick
(room / speaker / metadata).

## Class: `VoiceIsolationCollateFunc`

`NoiseSuppressionCollateFunc` plus collation of the scalar labels above, the
optional `far_target` waveform, and background-VAD labels/references.

## Recipe wiring

```yaml
dataset:
  task: voice_isolation      # selects this dataset in egs/noise_suppression/main.py
augmentation_realfar:  {used: True, prob: 0.20, lone_far_prob: 0.15,
                        pool_manifest: data/realfar_pool/voices.train.jsonl,
                        turn_taking_prob: 0.3}
augmentation_realnear: {used: True, prob: 0.15, turn_taking_prob: 0.5,
                        pool_manifest: data/realfar_pool/voices.near.train.jsonl}
```

The reference recipe is `egs/voice_isolate/config/train_dpcrn.yaml`.
