# puresound.task.ns

Generic noise-suppression dataset: on-the-fly synthesis of (noisy, clean) pairs
from a clean-speech metafile. This is also the **synthesis skeleton** that the
voice-isolation task specializes -- see [task.voice_isolation](voice_isolation.md).

## Class: `NoiseSuppressionDataset`

Extends `DynamicBaseDataset`. Per item: pick a clean utterance for the sampled
speaker, optionally give it a room channel (source-level RIR from the on-the-fly
simulator or a pre-generated bank), optionally add interfering speakers /
playback echo / noise, then run the device chain (speed, whole-mix RIR, SRC,
IIR, HPF, codec, packet loss, volume) with the clean target warped consistently.

### Constructor (all blocks optional; absent/disabled blocks never consume RNG)

```python
NoiseSuppressionDataset(
    metafile_path: str,
    min_utt_length_in_seconds: float = 3.0,
    min_utts_in_each_speaker: int = 5,
    target_sr: Optional[int] = None,
    training_sample_length_in_seconds: float = 6.0,
    audio_gain_normalized_to: Optional[int] = None,
    augmentation_speech_args=None,        # interferers: prob/add_n_cases/snr_range/
                                          #   media_voice/echo_playback/overlap_control
    augmentation_noise_args=None,         # background + white noise, SNR ranges
    augmentation_reverb_args=None,        # simulator / pre-generated bank / whole-mix RIR
    augmentation_speed_args=None,
    augmentation_ir_response_args=None,
    augmentation_src_args=None,
    augmentation_hpf_args=None,
    augmentation_volume_args=None,
    augmentation_codec_args=None,
    augmentation_packet_loss_args=None,
    augmentation_target_absent_args=None, # silence-injection rows (target = zeros)
    vad_label_args=None,                  # frame labels: energy backend or deferred Silero
)
```

Task-specific blocks are rejected here (`mix_mode` raises; the recipe main
rejects `augmentation_realfar/realnear` without `dataset.task: voice_isolation`).

### `__getitem__((speaker, sr) | (speaker, sr, item_seed)) -> Dict`

The 3-tuple form carries a per-item seed (deterministic validation): every RNG
the synthesis uses is reseeded so the same item regenerates bit-exact across
epochs, runs, and worker layouts.

Returns `noisy_speech`, `clean_speech`, `consistency_noise`
(`noisy - clean`), `sr`, VAD labels (or a deferred `vad_reference`), and the
scalar metadata hooks' output.

### Row-type hooks

The skeleton delegates every point where a task may substitute its own row
types; each base implementation *is* the generic behaviour and draws nothing
extra from the RNG stream:

| hook | decides |
|---|---|
| `_plan_row(target_speech)` | row type (`RowPlan`); may replace the foreground |
| `_prepare_foreground(target_speech, plan)` | the foreground's channel (room sim or verbatim) |
| `_sample_interferers(...)` | where interfering speech comes from |
| `_turn_taking_override(plan)` | row-level turn-taking rate |
| `_mix_foreground_with_interferers(...)` | foreground/interferer level relationship |
| `_emit_task_metadata(...)` | extra per-sample labels |

## Class: `RowPlan`

Dataclass of per-item decisions: `target_absent`, `force_interferer`,
`force_speech_interferers`, `skip_whole_mix_reverb`. Task subclasses extend it.

## Class: `NoiseSuppressionCollateFunc`

Pads and stacks the waveform keys (plus VAD labels/references) into batch
tensors.
