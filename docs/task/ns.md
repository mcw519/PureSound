# puresound.task.ns

繁體中文版本：[`ns.zh-TW.md`](ns.zh-TW.md)

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
    augmentation_noise_args=None,         # background + white noise, SNR ranges;
                                          #   optional room_coloring (noise gets a
                                          #   channel of the speech's room) and
                                          #   absolute_floor (dBFS-anchored capture
                                          #   floor, mixture only)
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

The 3-tuple form carries a per-item seed (deterministic validation, set by a
seeded [`SpeakerSampler`](sampler.md)): every RNG the synthesis uses is
reseeded so the same item regenerates bit-exact across epochs, runs, and
worker layouts.

Returns a per-item dict: `noisy_speech`, `clean_speech`, `added_noise` (`None`
if no noise was added), `consistency_noise` (`noisy - clean`), `far_target`
(summed post-SIR interferer signal, zeros when there is none -- consumed only
by an optional far decoder, see [task.voice_isolation](voice_isolation.md)),
`speaker_id`, `audio_sr`, `audio_length`, VAD labels (`vad_target`, or a
deferred `vad_reference` when `vad_label_args.backend` is `silero`) and the
same target/reference pair for background speech
(`background_vad_target`/`background_vad_reference`, present only when an
interferer was actually mixed in), plus the `_emit_task_metadata` hook's
RIR-provenance scalars (`RIR_PROVENANCE_KEYS`: empty strings/tuples when no
RIR metadata is available for that item).

Several of these per-item keys are **renamed or silently dropped** by
`NoiseSuppressionCollateFunc` -- see below.

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

Pads and stacks the waveform keys into batch tensors, and **renames** three
per-item scalar keys along the way:

| `__getitem__` key | collated batch key |
|---|---|
| `speaker_id` | `spkid` |
| `audio_sr` | `sr` |
| `audio_length` | `length` |

`noisy_speech`, `clean_speech`, `consistency_noise` keep their names (each
padded to the batch's longest item). `vad_target`/`vad_reference` are padded
and included only if at least one item in the batch carries them. The
`RIR_PROVENANCE_KEYS` (`rir_release_id`, `rir_release_sha256`,
`rir_recipe_id`, `rir_variant_id`, `rir_split`, `rir_origin`,
`rir_renderer_profile_id`, `rir_production_certificate_sha256`,
`rir_interferer_variant_ids`) pass through as plain Python lists -- one entry
per batch item, *not* padded/stacked into a tensor -- whenever any item
carries them.

**Dropped at this collate step** (present in `__getitem__`'s per-item dict,
but not read by `NoiseSuppressionCollateFunc`): `added_noise`, `far_target`,
`background_vad_target`, `background_vad_reference`. A caller that needs them
has to write its own collate function or use
[`VoiceIsolationCollateFunc`](voice_isolation.md), which layers `far_target`
and the `background_vad_*` pair (plus its own scalar labels) on top of this
base behaviour.

### Consumption in `system.siso`

[`EncDecMaskBase`](../system/siso.md) (the SISO trainer) reads
`batch["noisy_speech"]`, `batch["clean_speech"]` and
`batch.get("vad_target")` directly in `training_step`/`validation_step`, and
passes the whole batch dict through to `compute_loss`, so a registered loss
can opt in to extra keys by setting a flag attribute on itself
(`uses_batch`, `uses_vad_logits`, `uses_background_vad_logits`,
`uses_dist_preds`, `uses_vad_target`, `uses_inactive_labels`) instead of
requiring a fixed call signature. `BaseLightningModule.ensure_vad_targets`
(see [system.base](../system/base.md)) reads the renamed `sr` key to pick the
right sample rate when it lazily labels a deferred `vad_reference` /
`background_vad_reference` on GPU (the `silero` backend case).
`test_step`/`predict_step` also read `sr`, for per-metric resampling and to
save inference output at the original rate. The renamed `spkid` key is
carried through the collate step but is not read anywhere in `siso.py`
itself -- it exists for downstream tooling that wants per-sample speaker
identity (e.g. dumping training samples), not for the training loop.
