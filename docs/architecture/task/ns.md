# puresound.task.ns

繁體中文版本：[`ns.zh-TW.md`](ns.zh-TW.md)

Generic noise-suppression dataset: on-the-fly synthesis of (noisy, clean) pairs
from a clean-speech metafile. It is also the **synthesis skeleton** the
voice-isolation task specialises through row-type hooks -- see
[task.voice_isolation](voice_isolation.md). There is one synthesis path; a task
changes it only by overriding hooks.

## Class: `NoiseSuppressionDataset`

Extends [`DynamicBaseDataset`](../dataset/dynamic_base.md). Per item:

1. parse the sampler key and draw one clean utterance for the sampled speaker,
   cropped or padded to the row length;
2. plan the row (`_plan_row`: target-absent or a task row type);
3. give the foreground its channel (`_prepare_foreground`: a source-level RIR
   from the room simulator or a pre-generated bank, or the waveform verbatim);
4. optionally add interfering speech (`_sample_interferers`), gate who talks
   when (`OverlapGating`), and mix foreground with interferers
   (`_mix_foreground_with_interferers`);
5. on target-absent rows, subtract the foreground from the mixture and zero the
   target;
6. optionally add residual playback echo (needs source-level reverb);
7. rescale the pair together if either clips, then apply speed perturbation and
   the whole-mix RIR (skipped when source-level reverb ran or the plan forbids
   it);
8. optionally silence every speech component over a row-initial ambient lead;
9. add noise (`NoiseStage`: recorded noise at an SNR, white noise, absolute
   capture floor), mixture only;
10. snapshot the target as the VAD reference, then run the capture and
    transmission chain (`DeviceChain`: SRC, IIR, HPF, volume, compressor, A/D
    boundary, codec, packet loss), with the target following the linear stages;
11. crop to the row length and emit labels.

Every block is optional, and an absent or disabled block draws nothing from the
RNG stream -- each probability draw sits inside its guard -- so adding a knob
does not change what an existing seeded recipe produces. The algorithms behind
steps 3-10 are described in
[Scene construction](../../algorithms/augmentation/scene_construction.md) and
[Device chain](../../algorithms/augmentation/device_chain.md).

### Constructor

```python
NoiseSuppressionDataset(
    metafile_path: str,
    min_utt_length_in_seconds: float = 3.0,
    min_utts_in_each_speaker: int = 5,
    target_sr: Optional[int] = None,
    training_sample_length_in_seconds: float = 6.0,
    audio_gain_normalized_to: Optional[int] = None,
    dataset_role: str = "train",
    pipeline_role: Optional[str] = None,
    curriculum=None,
    **augmentation,   # the blocks named in AUGMENTATION_BLOCKS
)
```

The constructor is `DynamicBaseDataset`'s. Augmentation blocks are keyword
arguments named `<block>_args`, accepted only if listed in the class attribute
`AUGMENTATION_BLOCKS`; an unlisted name raises `TypeError`. Each value is the
validated config model, or a mapping that is validated into it. This class adds
four blocks to the base set:

| keyword | config model | what it does |
| --- | --- | --- |
| `augmentation_speech_args` (base) | `SpeechAugmentation` | interferers: `prob`, `add_n_cases`, `snr_range`, `is_target`, `media_voice`, `echo_playback`, `overlap_control` |
| `augmentation_noise_args` (base) | `NoiseAugmentation` | recorded noise (`noise_folder` or weighted `noise_sources`), SNR range or bands, white noise, `room_coloring`, `absolute_floor` |
| `augmentation_reverb_args` (base) | `ReverbAugmentation` | room simulator, pre-generated bank or RIR folder; `target_rir_type`; source-level or whole-mix |
| `augmentation_speed_args` (base) | `ContinuousSpeedAugmentation` | speed perturbation over `speed_range` in 0.05 steps, both ends included |
| `augmentation_ir_response_args`, `augmentation_src_args`, `augmentation_hpf_args`, `augmentation_volume_args`, `augmentation_compressor_args` (base) | see `puresound.config.augmentation` | analogue stages of the device chain |
| `vad_label_args` (base) | `VadLabelConfig` | frame labels: energy backend, or Silero deferred to the GPU |
| `augmentation_codec_args` | `CodecAugmentation` | codec stage (mixture only) |
| `augmentation_packet_loss_args` | `PacketLossAugmentation` | packet-loss stage (mixture only) |
| `augmentation_target_absent_args` | `TargetAbsentAugmentation` | target-absent rows: `prob`, `force_interferer` |
| `augmentation_row_initial_ambient_args` | `RowInitialAmbientAugmentation` | ambient lead: `prob`, `lead_seconds_range`, `fade_ms` |

The three components composed from these blocks -- `device_chain`
(`device_chain_from_blocks`), `noise_stage` (`NoiseStage`) and `overlap_gating`
(`OverlapGating`) -- are built in `rebind_augmentation_blocks()`, which the
constructor calls, so a curriculum that changes a block mid-run reaches them too.

Task-specific blocks are refused rather than ignored: a used
`augmentation_speech.mix_mode` raises `ValueError` here (and
`NoiseSuppressionRecipe` rejects it at load time). The real-recording and
session blocks are not fields of `NoiseSuppressionRecipe`, so a recipe that sets
them fails schema validation unless it declares `task: voice_isolation`.

### `__getitem__(key) -> Dict`

`key` is the sampler's tuple `(speaker, sample_rate[, seed[, seconds[, epoch]]])`,
read by `DynamicBaseDataset.parse_item_key`. With a seed, every RNG the
synthesis uses is reseeded, so the same item regenerates bit-exact across
epochs, runs and worker layouts (see [task.sampler](sampler.md)).

Per-item dict:

| key | content |
| --- | --- |
| `noisy_speech`, `clean_speech` | `[1, L]` mixture and target |
| `consistency_noise` | `noisy_speech - clean_speech` |
| `far_target` | summed post-SIR interferer speech, zeros when there is none. No shipped loss reads it; it is the reference for far-speech leakage evaluation |
| `speaker_id`, `audio_sr`, `audio_length` | integer speaker index, sample rate, length |
| `vad_target` or `vad_reference` | frame labels, or the clean reference when Silero labelling is deferred to the GPU |
| `background_vad_target` or `background_vad_reference` | the same for background speech, only when an interferer was mixed in |
| `DEVICE_CHAIN_SCALARS` | what each device-chain stage did (`*_applied` 0/1, parameters NaN when a stage did not fire), as float tensors |
| `RIR_PROVENANCE_KEYS` | RIR lineage strings from the foreground's channel metadata (or the first interferer's), empty when there is none |
| `paired_view`, `row_source_id` | only when the task selects the row for a second capture view (below) |

### Tracing a row

`puresound.task.trace.recording(dataset)` records one row stage by stage for
the web pipeline inspector. At every stage boundary the skeleton, `NoiseStage`
and `DeviceChain` hand the pair as it stands, with what the stage drew, to
`dataset.trace`; every `apply_rir` call is kept with its impulse response.
Recording copies and never draws, so a traced row is bit-identical to the same
seeded row untraced. `STAGES` lists the stage ids in synthesis order; a stage
that did not act on the row records nothing. Training never installs a trace,
and only the primary device-chain view is traced.

### Row-type hooks

The skeleton calls these at every point where a task may substitute its own
row types. Each base implementation is the generic behaviour and draws exactly
what the generic row needs, in the same order:

| hook | decides |
| --- | --- |
| `_plan_row(target_speech)` | row type (`RowPlan`); may replace the foreground |
| `_prepare_foreground(target_speech, plan)` | the foreground's channel; returns `(source_level_reverb, room_scene, fg_metadata, noisy, target)` |
| `_sample_interferers(...)` | where interfering speech comes from; returns `(target, interferers, metadata)` |
| `_turn_taking_override(plan)` | row-level turn-taking rate (`None` = the `overlap_control` value) |
| `_mix_foreground_with_interferers(...)` | foreground/interferer level relationship; the base draws one SIR from `snr_range` and reports mix mode `"legacy"` |
| `_emit_task_metadata(sample, ...)` | per-row labels; the base writes the RIR provenance keys |
| `_emit_row_labels(sample, plan, ...)` | per-frame or per-turn labels, on the grid `vad_target` was computed from; the base has none |
| `_auxiliary_chain_view_probability(plan)` | probability of a second device-chain draw on the same finished mixture; the base returns 0 and draws nothing |
| `_emit_auxiliary_chain_view_labels(sample, plan)` | task labels for that second view |

When a second view is drawn (`puresound.task.paired_views.apply_chain_views`),
the row carries `paired_view` -- the second chain's `noisy_speech` and
`clean_speech`, cropped to the primary row -- and a shared random
`row_source_id`. A second view identical to the primary one is discarded.

## Class: `RowPlan`

Dataclass of per-item decisions a task makes before synthesis starts; task
subclasses extend it.

| field | effect |
| --- | --- |
| `target_absent` | strip the foreground from the mixture and zero the target |
| `force_interferer` | the interferer block fires regardless of its probability |
| `force_speech_interferers` | the same, without consuming the probability draw |
| `skip_whole_mix_reverb` | keep the whole-mix RIR off a row whose channel must stay as the foreground provided it |
| `skip_overlap_gating` | for a row type that wrote its own turn script |
| `speed_perturb_companions` | apply the speed change to the background-speech reference as well |
| `speed_factor` | written by the skeleton: the speed actually applied, for mapping a pre-speed script onto the post-speed label grid |

## Class: `NoiseSuppressionCollateFunc`

Pads and stacks the waveform keys, and renames three per-item scalars:

| `__getitem__` key | batch key |
| --- | --- |
| `speaker_id` | `spkid` |
| `audio_sr` | `sr` |
| `audio_length` | `length` |

- `noisy_speech`, `clean_speech`, `consistency_noise` keep their names, padded
  to the longest item.
- `vad_target` / `vad_reference` are padded and included when at least one item
  carries them.
- `DEVICE_CHAIN_SCALARS` are concatenated into one tensor per key (every row
  carries every key).
- `RIR_PROVENANCE_KEYS` pass through as plain Python lists, one entry per item.
- `paired_view` rows are collated by `collate_paired_views` into a nested
  `batch["paired_view"]` whose `source_indices` map each view to its primary
  row, plus a per-row `row_source_id` tensor. Auxiliary views are never appended
  to the primary batch, so ordinary losses still see B independent rows.

Not collated here: `far_target`, `background_vad_target`,
`background_vad_reference`. [`VoiceIsolationCollateFunc`](voice_isolation.md)
adds them, together with its scalar labels.

### Consumption in `system.siso`

[`EncDecMaskBase`](../system/siso.md) reads `batch["noisy_speech"]`,
`batch["clean_speech"]` and `batch.get("vad_target")` in its training and
validation steps, and passes the whole batch to `compute_loss`. A loss asks for
extra inputs by declaring `required_inputs` (for example `"batch"`,
`"vad_target"`, `"background_vad_target"`, `"inactive_labels"`); the module calls
it with exactly those through `invoke_loss` (see
[system.base](../system/base.md)). `BaseLightningModule.ensure_vad_targets`
reads the renamed `sr` key to label a deferred `vad_reference` /
`background_vad_reference` on the GPU; `test_step` and `predict_step` read `sr`
for per-metric resampling and for saving output at the input rate. `spkid` is
not read by the training loop; it is carried for tooling that wants per-row
speaker identity.
