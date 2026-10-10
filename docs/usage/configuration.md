# Recipe configuration

Traditional Chinese: [configuration.zh-TW.md](configuration.zh-TW.md)

PureSound loads YAML recipes through `puresound.config.load_recipe`. Pydantic
validates the complete recipe before any dataset, model or trainer is created, so
a typo or a missing field fails at load time rather than hours into a run.

## Required fields

Every recipe declares a schema version, purpose and task:

```yaml
schema_version: 2
purpose: train
task: voice_isolation
```

| Field | Values |
| --- | --- |
| `purpose` | `train` (a full training recipe) or `inference` (model-only: `dataset.target_sample_rate`, `trainer.work_folder` and `model`, used by demos, benchmarks and streaming export) |
| `task` | `noise_suppression`, `voice_isolation`, `speaker_embedding`, `target_speaker_extraction` |

Unknown fields and invalid scalar types are errors. `speaker_embedding` and
`target_speaker_extraction` are frozen legacy tasks.

## Load a recipe

```python
from puresound.config import load_recipe

recipe = load_recipe(
    "egs/voice_isolate/config/train_dpcrn.yaml",
    expected_task="voice_isolation",
    expected_purpose="train",
)
```

Entry points pass both expected values, so a valid config cannot be used by the
wrong runner: `egs/noise_suppression/main.py` refuses a `voice_isolation` recipe
and the other way round.

Validation checks structure and field relationships. Audio files, manifests,
CUDA and optional backends are checked later, when the pipeline is constructed.

## Task-specific fields

A recipe model exposes only the blocks its task uses; a block from another task
is an unknown field.

| Task | Speed setting | Task-specific blocks |
| --- | --- | --- |
| Noise suppression | continuous `augmentation_speed.speed_range` | `augmentation_codec`, `augmentation_packet_loss`, `augmentation_target_absent`, `augmentation_row_initial_ambient` |
| Voice isolation | continuous `augmentation_speed.speed_range` | the same, plus `augmentation_speech.mix_mode`, `augmentation_realfar`, `augmentation_realnear`, `augmentation_session_rows` |
| Speaker embedding | discrete `augmentation_speed.speed_change` | `treat_as_new_speaker` |
| Target speaker extraction | continuous `augmentation_speed.speed_range` | `enroll_speech`, `signal_loss_func`, `class_loss_func` |

Constructor arguments under `model`, `loss_func`, `optimizer` and `scheduler` are
validated by the selected component when it is built (see
[recipes.md](recipes.md)).

## Dataset roles

`dataset_role` identifies a train, validation or test dataset.
`dataset.train_pipeline_role` and `dataset.validation_pipeline_role` select
stage-sensitive resources such as an RIR-bank split.

Validation uses the validation role by default. To validate against the training
room distribution, set it explicitly:

```yaml
dataset:
  validation_pipeline_role: train
```

## Sampler knobs

The `trainer` block carries two knobs that change what a training batch is made
of; validation ignores both, so its loss stays comparable across epochs.

```yaml
trainer:
  n_spk_per_batch: 4          # validation, and the batch without a length_schedule
  length_schedule:            # training: each batch draws one (length, batch size)
    - {seconds: 6,  n_spk: 4, prob: 0.6}
    - {seconds: 12, n_spk: 2, prob: 0.3}
    - {seconds: 30, n_spk: 1, prob: 0.1}
  speaker_source_weights: {dns5_: 0.7, ll_: 0.3}
```

- `length_schedule` supervises the model at several context lengths instead of
  one. The batch size travels with the length because activation memory does;
  probabilities must sum to 1. Long rows need a corpus whose utterances are that
  long.
- `speaker_source_weights` chooses each corpus's share of the batches by spkid
  prefix instead of leaving it to speaker counts. Its rules, and the noise and
  SNR knobs `augmentation_noise.noise_sources` and `snr_bands`, are in
  [data_preparation.md](data_preparation.md#recipe-knobs-these-feed).
- `train_sampler: coverage` (default `speaker`) draws training batches from
  shuffled queues: each source visits its speakers before repeating, and each
  speaker cycles through its eligible utterances for that row length. Long rows
  only take utterances that fill them. Checkpoints preserve the walk for resumes
  at epoch boundaries and warm-started stages; partial-epoch checkpoints require
  a warm start. Resumes require unchanged stage settings and eligible queues.
  Needs `n_utt_per_speaker: 1` and `target_sample_rate`. See
  [task.sampler](../architecture/task/sampler.md#class-coveragesampler).

## Curriculum

The optional `curriculum` block changes selected values by epoch, so one run can
express what would otherwise be a chain of warm-started runs:

```yaml
curriculum:
  used: true
  tracks:
    - path: "loss:OverSuppressionLoss"
      interp: linear
      points: [[0, 0.0], [30, 3.0]]
    - path: "aug:augmentation_noise.prob"
      interp: step
      points: [[0, 0.0], [40, 0.8]]
    - path: "bank:wide"
      interp: linear
      points: [[0, 0.1], [40, 0.5]]
```

| Prefix | Target |
| --- | --- |
| `aug:` | an allowlisted augmentation field (`SCHEDULABLE_AUGMENTATION_PATHS` in `puresound/config/curriculum.py`) |
| `bank:` | a named member of a union RIR bank; weights are relative and renormalised |
| `loss:` | a registered loss type, or `#index` into `loss_func` |

`linear` interpolates between points. `step` holds the previous value until the
next point. Values outside the point range use the nearest endpoint.

Restrictions:

- Schedule a block's probability, not its `used` flag. Synthesis skips a
  disabled block before its random draw, so toggling `used` would shift every
  draw after it. A block a schedule introduces is `used: true` with its
  probability at the schedule's first value.
- Paths, bank layouts, cache sizes and anything else read during construction
  cannot be scheduled; the allowlist refuses them.
- Validation data does not follow the training curriculum: it reads the file's
  constants.
- Resumed training continues from the restored epoch.
- A loss with weight zero is still evaluated so stateful losses stay current.

Invalid curriculum targets -- a disabled block, a bank member not in the union, a
loss that is not registered -- fail during recipe loading. The run logs the
epoch-0 value of every track when training starts.

## Adding a field

1. Add the field to the capability model in `puresound.config`.
2. Include that capability only in the tasks that use it.
3. Put cross-field validation beside the capability.
4. Update active YAML files and tests in the same change.
