# Recipe configuration

PureSound loads YAML recipes through `puresound.config.load_recipe`. Pydantic
validates the complete recipe before any dataset, model, or trainer is created.

## Required fields

Every recipe declares a schema version, purpose, and task:

```yaml
schema_version: 2
purpose: train
task: voice_isolation
```

Supported purposes:

- `train`
- `inference`

Supported tasks:

- `noise_suppression`
- `voice_isolation`
- `speaker_embedding`
- `target_speaker_extraction`

Unknown fields and invalid scalar types are errors.

## Load a recipe

```python
from puresound.config import load_recipe

recipe = load_recipe(
    "egs/voice_isolate/config/train_dpcrn.yaml",
    expected_task="voice_isolation",
    expected_purpose="train",
)
```

Entry points should pass both expected values. This prevents a valid config from
being used by the wrong runner.

Validation checks structure and field relationships. Audio files, manifests,
CUDA, and optional backends are checked later when the pipeline is constructed.

## Task-specific fields

Recipe models expose only the blocks used by their task.

| Task | Speed setting | Task-specific blocks |
| --- | --- | --- |
| Noise suppression | continuous `speed_range` | codec, packet loss, target-absent rows |
| Voice isolation | continuous `speed_range` | mix modes, real-near and real-far rows |
| Speaker embedding | discrete `speed_change` | `treat_as_new_speaker` |
| Target speaker extraction | continuous `speed_range` | enrollment and split losses |

Constructor arguments under model, loss, optimizer, and scheduler sections are
validated by the selected component.

## Dataset roles

`dataset_role` identifies a train, validation, or test dataset.
`train_pipeline_role` and `validation_pipeline_role` select stage-sensitive
resources such as an RIR-bank split.

Validation uses the validation role by default. To validate against the training
room distribution, set it explicitly:

```yaml
validation_pipeline_role: train
```

## Curriculum

The optional `curriculum` block changes selected values by epoch:

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
| `aug:` | An allowlisted augmentation field |
| `bank:` | A named member of a union RIR bank |
| `loss:` | A registered loss type or `#index` |

`linear` interpolates between points. `step` holds the previous value until
the next point. Values outside the point range use the nearest endpoint.

Restrictions:

- Schedule a block's probability, not its `used` flag. Toggling `used` changes
  RNG consumption.
- Paths and cache sizes read during construction cannot be scheduled.
- Validation data does not follow the training curriculum.
- Resumed training continues from the restored epoch.
- A loss with weight zero is still evaluated so stateful losses remain current.

Invalid curriculum targets fail during recipe loading.

## Adding a field

1. Add the field to the capability model in `puresound.config`.
2. Include that capability only in tasks that use it.
3. Put cross-field validation beside the capability.
4. Update active YAML files and tests in the same change.
