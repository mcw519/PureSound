# Typed recipe configuration

PureSound recipes are validated by Pydantic v2 before a dataset, model or
trainer is constructed. The public entry point is `puresound.config.load_recipe`, and it is the only
one: the twenty-tuple `recipes.load_siso_recipe_config` and its dict adapter are
gone, so nothing reaches a dataset without passing through a model.

## Dispatch model

A recipe has two discriminators:

- `purpose`: `train` or `inference`;
- `task`: `noise_suppression`, `voice_isolation`,
  `target_speaker_extraction`, or `speaker_embedding`.

Every config must declare both plus `schema_version: 2`. There is no legacy
inference or aliasing: a missing discriminator is an invalid recipe rather than
a request for the loader to guess from unrelated pipeline fields.

```yaml
schema_version: 2
purpose: train
task: voice_isolation
```

## Ownership

Config models follow domain capabilities, not Python dataset inheritance.
Reusable augmentation models live in `puresound.config.augmentation`; task
recipes compose only the capabilities their pipeline consumes:

| task | speed contract | task-only pipeline blocks |
|---|---|---|
| noise suppression | continuous `speed_range` | codec, packet loss, target absent |
| voice isolation | continuous `speed_range` | NS channel effects plus mix modes, real-far and real-near rows |
| target speaker extraction | continuous `speed_range` | enrollment pipeline and split signal/class losses |
| speaker embedding | discrete `speed_change` | `treat_as_new_speaker` |

Unknown keys and coercible-but-wrong scalar types are rejected at every modeled level. Conditional contracts are
validated with their owning capability: enabled reverb needs a folder or active
simulator, probability fields lie in `[0, 1]`, weighted choices have matching
lengths, and codec bitrate keys must match selected codecs.

Component constructor arguments under `model`, loss `args`, optimizer `args`,
and scheduler `args` remain explicit extension boundaries. Their selected
constructors validate those kwargs. Pipeline control-flow blocks must not use
untyped dictionaries.

## Runtime boundary

Pydantic validation is structural and semantic; it does not load audio, scan a
manifest, initialize CUDA or instantiate torch modules. Filesystem and backend
availability stay in runtime preflight/construction so config linting remains
lightweight.

```python
from puresound.config import load_recipe

recipe = load_recipe(
    "config/train_dpcrn.yaml",
    expected_task="voice_isolation",
    expected_purpose="train",
)
```

Executable entry points must pass both expectations. This catches a valid
recipe launched through the wrong task or a model-only recipe passed to a
training/batch-evaluation runner before anything expensive is constructed.

## Pipeline stages

`dataset_role` records whether a dataset instance is training, validation or
test. `train_pipeline_role` and `validation_pipeline_role` independently select
the data-source role used by stage-sensitive pipeline components such as an RIR
bank. They default to their matching stage; an experiment that intentionally
validates against the training room distribution must say
`validation_pipeline_role: train` explicitly.

## Adding a capability or task

1. Add or extend the capability model next to the other models in
   `puresound.config.augmentation`.
2. Compose it into only the task recipe classes that consume it.
3. Add cross-field validation beside the capability, not in an entry point.
4. Add positive shipped-config coverage and negative type/typo/contract tests.
5. Change all active YAML documents atomically with the schema. Historical,
   non-executable configs belong in version control history, not the live config tree.
