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

## Curriculum: knobs that move with the epoch

`curriculum` (optional, `puresound.config.curriculum`) lets one run walk a knob
from one value to another, where a recipe otherwise states it once and trains at
that value throughout. Without it the only way to express a knob that should
change partway is to split the run and warm-start the second half from the first,
which leaves the schedule outside the recipe.

```yaml
curriculum:
  used: True
  tracks:
    - path: "loss:OverSuppressionLoss"        # a registered loss's weight
      interp: linear
      points: [[0, 0.0], [30, 3.0]]
    - path: "aug:augmentation_noise.prob"     # a knob a row consults
      interp: step
      points: [[0, 0.0], [40, 0.80]]
    - path: "bank:wide"                       # a union RIR bank member's weight
      points: [[0, 0.1], [40, 0.5]]
```

`linear` interpolates between neighbouring points, `step` holds a value until
the next point's epoch. Outside the first and last points the value is held
flat: a track never extrapolates.

| prefix | reference | applied by |
|---|---|---|
| `aug:` | `<block>.<field>`, restricted to `SCHEDULABLE_AUGMENTATION_PATHS` | the dataset, in the DataLoader workers |
| `bank:` | a `banks:` member's `name` (unnamed members are `bank<index>`) | `UnionRoomBank.set_weights`, relative weights, renormalised |
| `loss:` | a loss `type`, or `#<index>` when a type appears more than once | `CurriculumCallback`, on the module |

Three properties the mechanism is built to keep:

* **Off costs nothing.** Without a `curriculum` block the sampler emits the item
  shape it always did and no knob is ever rebuilt, so a run is comparable with
  every run before it. A schedule whose values equal the recipe's constants
  produces bit-identical rows.
* **Validation does not follow the ramp.** Only the training dataset is given
  the curriculum, so validation loss stays comparable across epochs.
* **The epoch is told, not counted.** Lightning calls `set_epoch` on the batch
  sampler before each epoch's iterator is consumed, including a resumed one, and
  the epoch travels to the workers on the item itself. A run resumed at epoch N
  continues its schedule at N rather than replaying it from 0.

Two things are deliberately not schedulable, both because the failure would be
silent. A block's `used` flag: synthesis skips a disabled block *before* its
random draw, so switching it mid-run shifts every draw after it — ramp `prob`
from zero instead, which keeps the draw and changes only its outcome. And
anything read once when the dataset is constructed (corpus folders, bank
layouts, cache sizes): `SCHEDULABLE_AUGMENTATION_PATHS` refuses those by
omission.

A knob still counts as re-read per row when the row reads it through a component
composed from its block -- a capture chain, a noise stage, a gating helper.
Those components are built once and hold the block they were given, so a dataset
re-derives them in `rebind_augmentation_blocks()` whenever a scheduled value
moves. Extending the allowlist means checking one of the two: the row reads the
knob directly, or the component that does is re-derived there.

A loss at weight 0 is still evaluated — `reduce_losses` weights every registered
term rather than skipping it, deliberately, because a loss carrying internal
state (an EMA bank) would otherwise go stale exactly while a ramp waits to start
it. Ramping in a cheap term costs nothing worth measuring; ramping in an
`ASRFeatureLoss` pays for its frozen SSL forward from epoch 0.

A track that names a block this recipe disables, a bank member that is not in
the union, or a loss it does not register is a load-time error
(`BaseRecipe.curriculum_targets_exist`) — the whole point is that the recipe and
the run cannot disagree about what was trained.

## Adding a capability or task

1. Add or extend the capability model next to the other models in
   `puresound.config.augmentation`.
2. Compose it into only the task recipe classes that consume it.
3. Add cross-field validation beside the capability, not in an entry point.
4. Add positive shipped-config coverage and negative type/typo/contract tests.
5. Change all active YAML documents atomically with the schema. Historical,
   non-executable configs belong in version control history, not the live config tree.
