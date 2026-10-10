# Building models and losses from a recipe

Traditional Chinese: [recipes.zh-TW.md](recipes.zh-TW.md)

`puresound.recipes` turns a validated recipe's `model` and `loss_func` sections
into objects. Model and loss types are resolved by name with `getattr` on
`puresound.nnet`, `puresound.system` and `puresound.nnet.loss`, so **everything
exported from those packages is reachable from a config** -- which is why the
model library keeps every backbone exported.

For complete recipes built on these functions, see the released lineages:
`egs/voice_isolate/config/train_dpcrn.yaml` and
`egs/noise_suppression/config/train_dpcrn_mamba_*.yaml`.

## From recipe to objects

Config loading lives in [`puresound.config`](configuration.md), not here. A
typed recipe carries its own field names (`recipe.dataset.train_metafile`,
`recipe.augmentation_kwargs()`), and this module only builds what it names:

```python
from puresound.config import load_recipe
from puresound.recipes import init_loss_func, init_model_for_task

recipe = load_recipe(
    path, expected_task="voice_isolation", expected_purpose="train"
)
model = init_model_for_task(recipe.task)(recipe.model)
loss_list, loss_weights = init_loss_func(recipe.loss_func)
model.register_loss_func(loss_list, loss_weights)
```

`init_model_for_task(task)` returns the model factory a task needs:
`init_siso_model` for noise suppression, voice isolation and speaker embedding,
`init_miso_model` for target speaker extraction. The shared training driver
(`puresound.system.runner`) does exactly this.

## `init_siso_model(model_dict) -> LightningModule`

Builds `encoder -> features -> backbone` and wraps them in the configured
Lightning module:

```yaml
model:
  lightning_module: {type: EncDecMaskBase, module_args: {mask_type: complex, ...}}
  encoder:          {type: ConvEncDec,     encoder_args: {...}}
  features:         {feats_type: complex, drop_stft_first_bin: True, ...}
  freq_eq:          {type: FrequencyEQLayer, eq_args: {...}}    # optional
  backbone:         {type: DPCRN,          backbone_args: {...}}
```

`lightning_module.type` is resolved on `puresound.system`; `encoder`, `freq_eq`
and `backbone` types on `puresound.nnet`. `features` are the keyword arguments of
`nnet.FeatureEncoder`; a `freq_eq` block is built first and passed to it.

`init_miso_model` takes the same shape plus `c_encoder` / `c_features` /
`c_backbone` for the conditioning branch (absent when `module_args.siamese_encoder`
reuses the mixture's encoder and features).

## `init_loss_func(loss_configs) -> (loss_list, weight_list)`

Each entry of the `loss_func` list:

```yaml
loss_func:
  - type: SDRLoss          # resolved on puresound.nnet.loss
    weighted: 1.0          # scalar weight
    args: {scaled: False}  # constructor kwargs
```

The training system's loss registry consumes both lists
(`model.register_loss_func(loss_list, weight_list)`). A `curriculum` track
`loss:<Type>` or `loss:#<index>` moves one entry's weight by epoch (see
[configuration.md](configuration.md#curriculum)).
