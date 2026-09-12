# puresound.recipes

繁體中文版本：[recipes.zh-TW.md](recipes.zh-TW.md)

Config-driven construction: a YAML recipe in, a ready model / loss list out.
Model and loss types are resolved by name with `getattr` on `puresound.nnet` and
`puresound.nnet.loss`, so **everything exported from those packages is reachable
from a config** (the model library keeps all backbones exported for exactly this
reason).

See `egs/voice_isolate/config/train_dpcrn.yaml` for a complete, real recipe
built on these functions — it's the config behind the released voice-isolate
checkpoints.

## Loading a recipe

Config loading lives in [`puresound.config`](configuration.md), not here. This
module only builds the objects a validated recipe names.

```python
from puresound.config import load_recipe

recipe = load_recipe(
    path, expected_task="voice_isolation", expected_purpose="train"
)
model = init_siso_model(recipe.model)
loss_list, loss_weights = init_loss_func(recipe.loss_func)
```

`load_siso_recipe_config` and its positional twenty-tuple are gone. The tuple
existed so callers could unpack "the config"; a typed recipe carries its own
names, so `recipe.dataset.train_metafile` replaces index 0 and
`recipe.augmentation_kwargs()` replaces indices 6-19 in one call. Which blocks
arrive as `None` when disabled is now decided in one place --
`BaseRecipe.augmentation_kwargs` -- instead of by whether a given index went
through `_enabled_config` or a plain `config.get`.

## `init_siso_model(model_dict) -> LightningModule`

Builds `encoder -> features -> backbone` and wraps them in the configured
Lightning module:

```yaml
model:
  lightning_module: {type: EncDecMaskBase, module_args: {mask_type: complex, ...}}
  encoder:          {type: ConvEncDec,     encoder_args: {...}}
  features:         {feats_type: complex, drop_stft_first_bin: True, ...}
  freq_eq:          {type: FrequencyEQLayer, ...}    # optional
  backbone:         {type: DPCRN,          backbone_args: {...}}
```

`type` strings are resolved on `puresound.system` (lightning module) and
`puresound.nnet` (encoder / freq_eq / backbone).

## `init_loss_func(hparam_conf) -> (loss_list, weight_list)`

Each entry of the `loss_func` yaml list:

```yaml
loss_func:
  - type: SDRLoss          # resolved on puresound.nnet.loss
    weighted: 1.0          # scalar weight
    args: {scaled: False}  # constructor kwargs
```

The training system's loss registry consumes both lists
(`model.register_loss_func(loss_list, weight_list)`).
