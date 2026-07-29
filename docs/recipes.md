# puresound.recipes

Config-driven construction: a YAML recipe in, a ready model / loss list out.
Model and loss types are resolved by name with `getattr` on `puresound.nnet` and
`puresound.nnet.loss`, so **everything exported from those packages is reachable
from a config** (the model library keeps all backbones exported for exactly this
reason).

## `load_siso_recipe_config(f_path) -> Tuple`

Parses a recipe YAML into a positional 20-tuple. Consumers should unpack the
fields they need and absorb growth with a star target — the tuple grows by
appending:

| index | field | yaml section |
|---|---|---|
| 0 | corpus | `dataset` |
| 1 | trainer | `trainer` (incl. `lightning_trainer_args`) |
| 2 | optimizer | `optimizer` |
| 3 | scheduler | `scheduler` |
| 4 | loss | `loss_func` (list) |
| 5 | model | `model` (incl. `lightning_module`) |
| 6–13 | augmentation blocks | `augmentation_speech`, `_noise`, `_reverb`, `_speed`, `_ir_response`, `_src`, `_hpf`, `_volume` |
| 14–16 | more blocks | `augmentation_codec`, `_packet_loss`, `_target_absent` |
| 17 | vad label | `vad_label` |
| 18–19 | real-recording blocks | `augmentation_realfar`, `augmentation_realnear` (voice-isolation only) |

Blocks gated by `used: False` (or absent) come back as `None`.

```python
(corpus, trainer, _opt, _sch, _loss, model_dict, *rest) = load_siso_recipe_config(path)
```

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
