# puresound.system.optim

繁體中文版本：[`optim.zh-TW.md`](optim.zh-TW.md)

Reflection-based optimizer/LR-scheduler factory. There is no restricted enum of supported optimizer/scheduler names: `type` strings are looked up directly on `torch.optim` / `torch.optim.lr_scheduler` via `getattr`, so anything those modules export is usable. An unrecognized name fails with `AttributeError`, not a friendly config-validation error.

## Function: `create_optimizer_and_scheduler`

```python
create_optimizer_and_scheduler(
    overall_params_and_lr_factor: Dict,
    optimizer_args: Dict,
    scheduler_args: Dict,
) -> Tuple[Optimizer, LRScheduler]
```

### Parameters

- **`overall_params_and_lr_factor`** — `{group_name: {"params": Iterable[nn.Parameter], "lr_factor": float}}`. Not built by this function: it is produced by a Lightning module's own `get_total_param_groups()` (`siso.EncDecMaskBase`, `siso.EncPredClassBase`, `miso.EncDecCondMaskBase` — see [siso.md](siso.md)/[miso.md](miso.md)), one dict entry per named parameter group (e.g. `encoder`/`feats`/`backbone`, or a single `gate_head` group in `EncDecMaskBase`'s VAD-gate-only mode).
- **`optimizer_args`** — `{"type": str, "learning_rate": float, "args": Dict}`.
  - `type` — a class name in `torch.optim` (`Adam`, `AdamW`, `SGD`, `RAdam`, ...), resolved via `getattr(torch.optim, type)`.
  - `learning_rate` — the base LR; each group's effective rate is `lr_factor * learning_rate`. It is *also* passed positionally as the optimizer constructor's own `lr=` argument — every group here already carries an explicit `"lr"`, so that positional value never actually ends up setting any group's rate, but some optimizer classes require `lr` regardless of whether every param group supplies its own.
  - `args` — forwarded as `**kwargs` to the optimizer class (`weight_decay`, `betas`, `momentum`, ...).
- **`scheduler_args`** — `{"type": str, "args": Dict}`.
  - `type` — a class name in `torch.optim.lr_scheduler` (`StepLR`, `CosineAnnealingWarmRestarts`, `ReduceLROnPlateau`, ...), resolved via `getattr(torch.optim.lr_scheduler, type)`.
  - `args` — forwarded as `**kwargs` to the scheduler class.
  - Real recipes' YAML `scheduler:` blocks also carry a sibling `warmup_step` key — this function never reads it. It is pulled out separately by the training script and handed to [`BaseLightningModule.register_warmup_step()`](base.md), which drives a manual linear-warmup ramp inside `optimizer_step` — a mechanism entirely outside this module.

### Returns

`(optimizer, scheduler)` — always a 2-tuple. There is no "no scheduler" path: `scheduler_args["type"]` is required (`KeyError` otherwise), so a scheduler is always constructed and returned.

### Real config example

`egs/voice_isolate/config/train_dpcrn.yaml`:

```yaml
optimizer:
  type: AdamW
  learning_rate: 0.001
  args:
    weight_decay: 0.00001
    betas: [0.9, 0.999]

scheduler:
  # warmup_step is read separately by main.py -> register_warmup_step;
  # create_optimizer_and_scheduler itself only ever sees type/args.
  type: CosineAnnealingWarmRestarts
  warmup_step: 250
  args:
    T_0: 20
    T_mult: 1
    eta_min: 0.00001
```

`AdamW` is the only optimizer used across current recipes; scheduler choice varies — `egs/noise_suppression/config/dpcrn.yaml` uses `StepLR` instead:

```yaml
scheduler:
  type: StepLR
  args: {step_size: 10, gamma: 0.5}
```

Both go through the same reflection path — switching from one to the other is a config edit, not a code change.

### Real wiring (`puresound/system/runner.py`)

```python
param_groups = lightning_model.get_total_param_groups()
optimizer, scheduler = create_optimizer_and_scheduler(
    overall_params_and_lr_factor=param_groups,
    optimizer_args=optim_dict,       # config["optimizer"]
    scheduler_args=scheduler_dict,   # config["scheduler"]
)
lightning_model.register_optimizer(optimizer)
lightning_model.register_scheduler(scheduler)
lightning_model.register_warmup_step(scheduler_dict["warmup_step"])
```

## Example

```python
from puresound.system.optim import create_optimizer_and_scheduler

groups = model.get_total_param_groups()  # e.g. {"encoder": {...}, "feats": {...}, "backbone": {...}}
optimizer, scheduler = create_optimizer_and_scheduler(
    overall_params_and_lr_factor=groups,
    optimizer_args={"type": "AdamW", "learning_rate": 1e-3, "args": {"weight_decay": 1e-5}},
    scheduler_args={"type": "StepLR", "args": {"step_size": 10, "gamma": 0.5}},
)
model.register_optimizer(optimizer)
model.register_scheduler(scheduler)

# In PyTorch Lightning, BaseLightningModule already implements configure_optimizers()
# to return [self._optimizer], [self._scheduler] from these two registrations.
```
