# puresound.system.optim

繁體中文版本：[`optim.zh-TW.md`](optim.zh-TW.md)

Reflection-based optimizer/LR-scheduler factory. There is no restricted list of supported names: `type` strings are looked up directly on `torch.optim` / `torch.optim.lr_scheduler` via `getattr`, so anything those modules export is usable, and switching optimizer or scheduler is a config edit, not a code change. The config models check the shape of the block when a recipe loads; the name itself is resolved only when this function runs, so an unrecognized name fails here with `AttributeError`.

## Function: `create_optimizer_and_scheduler`

```python
create_optimizer_and_scheduler(
    overall_params_and_lr_factor: Dict,
    optimizer_args: OptimizerConfig,
    scheduler_args: SchedulerConfig,
) -> Tuple[Optimizer, LRScheduler]
```

`OptimizerConfig` and `SchedulerConfig` are the typed recipe blocks from `puresound.config.recipe` (strict Pydantic models: unknown keys are rejected and fields are read as attributes, so a plain dict is not accepted).

### Parameters

- **`overall_params_and_lr_factor`** — `{group_name: {"params": Iterable[nn.Parameter], "lr_factor": float}}`. Not built by this function: it is produced by a Lightning module's own `get_total_param_groups()` (`siso.EncDecMaskBase`, `siso.EncPredClassBase`, `miso.EncDecCondMaskBase` — see [siso.md](siso.md)/[miso.md](miso.md)), one entry per named parameter group (e.g. `encoder`/`feats`/`backbone`, plus `metric_disc` when MetricGAN is on, or a single `gate_head` group in `EncDecMaskBase`'s VAD-gate-only mode).
- **`optimizer_args`** — `OptimizerConfig(type: str, learning_rate: float > 0, args: dict = {})`.
  - `type` — a class name in `torch.optim` (`Adam`, `AdamW`, `SGD`, `RAdam`, ...), resolved via `getattr(torch.optim, type)`.
  - `learning_rate` — the base LR; each group's effective rate is `lr_factor * learning_rate`. It is also passed positionally as the optimizer constructor's own `lr` argument. Every group already carries an explicit `"lr"`, so that value never sets any group's rate, but some optimizer classes require `lr` regardless.
  - `args` — forwarded as `**kwargs` to the optimizer class (`weight_decay`, `betas`, `momentum`, ...).
- **`scheduler_args`** — `SchedulerConfig(type: str, warmup_step: int, args: dict = {})`.
  - `type` — a class name in `torch.optim.lr_scheduler` (`StepLR`, `CosineAnnealingWarmRestarts`, ...), resolved via `getattr(torch.optim.lr_scheduler, type)`.
  - `args` — forwarded as `**kwargs` to the scheduler class.
  - `warmup_step` — required by the config model but never read by this function. The runner hands it to [`BaseLightningModule.register_warmup_step()`](base.md), which drives a manual linear-warmup ramp inside `optimizer_step` — a mechanism outside this module.

### Returns

`(optimizer, scheduler)` — always a 2-tuple. There is no "no scheduler" path: `SchedulerConfig.type` is a required field, so a scheduler is always constructed and returned.

### Recipe example

`egs/voice_isolate/config/train_dpcrn.yaml`:

```yaml
optimizer:
  type: AdamW
  learning_rate: 0.001
  args:
    weight_decay: 0.00001
    betas: [0.9, 0.999]

scheduler:
  type: CosineAnnealingWarmRestarts
  warmup_step: 250      # read by the runner, not by create_optimizer_and_scheduler
  args:
    T_0: 20
    T_mult: 1
    eta_min: 0.00001
```

`egs/noise_suppression/config/dpcrn.yaml` uses `StepLR` instead:

```yaml
scheduler:
  type: StepLR
  warmup_step: 1
  args: {step_size: 10, gamma: 0.5}
```

Both go through the same reflection path.

### Wiring (`puresound.system.runner.run_training`)

```python
param_groups = lightning_model.get_total_param_groups()
optimizer, scheduler = create_optimizer_and_scheduler(
    overall_params_and_lr_factor=param_groups,
    optimizer_args=recipe.optimizer,
    scheduler_args=recipe.scheduler,
)
lightning_model.register_optimizer(optimizer)
lightning_model.register_scheduler(scheduler)
lightning_model.register_warmup_step(recipe.scheduler.warmup_step)
```

## Example

```python
from puresound.config.recipe import OptimizerConfig, SchedulerConfig
from puresound.system.optim import create_optimizer_and_scheduler

groups = model.get_total_param_groups()  # e.g. {"encoder": {...}, "feats": {...}, "backbone": {...}}
optimizer, scheduler = create_optimizer_and_scheduler(
    overall_params_and_lr_factor=groups,
    optimizer_args=OptimizerConfig(
        type="AdamW", learning_rate=1e-3, args={"weight_decay": 1e-5}
    ),
    scheduler_args=SchedulerConfig(
        type="StepLR", warmup_step=1, args={"step_size": 10, "gamma": 0.5}
    ),
)
model.register_optimizer(optimizer)
model.register_scheduler(scheduler)

# BaseLightningModule.configure_optimizers() returns
# [self._optimizer], [self._scheduler] from these two registrations.
```
