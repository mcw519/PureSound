# puresound.system.base

繁體中文版本：[`base.zh-TW.md`](base.zh-TW.md)

Shared Lightning-module infrastructure that every PureSound training system (`EncDecMaskBase`, `EncDecCondMaskBase`, `EncPredClassBase`) inherits from: the loss registry and its input dispatch, optimizer/scheduler plumbing with LR warmup, epoch logging, GPU-batched VAD labeling, and a tolerant checkpoint loader. `BaseLightningModule` is abstract — it defines no model and cannot train on its own.

## Function: `invoke_loss`

```python
invoke_loss(loss_func: nn.Module, providers: dict) -> Any
```

Calls one loss with exactly the inputs it asks for, in its own order. A loss declares `required_inputs` — a tuple of provider names matching its `forward` signature, e.g. `("vad_logits", "vad_target")` — and the module hands over `providers`, a `{name: callable}` table of what it can supply. A loss that declares nothing gets `DEFAULT_LOSS_INPUTS = ("enhanced", "target")`.

Why it is built this way:

- **A declaration replaces the parent's rather than extending it**, so a subclass (`BackgroundVADHeadBCELoss` of `VADHeadBCELoss`) cannot silently inherit an input it does not want and train against the wrong head.
- **An unknown name is a `TypeError`, not a fallback.** Calling a loss with the waveform pair when its real inputs are unavailable would produce a plausible number instead of a crash; the error message lists what the module does provide.
- **Providers are callables**, so a side output nothing asked for is never read, and a provider that synthesizes a missing target runs only when a registered loss needs it.

## Class: `BaseLightningModule`

Extends `lightning.pytorch.LightningModule`.

### Constructor

```python
BaseLightningModule(verbose: bool = False)
```

`verbose` gates whether subclasses additionally log a per-loss-term breakdown (`train_step_loss_0`, `train_step_loss_1`, ...) alongside the total; it has no effect inside `BaseLightningModule` itself. The constructor sets up:

- `self._optimizer = None`, `self._scheduler = None` — filled in later via `register_optimizer`/`register_scheduler`, not built here (see "Optimizer / scheduler / warmup" below).
- `self.warmup_step = 1` — the default makes warmup a no-op (see `optimizer_step`); the runner overrides it via `register_warmup_step`.
- `self.puresound_logging = Logging()` — the per-epoch metric accumulator ([`system.logger.Logging`](logger.md)).

### Abstract hooks (must be implemented by subclasses)

| Method | Raises |
|--------|--------|
| `forward(...)` | `NotImplementedError` |
| `training_step(batch, batch_idx)` | `NotImplementedError` |
| `validation_step(batch, batch_idx)` | `NotImplementedError` |
| `test_step(batch, batch_idx)` | `NotImplementedError` |
| `predict_step(batch, batch_idx, dataloader_idx=None)` | `NotImplementedError` |

[`siso.EncDecMaskBase`/`EncPredClassBase`](siso.md) and [`miso.EncDecCondMaskBase`](miso.md) implement all five.

### Loss registry

```python
register_loss_func(loss_func_list: nn.ModuleList, loss_func_list_weights: List)
reduce_losses(invoke, loss_funcs=None, weights=None) -> (total, per_loss_values)
```

`register_loss_func` stores `self.loss_func_list` / `self.loss_func_list_w`. The list is an `nn.ModuleList` (not a plain Python list) on purpose: assigning it registers each loss module as a submodule, so a loss with trainable parameters of its own (e.g. a margin-based classification loss with learned class centers) has them show up in `.parameters()`/`state_dict()` like any other submodule — which is why `EncPredClassBase.get_total_param_groups()` ([siso.md](siso.md)) can add an optimizer group per loss term. `loss_func_list_w` is a plain list and stays mutable: the curriculum callback (`system.curriculum.CurriculumCallback`, see [index](index.md)) rewrites entries of it at the start of each epoch.

`reduce_losses(invoke, ...)` is the weighted sum every module uses. `invoke(loss_func)` returns that loss's unweighted tensor; it is the only part that differs between modules (what each can provide, and so which losses it can host), so it is the only part left to the caller — each module's `invoke` delegates to `invoke_loss` with its own provider table. `loss_funcs` / `weights` default to the registered pair; pass them explicitly for a secondary set (MISO's conditional-branch losses). It returns `(total, values)`, where `values` holds each weighted term's `.item()`, and it sums out of place so the first weighted tensor is never mutated while it is part of the graph. `total` is `None` for an empty loss list.

`EncDecCondMaskBase` ([miso.md](miso.md)) overrides `register_loss_func` to additionally accept an optional second `(c_loss_func_list, c_loss_func_list_weights)` pair for its conditioning-embedding loss.

### Optimizer / scheduler / warmup

`BaseLightningModule` does not construct an optimizer — it only stores whatever is handed to it and exposes the storage to Lightning:

```python
register_optimizer(optimizer: Any)
register_scheduler(scheduler: Any)
register_warmup_step(warmup_step: int)
configure_optimizers()
```

`configure_optimizers()` (the Lightning hook fired once at trainer setup) returns `[self._optimizer], [self._scheduler]` if a scheduler was registered, otherwise the bare optimizer. Construction happens outside the module, in [`system.optim.create_optimizer_and_scheduler`](optim.md), fed by the subclass's own `get_total_param_groups()`. The wiring in `puresound.system.runner.run_training`:

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

Warmup is a manual linear ramp implemented in `optimizer_step`, layered on top of whatever the registered scheduler is doing — it is not a scheduler class itself:

```python
optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
```

On the first global step it snapshots every param group's current LR into `self.pg_lr`; a run resumed inside the warm-up never sees that step and takes each group's `initial_lr` (the unscaled rate the scheduler recorded) instead. While `trainer.global_step < self.warmup_step`, every group's `lr` is set to `min(1.0, (global_step + 1) / warmup_step) * pg_lr[group]`; from `warmup_step` onward the groups are left alone, so the scheduler takes over from there. It then steps the optimizer with the closure and calls `zero_grad(set_to_none=True)`. With `warmup_step = 1` the ramp condition is false from the first step on, i.e. warmup is off. A recipe sets its warmup length as `scheduler.warmup_step` (a required field of `SchedulerConfig`); the runner reads it and calls `register_warmup_step` — `create_optimizer_and_scheduler` never reads it ([optim.md](optim.md)).

### GPU-batched VAD labeling

A batched, GPU-resident alternative to labeling voice activity per item inside `DataLoader` workers. The dataset emits only the clean reference waveform under `vad_reference` (and, for background-speaker supervision, `background_vad_reference`); the Silero forward pass and its post-processing run once for the whole batch, on the training device, inside the Lightning module — not in the dataset/collate path.

```python
register_gpu_vad_labeler(labeler: Any)
ensure_vad_targets(batch: Any)
on_after_batch_transfer(batch: Any, dataloader_idx: int)
```

- **`register_gpu_vad_labeler(labeler)`** stores `self._gpu_vad_labeler = [labeler]`. The list wrapper is deliberate: it keeps `nn.Module.__setattr__` from registering the labeler as a submodule, so a TorchScript Silero model never lands in `state_dict()`/checkpoints — it is an auxiliary labeler, not trained weights.
- **`ensure_vad_targets(batch)`** is the idempotent conversion: a no-op if no labeler was registered or `batch` is not a `dict`. Otherwise it picks a sample rate (the first element of `batch["sr"]` if present and non-empty, else the labeler's own `model_sample_rate`), then for each of `vad_reference -> vad_target` and `background_vad_reference -> background_vad_target`, pops the reference key and runs the labeler on it — only if the target key is not already present, so calling it more than once on the same batch is safe.
- **`on_after_batch_transfer(batch, dataloader_idx)`** is the Lightning hook (fires right after the batch lands on the training device, before `training_step`/`validation_step`); it is just `return self.ensure_vad_targets(batch)`.

Because that hook's timing varies a little across Lightning strategies, `EncDecMaskBase`'s `training_step`/`validation_step`/`test_step`/`predict_step` ([siso.md](siso.md)) also call `self.ensure_vad_targets(batch)` at the top. That also makes calling those step methods directly in a unit test, bypassing the Lightning hook, safe.

The runner wires it up only when a recipe asks for the expensive backend (`puresound.system.runner.run_training`):

```python
vad_label = recipe.vad_label
if vad_label is not None and vad_label.used and vad_label.backend == "silero":
    from puresound.audio.vad import BatchedSileroVADLabeler

    lightning_model.register_gpu_vad_labeler(
        BatchedSileroVADLabeler(**vad_label.args)
    )
```

The cheap backend (`vad_label.backend: energy`) never touches this path — that labeling happens inside the dataset. The VAD-gate-only training mode (`EncDecMaskBase(train_vad_head_only=True)`, [siso.md](siso.md)) trains against these targets, so the machinery stays in place even for recipes that do not use it.

### Checkpoint loading

```python
reload_checkpoint(loaded_state: Dict, load_loss_func: bool = True)
```

A tolerant alternative to `nn.Module.load_state_dict`, for loading a finished checkpoint to score or run inference with: `runner.run_scoring` / `runner.run_inference` build a fresh model and pass it `torch.load(ckpt)["state_dict"]`, with no optimizer, scheduler or training involved. For every `(name, param)` in `loaded_state`:

- a `name` that is not a key in the live model's `state_dict()` is skipped with a warning (`"%s is not in the model."`) — e.g. a checkpoint saved from a superset architecture;
- if `load_loss_func=False`, any `name` containing `"loss_func_list"` is skipped (logged) — loading a checkpoint trained with a different loss setup without dragging its loss-module parameters along (the speaker-embedding recipe loads this way);
- otherwise the checkpoint tensor is copied into the live parameter in place (`self_state[name].copy_(param)`).

The loss-skip branch and the copy branch both mark the live name as accounted for. At the end it logs `"Loaded params is ok."` only if every live parameter/buffer name was accounted for; otherwise it warns with the list of live names `loaded_state` never mentioned. A consequence: a `load_loss_func=False` load can still report "ok" even though the skipped loss-module parameters stay at their fresh-init values — the summary tracks "was this name present in the checkpoint", not "was this tensor's value updated".

Warm-starting a *training* run is a different path: `--pretrained_ckpt_path` goes through `runner.load_warm_start`, which is non-strict on names (new parameters stay at init, extra checkpoint parameters are ignored, both reported) but strict on shapes unless `--pretrained_allow_reshaped` is given; a new optimizer is built right after.

### Other registration helpers

```python
register_metrics_func(metrics: Any)
register_proc_output_folder(fpath: str)
```

`register_metrics_func` stores a `{name: {"func": callable, "sr": int | None}}` dict that `test_step` iterates (each metric can demand its own sample rate; see [siso.md](siso.md)). The runner's default table is `runner.WAVEFORM_METRICS`. `register_proc_output_folder` stores the directory `predict_step` writes enhanced wavs (and, for embedding models, `.txt` embeddings) into, as `self.eval_output_folder_path`.

### Epoch-end logging

```python
on_train_epoch_end()
on_test_epoch_end()
```

`on_train_epoch_end` logs `epoch_train_loss` as the mean of everything `training_step` pushed into `self.puresound_logging` under that key this epoch (`prog_bar=True, sync_dist=False`), then clears that key. `on_test_epoch_end` never calls `self.log`: for every key accumulated by `test_step`, it prints `key, value.item()` and clears that key. The scores are printed rather than logged because they are the result a `--scoring` run exists to produce, and silencing the logger must not silence them.

## Example

```python
import torch
from puresound.config.recipe import OptimizerConfig, SchedulerConfig
from puresound.system.base import BaseLightningModule
from puresound.system.optim import create_optimizer_and_scheduler


class MyModel(BaseLightningModule):
    """Minimal illustration -- real systems are siso.EncDecMaskBase etc."""

    def __init__(self, encoder, backbone, verbose=False):
        super().__init__(verbose=verbose)
        self.encoder = encoder
        self.backbone = backbone

    def forward(self, wav):
        return self.backbone(self.encoder(wav))

    def training_step(self, batch, batch_idx):
        batch = self.ensure_vad_targets(batch)
        enhanced = self.forward(batch["noisy_speech"])
        target = batch["clean_speech"]
        loss, _ = self.reduce_losses(lambda loss_func: loss_func(enhanced, target))
        self.puresound_logging.update({"epoch_train_loss": loss.item()})
        self.log("train_step_loss", loss, prog_bar=True, sync_dist=False)
        return {"loss": loss}

    def get_total_param_groups(self):
        """Not part of BaseLightningModule -- each concrete subclass defines
        its own, consumed by system.optim.create_optimizer_and_scheduler."""
        return {
            "encoder": {"params": self.encoder.parameters(), "lr_factor": 0.1},
            "backbone": {"params": self.backbone.parameters(), "lr_factor": 1.0},
        }


model = MyModel(encoder, backbone)
model.register_loss_func(torch.nn.ModuleList([torch.nn.L1Loss()]), [1.0])

optimizer, scheduler = create_optimizer_and_scheduler(
    overall_params_and_lr_factor=model.get_total_param_groups(),
    optimizer_args=OptimizerConfig(type="AdamW", learning_rate=1e-3),
    scheduler_args=SchedulerConfig(
        type="StepLR", warmup_step=250, args={"step_size": 10, "gamma": 0.5}
    ),
)
model.register_optimizer(optimizer)
model.register_scheduler(scheduler)
model.register_warmup_step(250)
```
