# puresound.system.base

繁體中文版本：[`base.zh-TW.md`](base.zh-TW.md)

Shared Lightning-module infrastructure that every PureSound training system (`EncDecMaskBase`, `EncDecCondMaskBase`, `EncPredClassBase`) inherits from: loss registry, optimizer/scheduler plumbing + LR warmup, structured epoch logging, GPU-batched VAD labeling, and a tolerant checkpoint loader. `BaseLightningModule` is itself abstract — it defines no model and cannot train on its own.

## Class: `BaseLightningModule`

Extends `lightning.pytorch.LightningModule`.

### Constructor

```python
BaseLightningModule(verbose: bool = False)
```

`verbose` gates whether subclasses additionally log a per-loss-term breakdown (`train_step_loss_0`, `train_step_loss_1`, ...) alongside the total; it has no effect inside `BaseLightningModule` itself. The constructor sets up:

- `self._optimizer = None`, `self._scheduler = None` — filled in later via `register_optimizer`/`register_scheduler`, not built here (see "Optimizer / scheduler / warmup" below).
- `self.warmup_step = 1` — the default makes warmup a no-op (see `optimizer_step`); real recipes override it via `register_warmup_step`.
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
```

Stores `self.loss_func_list` / `self.loss_func_list_w`. The list is an `nn.ModuleList` (not a plain Python list) on purpose: assigning it registers each loss module as a real submodule, so a loss with its own trainable parameters (e.g. a margin-based classification loss with learned class centers) has those parameters show up in `.parameters()`/`state_dict()` like any other submodule — which is exactly why `EncPredClassBase.get_total_param_groups()` (siso.md) can add an optimizer group per loss term.

A subclass's `compute_loss` iterates the two lists in lockstep and dispatches each term by attribute flags the loss module sets on itself (`uses_vad_target`, `uses_batch`, ...) — see siso.md for the concrete dispatch table. `EncDecCondMaskBase` (miso.md) overrides `register_loss_func` to additionally accept an optional second `(c_loss_func_list, c_loss_func_list_weights)` pair for its conditioning-embedding loss; that override, not this base version, is what MISO recipes call.

### Optimizer / scheduler / warmup

`BaseLightningModule` does not construct an optimizer — it only stores whatever is handed to it and exposes the storage to Lightning:

```python
register_optimizer(optimizer: Any)
register_scheduler(scheduler: Any)
register_warmup_step(warmup_step: int)
configure_optimizers()
```

`configure_optimizers()` (the Lightning hook fired once at trainer setup) returns `[self._optimizer], [self._scheduler]` if a scheduler was registered, otherwise the bare optimizer. The actual construction happens outside the module, in [`system.optim.create_optimizer_and_scheduler`](optim.md), fed by the subclass's own `get_total_param_groups()`. Canonical wiring (`egs/noise_suppression/main.py`):

```python
param_groups = lightning_model.get_total_param_groups()
optimizer, scheduler = create_optimizer_and_scheduler(
    overall_params_and_lr_factor=param_groups,
    optimizer_args=optim_dict,
    scheduler_args=scheduler_dict,
)
lightning_model.register_optimizer(optimizer)
lightning_model.register_scheduler(scheduler)
lightning_model.register_warmup_step(scheduler_dict["warmup_step"])
```

Warmup is a manual linear ramp implemented in `optimizer_step`, layered on top of whatever the registered scheduler is already doing — it is not a scheduler class itself:

```python
optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
```

On the very first global step it snapshots every param group's current LR into `self.pg_lr`. While `trainer.global_step < self.warmup_step`, every group's `lr` is rescaled to `min(1.0, (global_step + 1) / warmup_step) * pg_lr[group]`; from `warmup_step` onward the groups are left alone (so whatever the scheduler already did to them in between takes over from there). With the default `warmup_step = 1` the ramp condition is false from the first step on, i.e. warmup is effectively off — recipes that want it call `register_warmup_step` explicitly (real example: `scheduler.warmup_step: 250` in `egs/voice_isolate/config/train_dpcrn.yaml`, read by `main.py` and passed to `register_warmup_step` — **not** consumed by `create_optimizer_and_scheduler` itself; see optim.md).

### GPU-batched VAD labeling

A batched, GPU-resident alternative to labeling voice activity per-item inside `DataLoader` workers. The dataset only emits the clean reference waveform under `vad_reference` (and, for background-speaker supervision, `background_vad_reference`); the Silero forward pass and its post-processing run once for the whole batch, on the training device, inside the Lightning module — not in the dataset/collate path.

```python
register_gpu_vad_labeler(labeler: Any)
ensure_vad_targets(batch: Any)
on_after_batch_transfer(batch: Any, dataloader_idx: int)
```

- **`register_gpu_vad_labeler(labeler)`** stores `self._gpu_vad_labeler = [labeler]`. The list wrapper is deliberate: it keeps `nn.Module.__setattr__` from registering the labeler as a submodule, so a TorchScript Silero model never lands in `state_dict()`/checkpoints — it's an auxiliary labeler, not trained weights.
- **`ensure_vad_targets(batch)`** is the idempotent conversion: a no-op if no labeler was registered or `batch` isn't a `dict`. Otherwise it picks a sample rate (`batch["sr"][0]` if present, else the labeler's own `model_sample_rate`), then for each of `vad_reference -> vad_target` and `background_vad_reference -> background_vad_target`, pops the reference key and runs the labeler on it — but only if the corresponding target key isn't already present, so calling it more than once on the same batch is safe.
- **`on_after_batch_transfer(batch, dataloader_idx)`** is the actual Lightning hook (fires right after the batch lands on the training device, before `training_step`/`validation_step`); it is just `return self.ensure_vad_targets(batch)`.

Because that hook's exact timing varies a little across Lightning strategies, `EncDecMaskBase`'s own `training_step`/`validation_step`/`test_step`/`predict_step` (siso.md) also call `self.ensure_vad_targets(batch)` defensively at the top — belt-and-suspenders, and it is what makes calling those step methods directly in a unit test (bypassing the Lightning hook entirely) safe too.

Wired up only when a recipe asks for the expensive backend (`egs/noise_suppression/main.py`):

```python
if vad_label_dict.get("backend", "energy").lower() == "silero":
    from puresound.audio.vad import BatchedSileroVADLabeler
    lightning_model.register_gpu_vad_labeler(
        BatchedSileroVADLabeler(**vad_label_dict.get("args", {}))
    )
```

The cheap default (`vad_label.backend: energy`) never touches this path — that labeling happens synchronously inside the dataset/collate instead. This machinery is what the VAD-gate-only training mode (`EncDecMaskBase(train_vad_head_only=True)`, siso.md) trains against, and it stays live infrastructure for that reuse even on recipes that otherwise have no use for it.

### Checkpoint loading

```python
reload_checkpoint(loaded_state: Dict, load_loss_func: bool = True)
```

A manual, tolerant alternative to Lightning/`nn.Module`'s own `load_state_dict`, used by the `--scoring`/`--inference` stages of `egs/*/main.py` (a freshly constructed model plus a raw `torch.load(ckpt)["state_dict"]`, no optimizer/scheduler/trainer involved). For every `(name, param)` in `loaded_state`:

- a `name` that isn't a key in the live model's `state_dict()` is skipped with a printed notice (`"{name} is not in the model."`) — e.g. a checkpoint saved from a superset architecture;
- if `load_loss_func=False`, any `name` containing `"loss_func_list"` is also skipped (its own printed notice) — warm-starting from a checkpoint trained with a different/incompatible loss setup without dragging its loss-module parameters along;
- otherwise the checkpoint tensor is copied into the live parameter in place (`self_state[name].copy_(param)`).

Both skip branches *and* the copy branch mark the name as "accounted for." At the end it prints `"Loaded params is ok."` only once every live-model parameter/buffer name has been accounted for one way or another; otherwise it prints the list of live names `loaded_state` never mentioned at all. One consequence worth knowing: a `load_loss_func=False` load can still print "ok" even though the skipped loss-module parameters were never actually copied and are left at their fresh-init values — the summary tracks "was this name present in the checkpoint," not "was this tensor's value updated."

This is a different path from the `--pretrained_ckpt_path` warm-start in `main.py`, which instead calls Lightning's own `load_state_dict(state_dict, strict=False)` directly (and reports `missing`/`unexpected` keys) — that one is for warm-starting a fresh *training* run (a new optimizer gets built right after); `reload_checkpoint` is for loading a finished checkpoint to score or run inference with, no optimizer involved at all.

### Other registration helpers

```python
register_metrics_func(metrics: Any)
register_proc_output_folder(fpath: str)
```

`register_metrics_func` stores a `{name: {"func": callable, "sr": int | None}}` dict that `test_step` iterates (each metric can demand its own sample rate; see siso.md/miso.md). `register_proc_output_folder` stores the directory `predict_step` writes enhanced wavs (and, for MISO/embedding models, `.txt` embeddings) into.

### Epoch-end logging

```python
on_train_epoch_end()
on_test_epoch_end()
```

`on_train_epoch_end` logs `epoch_train_loss` as the mean of everything `training_step` pushed into `self.puresound_logging` under that key this epoch (`prog_bar=True, sync_dist=False`), then clears that key. `on_test_epoch_end` never calls `self.log`: for every key accumulated by `test_step` via `puresound_logging.update(...)`, it prints `key, value.item()` and clears that key — test/eval metrics surface as stdout, not through the Lightning logger integration (TensorBoard/W&B) that training/validation use.

## Example

```python
import torch
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
        loss = sum(
            w * fn(enhanced, target)
            for fn, w in zip(self.loss_func_list, self.loss_func_list_w)
        )
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
    optimizer_args={"type": "AdamW", "learning_rate": 1e-3, "args": {}},
    scheduler_args={"type": "StepLR", "args": {"step_size": 10, "gamma": 0.5}},
)
model.register_optimizer(optimizer)
model.register_scheduler(scheduler)
model.register_warmup_step(250)
```
