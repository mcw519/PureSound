# puresound.system.base

Base PyTorch Lightning module providing the shared training infrastructure for all PureSound models.

## Class: `BaseLightningModule`

Extends `pytorch_lightning.LightningModule`. Abstract base class that all PureSound training systems inherit from.

### Responsibilities

- Manages loss functions and their weights
- Configures optimizers and schedulers
- Provides structured logging utilities
- Implements learning rate warmup
- Defines abstract hooks for training and validation

### Constructor

```python
BaseLightningModule(hparam: Dict)
```

**Parameters:**
- `hparam` – Hyperparameter dictionary loaded from YAML config

### Methods

#### `register_loss_func(loss_fns: List[Tuple[nn.Module, float]])`

Registers a list of `(loss_function, weight)` tuples for multi-objective training.

---

#### `configure_optimizers() -> Dict`

Returns the optimizer and optionally a scheduler dict, as required by PyTorch Lightning.

---

#### `log_metrics(metrics: Dict, prefix: str = "")`

Logs a dictionary of scalar metrics to the Lightning logger (TensorBoard, W&B, etc.).

---

#### `warmup_lr(step: int)`

Applies linear learning rate warmup for the first N steps. Scales per-parameter-group learning rates independently.

**Parameters:**
- `step` – Current training step

---

#### `on_train_epoch_end()`

Hook called at the end of each training epoch. Logs epoch-level averaged metrics.

---

#### `on_test_epoch_end()`

Hook called at the end of testing. Logs test-set averaged metrics.

### Abstract Methods (must be implemented by subclasses)

| Method | Description |
|--------|-------------|
| `training_step(batch, batch_idx)` | Define forward pass and loss computation for a training batch |
| `validation_step(batch, batch_idx)` | Define forward pass for a validation batch |
| `test_step(batch, batch_idx)` | Define forward pass for a test batch |

## Example

```python
class MyModel(BaseLightningModule):
    def training_step(self, batch, batch_idx):
        noisy, clean = batch["noisy_speech"], batch["clean_speech"]
        enhanced = self.forward(noisy)
        loss = sum(w * fn(enhanced, clean) for fn, w in self.loss_fns)
        self.log("train/loss", loss)
        return loss
```
