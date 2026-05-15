# puresound.system.optim

Optimizer and learning rate scheduler factory for PureSound training systems.

## Functions

### `create_optimizer_and_scheduler(model: nn.Module, hparam: Dict) -> Tuple`

Constructs a PyTorch optimizer (with optional per-component learning rate scaling) and an optional learning rate scheduler.

**Parameters:**
- `model` – The PyTorch model to optimize
- `hparam` – Hyperparameter config dict with an `"optim"` section

**Returns:** `(optimizer, scheduler_dict or None)`

### Config Structure

```yaml
optim:
  optimizer: adam         # adam | sgd | adamw
  lr: 0.001               # Base learning rate
  weight_decay: 0.0
  
  # Per-component LR scaling (optional)
  lr_factor:
    encoder: 0.1          # encoder gets lr * 0.1
    cond_backbone: 0.5    # cond_backbone gets lr * 0.5

  scheduler:
    type: reduce_lr       # reduce_lr | warmup_cosine | step
    patience: 3           # for reduce_lr
    factor: 0.5           # for reduce_lr / step
    warmup_steps: 4000    # for warmup_cosine
```

### Supported Optimizers

| Name | PyTorch Class |
|------|--------------|
| `"adam"` | `torch.optim.Adam` |
| `"adamw"` | `torch.optim.AdamW` |
| `"sgd"` | `torch.optim.SGD` |

### Supported Schedulers

| Type | Description |
|------|-------------|
| `"reduce_lr"` | `ReduceLROnPlateau` — reduces LR when metric plateaus |
| `"warmup_cosine"` | Linear warmup + cosine annealing |
| `"step"` | `StepLR` — reduces LR by `factor` every N epochs |

### Per-Component LR Scaling

If `lr_factor` is specified in the config, model parameters are grouped by component name, with each group receiving a scaled learning rate:

```
effective_lr[component] = base_lr * lr_factor[component]
```

Parameters not matching any component key use the base `lr`.

## Example

```python
from puresound.system.optim import create_optimizer_and_scheduler

optimizer, scheduler = create_optimizer_and_scheduler(model, hparam)

# In PyTorch Lightning:
def configure_optimizers(self):
    opt, sch = create_optimizer_and_scheduler(self, self.hparam)
    if sch:
        return {"optimizer": opt, "lr_scheduler": sch}
    return opt
```
