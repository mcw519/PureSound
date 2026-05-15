# puresound.system.logger

Training metric accumulation and averaging utility.

## Class: `Logging`

Accumulates scalar and tensor metric values across batches and computes their averages, useful for per-epoch metric reporting in training loops.

### Constructor

```python
Logging()
```

Initializes with empty internal metric storage.

### Methods

#### `update(outs: Dict)`

Accumulates values from a dictionary of metrics.

**Parameters:**
- `outs` – Dictionary mapping metric names to scalar floats or `torch.Tensor` values

**Notes:**
- Tensor values are automatically detached and moved to CPU before accumulation
- Accumulates lists internally for later averaging

**Example:**
```python
logger.update({"loss": loss_val, "sisnr": sisnr_val})
```

---

#### `average(key: Optional[str] = None) -> Union[Dict, float]`

Computes the average of accumulated values.

**Parameters:**
- `key` – If provided, returns the average for that specific metric only
- If `None`, returns a dict of averages for all accumulated metrics

**Returns:**
- `float` – If `key` is specified
- `Dict[str, float]` – If `key` is `None`

---

#### `clear(key: Optional[str] = None)`

Clears accumulated values.

**Parameters:**
- `key` – If provided, clears only that specific metric
- If `None`, clears all accumulated metrics

## Example

```python
from puresound.system.logger import Logging

logger = Logging()

for batch in dataloader:
    loss = compute_loss(batch)
    logger.update({"train/loss": loss.item(), "train/sisnr": sisnr.item()})

epoch_metrics = logger.average()
print(f"Epoch loss: {epoch_metrics['train/loss']:.4f}")
logger.clear()
```
