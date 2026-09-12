# puresound.system.logger

繁體中文版本：[`logger.zh-TW.md`](logger.zh-TW.md)

Training metric accumulation and averaging utility.

## Class: `Logging`

Accumulates scalar and tensor metric values across batches and computes their averages, useful for per-epoch metric reporting in training loops.

### Constructor

```python
Logging()
```

Initializes with empty internal metric storage (`self.bag = {}`).

### Methods

#### `update(outs: Dict)`

Accumulates values from a dictionary of metrics.

**Parameters:**
- `outs` – Dictionary mapping metric names to scalar floats or `torch.Tensor` values

**Notes:**
- `torch.Tensor` values are converted to a plain Python float via `.item()` before accumulation (the tensor must hold a single element); non-tensor values are stored as-is.
- Accumulates into a `List` per key internally, for later averaging.

**Example:**
```python
logger.update({"loss": loss_val, "sisnr": sisnr_val})
```

---

#### `average(key: Optional[str] = None)`

Computes the average of accumulated values.

**Parameters:**
- `key` – If provided, returns the average for that specific metric only
- If `None`, returns a dict of averages for all accumulated metrics

**Returns:**
- `torch.Tensor` (0-dim) – if `key` is specified: `torch.Tensor(self.bag[key]).mean()`
- `Dict[str, torch.Tensor]` – if `key` is `None`: one 0-dim tensor per accumulated key

This always returns `torch.Tensor` values, **never** a native Python `float` — call `.item()` on the result (or on each dict value) if a plain float is needed.

---

#### `clear(key: Optional[str] = None)`

Clears accumulated values.

**Parameters:**
- `key` – If provided, deletes only that key's accumulated list (raises `KeyError` if `key` was never accumulated via `update`)
- If `None`, resets the whole bag (`self.bag = {}`)

## Example

```python
from puresound.system.logger import Logging

logger = Logging()

for batch in dataloader:
    loss = compute_loss(batch)
    logger.update({"train/loss": loss.item(), "train/sisnr": sisnr.item()})

epoch_metrics = logger.average()
print(f"Epoch loss: {epoch_metrics['train/loss'].item():.4f}")
logger.clear()
```
