# puresound.system.logger

English version: [`logger.md`](logger.md)

訓練 metric 的累積與平均工具。

## Class: `Logging`

跨 batch 累積 scalar 與 tensor 兩種 metric 數值並計算平均，供 per-epoch 的 metric 報告使用。它是 `BaseLightningModule.puresound_logging`（[base.zh-TW.md](base.zh-TW.md)）背後的累積器：`training_step` 餵入 `epoch_train_loss`、`test_step` 餵入評分 metric，epoch 結束的 hook 負責取平均並清除。它不是 Lightning logger，也不做跨 rank 的 reduction。

### Constructor

```python
Logging()
```

初始化時內部 metric 儲存為空（`self.bag = {}`）。

### Methods

#### `update(outs: Dict)`

從一個 dict 累積 metric 數值。

**Parameters:**
- `outs` – 一個把 metric 名稱對應到 scalar float 或 `torch.Tensor` 數值的 dict

**Notes:**
- `torch.Tensor` 數值在累積前會透過 `.item()` 轉成純 Python float（該 tensor 必須只含單一元素）；非 tensor 數值則原樣儲存。
- 內部依 key 各自累積成一個 `List`，供之後平均使用。

**Example:**
```python
logger.update({"loss": loss_val, "sisnr": sisnr_val})
```

---

#### `average(key: Optional[str] = None)`

計算已累積數值的平均。

**Parameters:**
- `key` – 若給定，只回傳該 metric 的平均
- 若為 `None`，回傳所有已累積 metric 的平均 dict

**Returns:**
- `torch.Tensor`（0 維）– 當給定 `key` 時：`torch.Tensor(self.bag[key]).mean()`
- `Dict[str, torch.Tensor]` – 當 `key` 為 `None` 時：每個已累積的 key 各對應一個 0 維 tensor

這裡永遠回傳 `torch.Tensor`，**從不會**是原生 Python `float`——若需要純 float，要自行對回傳值（或 dict 裡每個 value）呼叫 `.item()`。

---

#### `clear(key: Optional[str] = None)`

清除已累積的數值。

**Parameters:**
- `key` – 若給定，只刪除該 key 累積的 list（若該 `key` 從未被 `update` 累積過，會拋出 `KeyError`）
- 若為 `None`，重置整個 bag（`self.bag = {}`）

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
