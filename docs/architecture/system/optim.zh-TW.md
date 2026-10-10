# puresound.system.optim

English version: [`optim.md`](optim.md)

以 reflection 為基礎的 optimizer/LR-scheduler 工廠。這裡沒有一份限制名稱的清單：`type` 字串會直接透過 `getattr` 到 `torch.optim` / `torch.optim.lr_scheduler` 上查找，所以這兩個模組匯出的任何東西都能用，換 optimizer 或 scheduler 只是改設定檔，不需要改程式碼。Config model 在 recipe 載入時檢查區塊的形狀；名稱本身要到這個函式執行時才解析，所以認不得的名稱會在這裡以 `AttributeError` 失敗。

## Function: `create_optimizer_and_scheduler`

```python
create_optimizer_and_scheduler(
    overall_params_and_lr_factor: Dict,
    optimizer_args: OptimizerConfig,
    scheduler_args: SchedulerConfig,
) -> Tuple[Optimizer, LRScheduler]
```

`OptimizerConfig` 與 `SchedulerConfig` 是 `puresound.config.recipe` 裡有型別的 recipe 區塊（嚴格的 Pydantic model：未知 key 會被拒絕、欄位以屬性讀取，所以不接受普通 dict）。

### Parameters

- **`overall_params_and_lr_factor`** — `{group_name: {"params": Iterable[nn.Parameter], "lr_factor": float}}`。不是由這個函式建構的：它由 Lightning module 自己的 `get_total_param_groups()`（`siso.EncDecMaskBase`、`siso.EncPredClassBase`、`miso.EncDecCondMaskBase`——見 [siso.zh-TW.md](siso.zh-TW.md)/[miso.zh-TW.md](miso.zh-TW.md)）產生，每個具名的參數群組一筆（例如 `encoder`/`feats`/`backbone`，開啟 MetricGAN 時多一個 `metric_disc`，或是 `EncDecMaskBase` 的 VAD-gate-only 模式下單一的 `gate_head` 群組）。
- **`optimizer_args`** — `OptimizerConfig(type: str, learning_rate: float > 0, args: dict = {})`。
  - `type` — `torch.optim` 底下的類別名稱（`Adam`、`AdamW`、`SGD`、`RAdam`、...），透過 `getattr(torch.optim, type)` 解析。
  - `learning_rate` — 基礎 LR；每個群組的實際學習率是 `lr_factor * learning_rate`。它也會以位置參數傳給 optimizer 建構子自己的 `lr` 參數。每個群組都已帶有明確的 `"lr"`，所以這個值不會設定任何群組的學習率，但某些 optimizer 類別無論如何都要求這個引數。
  - `args` — 以 `**kwargs` 轉送給 optimizer 類別（`weight_decay`、`betas`、`momentum`、...）。
- **`scheduler_args`** — `SchedulerConfig(type: str, warmup_step: int, args: dict = {})`。
  - `type` — `torch.optim.lr_scheduler` 底下的類別名稱（`StepLR`、`CosineAnnealingWarmRestarts`、...），透過 `getattr(torch.optim.lr_scheduler, type)` 解析。
  - `args` — 以 `**kwargs` 轉送給 scheduler 類別。
  - `warmup_step` — config model 要求必填，但這個函式從不讀取。Runner 把它交給 [`BaseLightningModule.register_warmup_step()`](base.zh-TW.md)，由它在 `optimizer_step` 裡驅動手刻的線性 warmup 爬升——這套機制在這個模組之外。

### Returns

`(optimizer, scheduler)`——永遠是一個二元組。這裡沒有「不給 scheduler」的路徑：`SchedulerConfig.type` 是必填欄位，所以一定會建構並回傳一個 scheduler。

### Recipe 範例

`egs/voice_isolate/config/train_dpcrn.yaml`：

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

`egs/noise_suppression/config/dpcrn.yaml` 則用 `StepLR`：

```yaml
scheduler:
  type: StepLR
  warmup_step: 1
  args: {step_size: 10, gamma: 0.5}
```

兩者都走同一條 reflection 路徑。

### 接法（`puresound.system.runner.run_training`）

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
