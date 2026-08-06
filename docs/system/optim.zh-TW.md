# puresound.system.optim

English version: [`optim.md`](optim.md)

以 reflection 為基礎的 optimizer/LR-scheduler 工廠。這裡沒有一份「支援哪些 optimizer/scheduler 名稱」的限制清單：`type` 字串會直接透過 `getattr` 到 `torch.optim` / `torch.optim.lr_scheduler` 上查找，所以這兩個模組底下匯出的任何東西都能用。認不得的名稱會直接以 `AttributeError` 失敗，而不是一個友善的設定檔驗證錯誤。

## Function: `create_optimizer_and_scheduler`

```python
create_optimizer_and_scheduler(
    overall_params_and_lr_factor: Dict,
    optimizer_args: Dict,
    scheduler_args: Dict,
) -> Tuple[Optimizer, LRScheduler]
```

### Parameters

- **`overall_params_and_lr_factor`** — `{group_name: {"params": Iterable[nn.Parameter], "lr_factor": float}}`。不是由這個函式建構的：它是由 Lightning module 自己的 `get_total_param_groups()`（`siso.EncDecMaskBase`、`siso.EncPredClassBase`、`miso.EncDecCondMaskBase`——見 [siso.zh-TW.md](siso.zh-TW.md)/[miso.zh-TW.md](miso.zh-TW.md)）產生，每個具名的參數群組各對應一筆 dict 項目（例如 `encoder`/`feats`/`backbone`，或是 `EncDecMaskBase` 的 VAD-gate-only 模式下單一的 `gate_head` 群組）。
- **`optimizer_args`** — `{"type": str, "learning_rate": float, "args": Dict}`。
  - `type` — `torch.optim` 底下的類別名稱（`Adam`、`AdamW`、`SGD`、`RAdam`、...），透過 `getattr(torch.optim, type)` 解析。
  - `learning_rate` — 基礎 LR；每個群組的實際學習率是 `lr_factor * learning_rate`。它同時也會以位置參數的形式傳給 optimizer 建構子自己的 `lr=` 參數——這裡每個群組都已經帶有明確的 `"lr"`，所以這個位置參數實際上不會拿去設定任何群組的學習率，但某些 optimizer 類別無論每個 param group 是否各自提供 `lr`，都要求一定要有這個引數。
  - `args` — 以 `**kwargs` 的形式轉送給 optimizer 類別（`weight_decay`、`betas`、`momentum`、...）。
- **`scheduler_args`** — `{"type": str, "args": Dict}`。
  - `type` — `torch.optim.lr_scheduler` 底下的類別名稱（`StepLR`、`CosineAnnealingWarmRestarts`、`ReduceLROnPlateau`、...），透過 `getattr(torch.optim.lr_scheduler, type)` 解析。
  - `args` — 以 `**kwargs` 的形式轉送給 scheduler 類別。
  - 實際 recipe 的 YAML `scheduler:` 區塊通常還會帶一個同層的 `warmup_step` key——這個函式從不讀取它。它是由訓練腳本另外抽出來，交給 [`BaseLightningModule.register_warmup_step()`](base.zh-TW.md)，由它在 `optimizer_step` 裡驅動手刻的線性 warmup 爬升——這整套機制完全在這個模組之外。

### Returns

`(optimizer, scheduler)`——永遠是一個二元組。這裡沒有「不給 scheduler」的路徑：`scheduler_args["type"]` 是必要的（沒給會 `KeyError`），所以一定會建構並回傳一個 scheduler。

### 真實設定檔範例

`egs/voice_isolate/config/train_dpcrn.yaml`：

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

目前所有 recipe 用的 optimizer 都是 `AdamW`；scheduler 則依 recipe 而異——`egs/noise_suppression/config/dpcrn.yaml` 用的是 `StepLR`：

```yaml
scheduler:
  type: StepLR
  args: {step_size: 10, gamma: 0.5}
```

兩者都走同一條 reflection 路徑——從一個換到另一個只是改設定檔，不需要改程式碼。

### 真實接法（`egs/noise_suppression/main.py`）

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
