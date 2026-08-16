# puresound.system.base

English version: [`base.md`](base.md)

所有 PureSound 訓練系統（`EncDecMaskBase`、`EncDecCondMaskBase`、`EncPredClassBase`）共用的 Lightning module 基礎設施：loss registry、optimizer/scheduler 管線 + LR warmup、結構化的 epoch logging、GPU-batched VAD labeling，以及一個寬容的 checkpoint loader。`BaseLightningModule` 本身是抽象的——它沒有定義任何模型，也無法單獨訓練。

## Class: `BaseLightningModule`

繼承自 `lightning.pytorch.LightningModule`。

### Constructor

```python
BaseLightningModule(verbose: bool = False)
```

`verbose` 只控制子類別是否額外記錄每個 loss 項的明細（`train_step_loss_0`、`train_step_loss_1`、...），而不只是總和；對 `BaseLightningModule` 本身沒有作用。建構子會設定：

- `self._optimizer = None`、`self._scheduler = None`——之後透過 `register_optimizer`/`register_scheduler` 填入，並非在這裡建構（見下方「Optimizer / scheduler / warmup」）。
- `self.warmup_step = 1`——預設值讓 warmup 變成 no-op（見 `optimizer_step`）；實際 recipe 會透過 `register_warmup_step` 覆寫它。
- `self.puresound_logging = Logging()`——per-epoch 的 metric 累積器（[`system.logger.Logging`](logger.zh-TW.md)）。

### Abstract hooks（子類別必須實作）

| Method | Raises |
|--------|--------|
| `forward(...)` | `NotImplementedError` |
| `training_step(batch, batch_idx)` | `NotImplementedError` |
| `validation_step(batch, batch_idx)` | `NotImplementedError` |
| `test_step(batch, batch_idx)` | `NotImplementedError` |
| `predict_step(batch, batch_idx, dataloader_idx=None)` | `NotImplementedError` |

[`siso.EncDecMaskBase`/`EncPredClassBase`](siso.zh-TW.md) 與 [`miso.EncDecCondMaskBase`](miso.zh-TW.md) 實作了全部五個。

### Loss registry

```python
register_loss_func(loss_func_list: nn.ModuleList, loss_func_list_weights: List)
```

儲存 `self.loss_func_list` / `self.loss_func_list_w`。這裡刻意用 `nn.ModuleList`（而非普通的 Python list）：指定它會把每個 loss module 註冊成真正的 submodule，因此帶有自己可訓練參數的 loss（例如帶有可學習類別中心的 margin-based 分類 loss）其參數會出現在 `.parameters()`/`state_dict()` 裡，就像任何其他 submodule 一樣——這正是為什麼 `EncPredClassBase.get_total_param_groups()`（見 siso.zh-TW.md）能為每個 loss 項多加一個 optimizer group 的原因。

子類別的 `compute_loss` 會同步走訪這兩個 list，並依 loss module 自己設定的屬性旗標（`uses_vad_target`、`uses_batch`、...）分派每一項——詳細的分派表見 siso.zh-TW.md。`EncDecCondMaskBase`（miso.zh-TW.md）覆寫了 `register_loss_func`，多接受一組可選的 `(c_loss_func_list, c_loss_func_list_weights)`，用於它的 conditioning-embedding loss；MISO recipe 呼叫的是那個覆寫版本，不是這個 base 版本。

### Optimizer / scheduler / warmup

`BaseLightningModule` 不會建構 optimizer——它只儲存外部交給它的東西，並把這份儲存暴露給 Lightning：

```python
register_optimizer(optimizer: Any)
register_scheduler(scheduler: Any)
register_warmup_step(warmup_step: int)
configure_optimizers()
```

`configure_optimizers()`（在 trainer 設定時觸發一次的 Lightning hook）在已註冊 scheduler 時回傳 `[self._optimizer], [self._scheduler]`，否則回傳裸的 optimizer。實際的建構發生在 module 之外，由 [`system.optim.create_optimizer_and_scheduler`](optim.zh-TW.md) 完成，輸入來自子類別自己的 `get_total_param_groups()`。標準接法（`puresound/system/runner.py`）：

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

Warmup 是在 `optimizer_step` 裡手刻的線性爬升，疊加在已註冊 scheduler 原本的行為之上——它本身並不是一個 scheduler 類別：

```python
optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
```

在第一個 global step，它會把每個 param group 當下的 LR 快照進 `self.pg_lr`。當 `trainer.global_step < self.warmup_step` 時，每個 group 的 `lr` 會被重新縮放成 `min(1.0, (global_step + 1) / warmup_step) * pg_lr[group]`；超過 `warmup_step` 之後就不再處理這些 group（所以 scheduler 在這段期間對它們做的事，會從那之後接手）。預設 `warmup_step = 1` 時，這個爬升條件從第一步開始就是 false，也就是說 warmup 實質上是關閉的——需要 warmup 的 recipe 會明確呼叫 `register_warmup_step`（實例：`egs/voice_isolate/config/train_dpcrn.yaml` 裡的 `scheduler.warmup_step: 250`，由 `main.py` 讀出並傳給 `register_warmup_step`——**不是**由 `create_optimizer_and_scheduler` 自己消費；見 optim.zh-TW.md）。

### GPU-batched VAD labeling

一種批次化、常駐 GPU 的替代方案，取代在 `DataLoader` worker 裡逐筆做語音活性標記。Dataset 只需要在 `vad_reference`（若要做背景語者的監督，還有 `background_vad_reference`）底下送出乾淨的參考波形；Silero 的 forward pass 與其後處理只會對整個 batch 跑一次，在訓練裝置上、於 Lightning module 內完成——而不是在 dataset/collate 路徑裡。

```python
register_gpu_vad_labeler(labeler: Any)
ensure_vad_targets(batch: Any)
on_after_batch_transfer(batch: Any, dataloader_idx: int)
```

- **`register_gpu_vad_labeler(labeler)`** 會儲存 `self._gpu_vad_labeler = [labeler]`。用 list 包起來是刻意的：這樣可以避免 `nn.Module.__setattr__` 把這個 labeler 註冊成 submodule，讓 TorchScript 版的 Silero 模型永遠不會出現在 `state_dict()`/checkpoint 裡——它是輔助用的 labeler，不是訓練出來的權重。
- **`ensure_vad_targets(batch)`** 是具冪等性（idempotent）的轉換：若沒有註冊 labeler、或 `batch` 不是 `dict`，就直接不做事。否則會先決定 sample rate（若有 `batch["sr"][0]` 就用它，否則用 labeler 自己的 `model_sample_rate`），然後對 `vad_reference -> vad_target` 與 `background_vad_reference -> background_vad_target` 這兩組，各自 pop 出 reference key、用 labeler 跑過——但只有在對應的 target key 還不存在時才會做，所以對同一個 batch 呼叫多次也是安全的。
- **`on_after_batch_transfer(batch, dataloader_idx)`** 是真正的 Lightning hook（在 batch 移到訓練裝置之後、`training_step`/`validation_step` 之前觸發）；內容就是 `return self.ensure_vad_targets(batch)`。

由於這個 hook 的確切觸發時機在不同 Lightning strategy 下略有差異，`EncDecMaskBase` 自己的 `training_step`/`validation_step`/`test_step`/`predict_step`（見 siso.zh-TW.md）也會在開頭主動呼叫 `self.ensure_vad_targets(batch)`——雙重保險，這也讓直接在單元測試裡呼叫這些 step 方法（完全繞過 Lightning hook）變得安全。

只有在 recipe 要求較昂貴的 backend 時才會接上（`puresound/system/runner.py`）：

```python
if vad_label_dict.get("backend", "energy").lower() == "silero":
    from puresound.audio.vad import BatchedSileroVADLabeler
    lightning_model.register_gpu_vad_labeler(
        BatchedSileroVADLabeler(**vad_label_dict.get("args", {}))
    )
```

便宜的預設值（`vad_label.backend: energy`）完全不會碰到這條路徑——那種標記是在 dataset/collate 內同步完成的。這套機制正是 VAD-gate-only 訓練模式（`EncDecMaskBase(train_vad_head_only=True)`，見 siso.zh-TW.md）訓練時所依賴的對象；即使某些 recipe 本身用不到它，這套基礎設施仍會留在現役程式碼中，以供這種重用。

### Checkpoint loading

```python
reload_checkpoint(loaded_state: Dict, load_loss_func: bool = True)
```

一個手刻、寬容版本的 `load_state_dict` 替代方案，用於 `egs/*/main.py` 的 `--scoring`/`--inference` 階段（一個剛建構好的模型，加上一份原始的 `torch.load(ckpt)["state_dict"]`，完全不涉及 optimizer/scheduler/trainer）。對 `loaded_state` 裡的每組 `(name, param)`：

- 若 `name` 不是現有模型 `state_dict()` 裡的 key，會印出提示後跳過（`"{name} is not in the model."`）——例如從一個超集架構存下來的 checkpoint；
- 若 `load_loss_func=False`，任何名稱含有 `"loss_func_list"` 的也會被跳過（同樣會印出提示）——用意是從一個 loss 設定不同、不相容的 checkpoint 做 warm-start，又不想把它的 loss-module 參數一併帶進來；
- 其餘情況則把 checkpoint tensor 原地複製進現有參數（`self_state[name].copy_(param)`）。

無論是這兩種跳過情形，或是複製的情形，都會把該名稱標記為「已處理」。最後，只有當現有模型的每個參數/buffer 名稱都以某種方式被處理過，才會印出 `"Loaded params is ok."`；否則會印出 `loaded_state` 完全沒提到的現有名稱清單。有個值得注意的地方：`load_loss_func=False` 的載入即使印出 "ok"，被跳過的 loss-module 參數其實從未被複製，仍停留在剛初始化的數值——這份總結追蹤的是「這個名稱有沒有出現在 checkpoint 裡」，而不是「這個 tensor 的值有沒有被更新」。

這和 `main.py` 裡 `--pretrained_ckpt_path` 的 warm-start 路徑不同，後者是直接呼叫 Lightning 自己的 `load_state_dict(state_dict, strict=False)`（並回報 `missing`/`unexpected` key）——那條路徑是用來 warm-start 一次全新的*訓練*（之後會立刻建一個新的 optimizer）；`reload_checkpoint` 則是用來載入一個已完成的 checkpoint 做評分或推論，完全不牽涉 optimizer。

### 其他註冊用的輔助方法

```python
register_metrics_func(metrics: Any)
register_proc_output_folder(fpath: str)
```

`register_metrics_func` 儲存一個 `{name: {"func": callable, "sr": int | None}}` 的 dict，供 `test_step` 走訪（每個 metric 可以要求自己的 sample rate；見 siso.zh-TW.md/miso.zh-TW.md）。`register_proc_output_folder` 儲存 `predict_step` 寫出強化後 wav（以及對 MISO/embedding 模型而言的 `.txt` embedding）所用的目錄。

### Epoch-end logging

```python
on_train_epoch_end()
on_test_epoch_end()
```

`on_train_epoch_end` 把這個 epoch 裡 `training_step` 塞進 `self.puresound_logging`（key 為 `epoch_train_loss`）的所有數值取平均後記錄下來（`prog_bar=True, sync_dist=False`），然後清掉這個 key。`on_test_epoch_end` 完全不呼叫 `self.log`：對 `test_step` 透過 `puresound_logging.update(...)` 累積的每個 key，它會印出 `key, value.item()` 然後清掉該 key——test/eval 的 metric 是以 stdout 呈現，而不是透過 training/validation 所使用的 Lightning logger 整合（TensorBoard/W&B）。

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
