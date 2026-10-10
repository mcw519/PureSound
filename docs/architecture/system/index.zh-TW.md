# puresound.system

English version: [`index.md`](index.md)

PyTorch Lightning 訓練系統、共用的訓練 driver，以及在模型 mask 之後執行、僅用於推論的階段。

## Sub-modules

| Module | Status | Description |
|--------|--------|-------------|
| [system.base](base.zh-TW.md) | active | base Lightning module：loss registry 與以 provider 為基礎的 loss 分派（`invoke_loss`、`reduce_losses`）、optimizer/scheduler 管線、LR warmup、GPU-batched VAD labeling、checkpoint 重新載入 |
| [system.siso](siso.zh-TW.md) | active | 單輸入 enhancement trainer（`EncDecMaskBase`）+ embedding classifier（`EncPredClassBase`，legacy） |
| [system.optim](optim.zh-TW.md) | active | 以有型別的 `optimizer` / `scheduler` recipe 區塊建構 optimizer 與 LR-scheduler 的工廠 |
| [system.logger](logger.zh-TW.md) | active | per-epoch metric 累積器（`Logging`） |
| `system.runner` | active | 每個 recipe 的 `main.py` 交棒的 driver：CLI（`build_arg_parser`）、run 的 seed 設定（`seed_run`）、train/validation dataloader（`build_dataloaders`，其 worker 以 `seed_worker` 啟動）、Lightning trainer 與 DDP strategy、warm start（`load_warm_start`），以及 `--training` / `--scoring` / `--inference` / `--dump_training_samples` 各階段（`run_stages`） |
| `system.curriculum` | active | `CurriculumCallback`：把 recipe curriculum 每個 epoch 的 loss 權重套到 module 上，並記錄每個排程值；curriculum 的 dataset 那一半則由 sampler 隨每個 item 帶過去（[configuration](../../usage/configuration.zh-TW.md)） |
| `system.sampling` | active | 讓 `CoverageSampler` 的抽樣序列跨 run 延續：寫進每個 checkpoint，並為 `--ckpt_path` / `--pretrained_ckpt_path` 定位（[task.sampler](../task/sampler.zh-TW.md#class-coveragesampler)） |
| `system.metric_gan` | active | `EncDecMaskBase(metric_gan=...)` 的 MetricGAN 訓練項：`MetricGanConfig`、非同步的 PESQ replay buffer（`PesqReplay`）、discriminator 與 generator loss |
| `system.paired_views` | active | `EncDecMaskBase(paired_view_consistency=...)` 的輔助視角一致性：`PairedViewConsistencyConfig` 與 `paired_view_loss`，一次跨 rank 同步的額外 forward，由所有宣告 `paired_output` 的 loss 共用 |
| `system.postprocess` | active | `Postprocessor`：僅用於推論的過度抑制緩解（`dry_blend`、`spec_floor`）、它的抑制上限，以及它在 manifest 裡的項目 |
| `system.presence_gate` | active | `PresenceGate`：僅用於推論、在 blend 之後套用的近場在場增益，來源是 bottleneck 讀出或訓練過的 presence head 的 logits |
| `system.onset_guard` | active | `OnsetGuard`：僅用於推論的起音保護，在聽到某位說話者持續說話一段時間之前都回傳乾的輸入；有離線的 torch 介面（`apply`）與給 ONNX runtime 逐 hop 使用的 numpy 介面（`streaming_state` / `step`），兩者在設計上逐位元相同 |
| [system.miso](miso.zh-TW.md) | legacy | TSE 用的 conditional（speaker-aware）enhancement trainer |

這三個僅用於推論的階段不是學出來的，也不在匯出的 graph 裡。`EncDecMaskBase.forward` 以固定順序套用它們——先 `Postprocessor`、再 `PresenceGate`、最後 `OnsetGuard`——只跑 graph 的部署就是在跑另一個系統。因此串流匯出會把 `Postprocessor` 與 `OnsetGuard` 的設定記錄在 manifest 裡，由 ONNX runtime 套用（[streaming](../../usage/streaming/index.zh-TW.md)）；`PresenceGate` 有 manifest 形式（`as_manifest`），但只在 PyTorch 的 `forward` 裡執行，串流匯出不帶它。

## Seed 與 DataLoader worker

`runner.seed_run(seed)` 呼叫 `seed_everything(seed, workers=True)`。Lightning 會在每個 DDP rank 重新套用同一個 seed，所以沒有 `workers=True` 時，每個 rank 的第 *k* 個 worker 都從同一個亂數狀態開始：各 rank 抽到的語者不同（sampler 依 rank 錯開它的亂數流），但替這些語者抽的 SNR、噪音片段、房間與混合模式都相同，雙 GPU 的 run 只看到一張卡份量的增強。自帶 seed 的 item（`CoverageSampler`、帶 seed 的 validation sampler）在 `__getitem__` 一開始就重設所有 RNG，兩種情況抽到的都一樣。

兩個訓練 loader 的 worker 都以 `runner.seed_worker` 啟動。當 `seed_everything(..., workers=True)` 有要求時，它會執行 Lightning 依 rank 區分的 worker seed 設定——Lightning 只會替沒有 initialiser 的 loader 自動加上它自己的——接著把 worker 裡每個 thread pool 都釘在一條執行緒（`puresound.utils.pin_thread_pools`）：torch 的 intra-op 與 inter-op pool、numba，以及 numpy 與 scipy 使用的 BLAS/OpenMP pool；否則 worker 會繼承每個核心一條執行緒的設定。

## Top-level Exports

```python
from puresound.system import (
    EncDecCondMaskBase,   # MISO: conditional (speaker-aware) enhancement  [legacy]
    EncDecMaskBase,       # SISO: mask/mapping enhancement                  [active]
    EncPredClassBase,     # SISO: classification (speaker embedding)        [legacy]
)
```

其他名稱都從各自的子模組 import（例如 `from puresound.system import runner`、`from puresound.system.postprocess import Postprocessor`）。
