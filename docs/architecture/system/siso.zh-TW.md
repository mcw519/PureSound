# puresound.system.siso

English version: [`siso.md`](siso.md)

Single-Input Single-Output（SISO）PyTorch Lightning 訓練模組。「Single input」計算的是*聲音*觀測的數量：一段 noisy waveform 進，一段 enhanced waveform 出。MISO（[`miso.EncDecCondMaskBase`](miso.zh-TW.md)）則保留給第二個輸入本身就是一段*音訊串流*、需要自己前端的情況（例如 target-speaker extraction 用的 enrollment utterance）。

Use cases：`EncDecMaskBase` — 以 mask 或 mapping 為基礎的語音增強（noise-suppression 與 voice-isolation recipe）。`EncPredClassBase` — speaker embedding（**legacy**）。

## Class: `EncDecMaskBase`

以 mask（或 mapping）為基礎的 enhancement trainer。

```
Wav -> Encoder -> Features -> Backbone -> Apply Mask -> Restore Features -> Decoder -> Wav
```

### Constructor

```python
EncDecMaskBase(
    encoder: nn.Module,
    feats: nn.Module,
    backbone: nn.Module,
    mask_type: str = "complex",
    encoder_lr_factor: float = 1.0,
    feats_lr_factor: float = 1.0,
    backbone_lr_factor: float = 1.0,
    train_vad_head_only: bool = False,
    gate_head_lr_factor: float = 1.0,
    channel_consistency: Optional[dict] = None,
    paired_view_consistency: Optional[dict] = None,
    metric_gan: Optional[dict] = None,
    verbose: bool = False,
)
```

**Core args:**
- `encoder` — 以 STFT/Conv1D 為基礎的 encode/decode 結構（例如 `ConvEncDec`、`FreeEncDec`）。
- `feats` — encoder 與 backbone 之間的 feature transform（例如 `FeatureEncoder`）；回傳 `(features, features_for_enhanced)`。
- `backbone` — 預測 mask 的模型 backbone。
- `mask_type` — 會轉小寫，但建構時不驗證。`forward()` 處理 `"complex"`、`"deepfilter"`、`"wiener"`、`"mvdr"`、`"mapping"`；任何其他值會在第一次執行 `forward()` 時拋出 `ValueError`。
- `*_lr_factor` — 模組級學習率倍率，供 `get_total_param_groups()` 使用。

**Optional training features**（預設都關閉，各自是 recipe 裡 `lightning_module.module_args` 底下的一個設定開關）：

- **`train_vad_head_only` + `gate_head_lr_factor`** — gate-only 訓練。凍結 `encoder` / `feats` / `backbone`（`requires_grad_(False)`，並保持在 `.eval()`，讓 BatchNorm 的 running statistics 與 dropout 即使跨過 Lightning 每個 epoch 的 `.train()` 也維持固定），只訓練 `backbone.vad_head`。需要 backbone 提供一個已啟用的 `vad_head`（例如 DPCRN 的 `vad_head` 建構參數）——否則 `__init__` 會拋出 `ValueError("train_vad_head_only=True requires an enabled backbone vad_head")`。此時 `get_total_param_groups()` 只回傳單一個 `"gate_head"` 群組。由 `test/system/test_siso.py` 覆蓋測試。
- **`channel_consistency`** — 以平均比率 `prob` 在 training step 中，對（一個子 batch 的）mixture 做 channel 擾動後重跑一次 `forward()`，並懲罰 mask 因此而改變。這個擾動（`_random_channel_perturb`）是對*整段* mixture 套用一個平滑、零相位的隨機 cosine-series EQ，加上逐列的隨機增益——這正是裝置錄音鏈的物理形式，會用同一個實數且為正的 `H(f)` 縮放混音裡的每個來源，所以近/遠場的音量對比（以及理想的 complex ratio mask）不會改變；在這種擾動下發生的任何 mask 變化都是 channel 敏感度，這一項就是要把它正則掉。設定鍵值：`{enabled, prob, weight, eq_db, gain_db, eq_orders, period, max_rows}`（預設 `eq_db 6.0`、`gain_db 4.0`、`eq_orders 4`、`period 10`、`weight 1.0`）；`None`、不設定或 `{"enabled": False}` 都是嚴格的 no-op。EQ 背後的 FFT 在裝置無法規劃 transform 時，會先釋放快取的裝置記憶體重試一次，再退回 CPU——結果相同，只差那一步的延遲，每次退回都會記錄。由 `test/system/test_siso.py` 覆蓋測試。
- **`paired_view_consistency`** — `{enabled, max_rows}`（[`paired_views.PairedViewConsistencyConfig`](index.zh-TW.md)）。當 batch 帶有 `paired_view`（部分列的另一種輔助渲染）時，最多多跑一次 forward，並把主要與輔助輸出交給每個宣告了 `paired_output` / `paired_consistency` 的已註冊 loss。這次額外 forward 的 batch 大小取各 rank 的最大值，讓每個 rank 跑同樣的 collective；主要 forward 的 side output（`last_mask`、`backbone.last_*`）之後會被還原。啟用卻沒有任何 loss 宣告 `paired_output` 時會拋錯。由 `test/system/test_paired_views.py` 覆蓋測試。
- **`metric_gan`** — 一個學習出來的 PESQ critic（[`metric_gan.MetricGanConfig`](index.zh-TW.md)：`enabled, weight, warmup_steps, n_fft, hop, ndf, lr_factor, pesq_workers, rows_per_step, buffer_rows, d_rows, sample_rate`）。建立一個 `MetricDiscriminator` 作為 `self.metric_disc`；每個 training step 加上一個 discriminator 項（在 replay 的列上，其 PESQ-WB 標籤由背景 process pool 計算），並在 `warmup_steps` 之後再加上 generator 項 `weight * (D(clean, enhanced) - 1)^2`。Target-absent 的列（乾淨參考為靜音）兩項都排除。Critic 的權重會存進 checkpoint；推論與串流匯出都不讀它。由 `test/system/test_metric_gan.py` 覆蓋測試。

### `train(mode: bool = True)`

覆寫的目的是讓 gate-only 訓練時凍結的 separator 保持確定性：光是 `requires_grad=False` 並不會阻止 BatchNorm/dropout 漂移，所以只要設定了 `train_vad_head_only`，這個覆寫版本就會把 `encoder` / `feats` / `backbone` 強制設回 `.eval()`（不論 `mode` 為何），而 `backbone.vad_head` 照要求的 mode 走。沒有設定 `train_vad_head_only` 時，行為就和 `nn.Module.train` 相同。

### `forward(wav, dry_blend=1.0, spec_floor=0.0, postprocess=None, presence_gate=None, onset_guard=None) -> Tensor`

- `wav` — `[N, T]`（開頭多出的單一 channel 維度會被 squeeze 掉）。
- `dry_blend`、`spec_floor` — 僅用於推論的過度抑制緩解，會被整合成一個 [`Postprocessor`](index.zh-TW.md)（`system.postprocess`）。`spec_floor` 範圍 `[0, 1)`，把每個強化後 T-F bin 的振幅限制在至少 `spec_floor * |mix bin|`，並保留強化後的相位；只適用於 complex-mask 路徑，其他 mask type 會拋錯而不是忽略它。`dry_blend` 範圍 `(0, 1]`，在完成的波形上把未處理的輸入混回去：在兩者重疊的長度上做 `dry_blend * enh + (1 - dry_blend) * input`，再 clamp 到 `[-1, 1]`；與 mask type 無關。它為抑制設下 `20*log10(1 - dry_blend)` 的硬上限（0.9 把衰減限制在 −20 dB）。兩者的預設值都是 no-op，訓練路徑也從不設定它們。
- `postprocess` — 一個已建好的 `Postprocessor`，作為上述兩個關鍵字的替代；兩者同時給會拋出 `ValueError`。
- `presence_gate` — 一個已建好的 [`PresenceGate`](index.zh-TW.md)（`system.presence_gate`）：僅用於推論、在 blend 之後套用的增益，由 backbone bottleneck 上的線性讀出驅動。需要 backbone 有 `last_bottleneck`（否則 `ValueError`）；呼叫期間會開啟 backbone 的 `stash_bottleneck`。
- `onset_guard` — 一個已建好的 [`OnsetGuard`](index.zh-TW.md)（`system.onset_guard`），最後才套用，在 blend 與 presence gate 之後，因為它的工作是不論前面幾個階段做了什麼，都能還原乾的輸入。它只讀輸入波形。`None` 與沒有它時逐位元相同。
- 回傳 clamp 到 `[-1, 1]` 的強化波形。

`mask_type="complex"` 時，mask 由 `Masker.apply_complex_mask_with_df` 套用；當 backbone 有 deep-filter 殘差 head 時，它也會拿 backbone 的 `last_df_coefs`（否則為 `None`，此時就是單純的 complex mask）。副作用：`self.last_mask` 每次呼叫都會被重新賦值——channel-consistency 項會在呼叫後立刻讀取它；對不相關的呼叫而言，這個值沒有意義。

Mask 之後的這些階段都不在匯出的 graph 裡。串流匯出會把 `Postprocessor` 與 `OnsetGuard` 的設定記錄在 manifest 裡，由 ONNX runtime 套用；`PresenceGate` 只在這裡執行（見 [streaming](../../usage/streaming/index.zh-TW.md)）。

### `compute_loss(enhanced, target, vad_target=None, batch=None) -> (Tensor, List[float])`

把 `enhanced`/`target` 對齊到較短的長度，計算 `inactive_labels`（參考訊號完全靜音的列，`target.abs().amax(dim=-1) == 0`），再用 `reduce_losses` 加總已註冊的 loss：每個 loss 都透過 [`invoke_loss`](base.zh-TW.md) 呼叫，對照 `_loss_providers` 建出的 provider 表。Loss 以宣告 `required_inputs` 來要求輸入；這個 module 能提供：

| provider 名稱 | 值 |
|---|---|
| `enhanced`、`target` | 對齊後的波形對 |
| `batch` | batch dict（沒有時為 `{}`） |
| `inactive_labels` | 逐列的 target-absent mask |
| `vad_target` | 前景 VAD target |
| `vad_logits` | `backbone.last_vad_logits` |
| `background_vad_logits` | `backbone.last_background_vad_logits` |
| `background_vad_target` | `batch["background_vad_target"]`，或與 logits 同形狀的全零（見下方） |
| `dist_preds` | `backbone.last_dist_preds` |
| `bottleneck` | `backbone.last_bottleneck_graph`——帶著 graph、在頻率上 pool 過的 bottleneck |
| `identity_emb` | `backbone.last_identity_emb` |
| `identity_head` | `backbone.identity_head` module 本身 |
| `proximity` | `backbone.last_proximity` |

Backbone 的項目是*前一次* `forward()` 留下的 side output，以 `getattr(self.backbone, name, None)` 讀取；並非每個 backbone 都有定義。DPCRN 由它可選的 head（`vad_head`、`background_vad_head`、`dist_head`、`identity_head`、`proximity_head`；見 `puresound/nnet/dpcrn.py` 與 `puresound/nnet/lobe/heads.py`）產生這些輸出，`last_bottleneck_graph` 則只在以 `expose_bottleneck` 建構時才有。這個帶 graph 的 bottleneck 刻意與 `last_bottleneck`（presence gate 讀取、已 detach 的推論暫存）是不同的屬性與開關。`identity_head` 是唯一交出 module 而非 tensor 的項目：identity loss 在自己內部保留一份 head 權重的 stop-gradient EMA 副本，這樣未訓練的 teacher 權重就不會進到 checkpoint。`test/system/test_base.py` 會把每個出貨 loss 的 `required_inputs` 與這張表對照檢查。

`inactive_labels` 是給有選擇加入的 loss（SDR/STFT 系列）用的：它們把 target-absent 的列導向一個能避開 vanilla SDR 在全零參考上 `10*log10(0/X)` 爆炸的變體。

當 batch 沒有 `background_vad_target`、但 backbone 產生了 logits 時，`background_vad_target` 會合成為 `torch.zeros_like(background_vad_logits)`。背景完全靜音的 batch，dataset/collate 根本不會產生背景參考（collate 只在*某些*列有背景語音時才幫缺的列補零）；這是一個合法的 batch 狀態，意思是「沒有背景活動」，所以被當成全零，而不是讓 `BackgroundVADHeadBCELoss` 在 `None` 上崩潰（`test/system/test_siso.py`）。前景的情況則不會這樣補救：要求 `vad_target` 的 loss 若拿不到，會由 loss 本身拋出 `ValueError`——前景標記缺失是設定錯誤，不是合法的靜音狀態。

### `training_step` / `validation_step` / `test_step` / `predict_step`

四者開頭都會先做 `batch = self.ensure_vad_targets(batch)`（見 [base.zh-TW.md](base.zh-TW.md)）。

- **`training_step`** — `forward(noisy_speech)` → `compute_loss(..., batch=batch)` → 可選的 paired-view 項 → 可選的 MetricGAN 項 → 可選的 channel-consistency 項 → 記錄 `train_step_loss`（`sync_dist=False`：做過同步的 progress-bar metric 會讓 DDP 死鎖，因為 progress bar 的刷新——以及它觸發的 collective——並非跨 rank 同步）〔若 `verbose` 則額外記錄每一項〕→ 累積 `epoch_train_loss` → 回傳 `{"loss": total_loss}`。

  Channel-consistency 項依一個**跨 rank 同步的排程**觸發：`(batch_idx % period) < round(prob * period)`——純粹是 `batch_idx` 的函式，絕不是每個 rank 各自的隨機抽樣。這是 DDP 安全需求，不是風格選擇：多出來的 forward 會執行 SyncBatchNorm 的 all-gather，若各 rank 各自抽樣決定是否觸發，它們的 collective 順序就會失去同步，job 會卡住直到被 NCCL watchdog 殺掉。觸發時加上的 loss 是 `weight * L1(mask(perturbed_subbatch), mask(clean_subbatch).detach())`，記錄為 `train_step_cons_loss`；`max_rows` 限制子 batch 的大小（每個 rank 取相同的前幾列），避免多出來的 forward 的 activation（疊在主 forward 之上）讓峰值記憶體加倍。

- **`validation_step`** — `forward` → `compute_loss(..., batch=batch)` → 記錄 `valid_step_loss`，loss 不只一個時另記 `valid_step_loss_{i}`（`sync_dist=True`，以 epoch 為單位）。Validation 時沒有輔助項。
- **`test_step`** — 對每個註冊的 metric，若其宣告的 sample rate 與 `batch["sr"]` 不同，就把 `clean_speech`/`enhanced_speech` 的副本重新取樣過去（`wav_resampling(..., backend="sox")`），再評分並累積進 `puresound_logging`（回傳 dict 的 metric 會把它的每個 key 都加進去）。
- **`predict_step`** — `forward` → 用 `AudioIO.save` 把強化後的 wav 存到 `{eval_output_folder_path}/{batch['name'][0]}.wav`。沒有 embedding 匯出（對照下面的 `EncPredClassBase.predict_step`）。

若有啟動 MetricGAN 的 PESQ worker pool，`on_train_end` 會把它關掉。

### `get_total_param_groups()`

若 `train_vad_head_only`：只回傳 `{"gate_head": {"params": backbone.vad_head.parameters(), "lr_factor": gate_head_lr_factor}}`。否則回傳 `{"encoder": ..., "feats": ..., "backbone": ...}`，每個都是 `{"params": <module>.parameters(), "lr_factor": <module>_lr_factor}`；開啟 MetricGAN 時再加上 `"metric_disc"`（`lr_factor` 取自 MetricGAN 設定）。餵給 [`system.optim.create_optimizer_and_scheduler`](optim.zh-TW.md)。

---

## Class: `EncPredClassBase`

> **Status: legacy.**

用於 speaker-embedding 抽取的 SISO 分類 pipeline——沒有 mask，也沒有 decoder。

```
Wav -> Encoder -> Features -> Backbone -> Predict classes/embedding
```

### Constructor

```python
EncPredClassBase(
    encoder: nn.Module,
    feats: nn.Module,
    backbone: nn.Module,
    encoder_lr_factor: float = 1.0,
    feats_lr_factor: float = 1.0,
    backbone_lr_factor: float = 1.0,
    verbose: bool = False,
)
```

### `forward(wav) -> Tensor`

`encoder` → `feats`（只取它 `(features, features_for_enhanced)` 回傳值裡的 `features` 那一半）→ squeeze channel 維度 → `backbone(features)`。沒有 mask 的套用，也沒有 decoder。

### `compute_loss(pred, target)`

以 `reduce_losses` 加總，每個 loss 都以 `loss_func(pred, target)` 呼叫——沒有 provider 分派（不同於 `EncDecMaskBase.compute_loss`）。

### `training_step` / `validation_step` / `test_step` / `predict_step`

和 `EncDecMaskBase` 一樣的總 loss 記錄模式（`train_step_loss` 搭配 `sync_dist=False`，`valid_step_loss(_i)` 搭配 `sync_dist=True`）；不呼叫 `ensure_vad_targets`。`test_step` 直接呼叫 `_metrics_func[name]["func"](pred, target)`——沒有 per-metric 重新取樣（沒有波形輸出需要重新取樣）。`predict_step` 會把 embedding 做 L2 normalize、若 batch 內超過一筆就取平均池化，再透過 `np.savetxt` 寫進 `{eval_output_folder_path}/{batch['name'][0]}.txt`——沒有 wav 輸出。

### `get_total_param_groups()`

`encoder` / `feats` / `backbone` 群組，**再加上 `loss_func_list` 裡每一項各一個 `loss{i}` 群組**（`lr_factor: 1.0`）——帶參數的 loss（例如有可學習類別中心的 margin-based 分類 loss）本身帶有需要進 optimizer 的可訓練權重。

## Example

```python
import torch
from puresound.system.siso import EncDecMaskBase
from puresound.nnet.lobe.encoder import ConvEncDec
from puresound.nnet import FeatureEncoder, DPCRN
from puresound.nnet.loss.sdr import SDRLoss

encoder  = ConvEncDec(fft_length=512, win_length=512, hop_length=160, fmin=0, fmax=8000, sr=16000, trainable=False)
feats    = FeatureEncoder(feats_type="complex", drop_stft_first_bin=True, trainable=False)
# Every per-layer tuple has len(channels) - 1 entries.
backbone = DPCRN(
    input_dim=256,
    channels=(2, 16, 32, 64),
    kernel_t=(2, 2, 2), stride_t=(1, 1, 1), dilation_t=(1, 1, 1),
    kernel_f=(5, 3, 3), stride_f=(2, 2, 1), dilation_f=(1, 1, 1),
    delay=(0, 0, 0),
    rnn_hidden=96,
)

model = EncDecMaskBase(encoder=encoder, feats=feats, backbone=backbone, mask_type="complex")
model.register_loss_func(torch.nn.ModuleList([SDRLoss()]), [1.0])

enhanced = model(torch.randn(2, 16000) * 0.1)                  # training-time forward
relieved = model(torch.randn(2, 16000) * 0.1, dry_blend=0.9)   # inference relief
```

任何暴露相同 `forward(features) -> mask` 介面的 backbone 都能用同樣方式接上——例如 `DPRNN`（`puresound/nnet/dprnn.py`；建構參數為 `input_size, hidden_size, output_size, n_blocks=2, seg_size=20, seg_overlap=False, causal=True, embed_dim=0, ...`）：

```python
from puresound.nnet import DPRNN

backbone = DPRNN(input_size=256, hidden_size=64, output_size=256, n_blocks=6)
```
