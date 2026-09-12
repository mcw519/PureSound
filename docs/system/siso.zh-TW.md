# puresound.system.siso

English version: [`siso.md`](siso.md)

Single-Input Single-Output（SISO）PyTorch Lightning 訓練模組。「Single input」計算的是*聲音*觀測的數量：一段 noisy waveform 進，一段 enhanced waveform 出。MISO（[`miso.EncDecCondMaskBase`](miso.zh-TW.md)）則是保留給第二個輸入本身就是一段*音訊串流*、需要自己前端的情況（例如 target-speaker extraction 用的 enrollment utterance）。

Use cases：`EncDecMaskBase` — 以 mask 或 mapping 為基礎的語音增強。`EncPredClassBase` — speaker embedding（**legacy**）。

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
    verbose: bool = False,
)
```

**Core args:**
- `encoder` — 以 STFT/Conv1D 為基礎的 encode/decode 結構（例如 `ConvEncDec`、`FreeEncDec`）。
- `feats` — encoder 與 backbone 之間的 feature transform（例如 `FeatureEncoder`）；回傳 `(features, features_for_enhanced)`。
- `backbone` — 預測 mask 的模型 backbone。
- `mask_type` — 會轉小寫，但不會提早驗證。`forward()` 只處理 `"complex"`、`"deepfilter"`、`"wiener"`、`"mvdr"`、`"mapping"`；任何其他值（例如 `"real"`、`"polar"`）都會在第一次執行 `forward()` 時（而非建構當下）落入一個單純的 `raise NameError`。
- `*_lr_factor` — 模組級學習率倍率，供 `get_total_param_groups()` 使用。

**Optional training features**（預設都關閉，各自是一個設定開關）：

- **`train_vad_head_only` + `gate_head_lr_factor`** — gate-only 訓練模式。凍結 `encoder` / `feats` / `backbone`（`requires_grad_(False)`，並保持在 `.eval()`，讓 BatchNorm 的 running statistics 與 dropout 即使跨過 Lightning 每個 epoch 的 `.train()` 呼叫也維持固定），只訓練 `backbone.vad_head`。這需要 backbone 本身已經啟用了 `vad_head`（例如 DPCRN 的 `vad_head={"enabled": True, ...}` 建構參數）——否則 `__init__` 會拋出 `ValueError("train_vad_head_only=True requires an enabled backbone vad_head")`。此時 `get_total_param_groups()` 只會回傳單一個 `"gate_head"` 群組，而不是平常的三個。實際 recipe：a gate-only recipe（`backbone_lr_factor: 0.0`、`train_vad_head_only: True`，從一個凍結的 separator checkpoint warm-start）。由 `test/test_system/test_dpcrn_gate_training.py` 與 `test_param_groups_optim.py` 覆蓋測試。
- **`channel_consistency`** — 每個 training step 有機率 `prob`，對（一個子 batch 的）mixture 做 channel 擾動後重跑一次 `forward()`，並懲罰 mask 因此而改變。這個擾動（`_random_channel_perturb`）是對*整段*訊號套用一個平滑的隨機 cosine-series EQ 加上隨機增益——這正是裝置錄音鏈的物理形式，會用同一個實數且為正的 `H(f)` 縮放混音裡的每個來源，所以近/遠場的音量對比（以及理想的 complex ratio mask）並不會改變；任何在這種擾動下發生的 mask 變化，都是純粹的 channel 敏感度，而這個正則項的目的就是壓制它。設定鍵值：`{enabled, prob, weight, eq_db, gain_db, eq_orders, period, max_rows}`；`None`/不設定/`{"enabled": False}` 都是嚴格的 no-op。實際 recipe：`egs/voice_isolate/config/train_dpcrn.yaml`（`{enabled: True, prob: 0.3, weight: 1.0, eq_db: 6.0, gain_db: 4.0, max_rows: 4}`）。由 `test/test_system/test_channel_consistency.py` 覆蓋測試。

**Optional inference knobs**（`forward()` 的參數，對訓練沒有影響——訓練時的呼叫端永遠不會傳非預設值）：
- `dry_blend`、`spec_floor` — 用來緩解過度抑制；見下方 `forward()`。

### `train(mode: bool = True)`

之所以覆寫，是為了讓 gate-only 訓練時凍結的 separator 保持確定性：光是 `requires_grad=False` 並不會阻止 BatchNorm/dropout 漂移，所以只要設定了 `train_vad_head_only`，這個覆寫版本就永遠會把 `encoder` / `feats` / `backbone` 強制設回 `.eval()`（不論傳入的 `mode` 為何），同時讓 `backbone.vad_head` 照要求的 mode 走。若沒有設定 `train_vad_head_only`，行為就和 `nn.Module.train` 完全相同。

### `forward(wav, dry_blend=1.0, spec_floor=0.0) -> Tensor`

- `wav` — `[N, T]`（開頭多出的單一 channel 維度會被 squeeze 掉）。
- `dry_blend` — 僅用於推論時緩解過度抑制，範圍 `(0, 1]`。輸出會變成兩者重疊長度上的 `dry_blend * enh + (1 - dry_blend) * input`，再 clamp 到 `[-1, 1]`；`1.0`（預設值）是 no-op。這正是 `egs/voice_isolate` 部署 checkpoint 所搭配的 release blend——`dpcrn_v8`/`dpcrn_curriculum_v1` 都是以 `dry_blend 0.9` 發布（`out = 0.9*enhanced + 0.1*input`），把最差情況下的衰減限制在約 −20 dB，以少量的 interferer 洩漏換取更少的刪除（見 `egs/voice_isolate/README.md`）。這個參數與 `mask_type` 無關——它是在波形域、解碼之後套用的，不論 `mask_type` 為何都一樣。
- `spec_floor` — 僅用於推論時的頻譜下限，範圍 `[0, 1)`，**只適用於 complex-mask 模型**。會把每個強化後 T-F bin 的振幅限制在至少 `spec_floor * |mix bin|`，同時保留強化後的相位（`_apply_spec_floor`）；`0.0`（預設值）是 no-op。
- 回傳 clamp 到 `[-1, 1]` 的強化波形。
- 副作用：`self.last_mask` 每次呼叫都會被重新賦值——`training_step` 裡的 channel-consistency 正則項會在呼叫後立刻讀取它；對不相關的呼叫而言，這個值沒有任何意義。

### `compute_loss(enhanced, target, vad_target=None, batch=None) -> (Tensor, List[float])`

先把 `enhanced`/`target` 對齊到較短的長度，然後走訪 `self.loss_func_list` / `self.loss_func_list_w`，依 loss module 自己設定的屬性旗標分派每一項：

| loss_func 上的旗標 | 呼叫方式 |
|---|---|
| `uses_vad_logits` | `loss_func(backbone.last_vad_logits, vad_target)` |
| `uses_background_vad_logits` | `loss_func(backbone.last_background_vad_logits, bg_target)` — 見下方說明 |
| `uses_dist_preds` | `loss_func(backbone.last_dist_preds, batch or {})` |
| `uses_batch` | `loss_func(enhanced, target, batch or {})` |
| `uses_vad_target` | `loss_func(enhanced, target, vad_target=vad_target)` |
| `uses_inactive_labels` | `loss_func(enhanced, target, inactive_labels=inactive_labels)` |
| （以上皆非） | `loss_func(enhanced, target)` |

`inactive_labels` 標記的是參考訊號完全靜音的那些列（`target.abs().amax(dim=-1) == 0`，也就是 target-absent 的訓練列）——有選擇加入（opt in）的 loss（例如 SDR/STFT 系列）會把這些列導向一個能避開 vanilla SDR 在全零參考上 `10*log10(0/X)` 爆炸的變體。`last_vad_logits` / `last_background_vad_logits` / `last_dist_preds` 都是*前一次* `forward()` 呼叫留下的 backbone 側輸出，透過 `getattr(self.backbone, ..., None)` 讀取——並非每個 backbone 都定義了全部三個。DPCRN 目前透過可選的 `vad_head` / `dist_head` 建構參數提供 `last_vad_logits` / `last_dist_preds`（見 `puresound/nnet/dpcrn.py`，head 定義在 `puresound/nnet/lobe/heads.py`）；目前 repo 裡還沒有任何 backbone 定義 `last_background_vad_logits`，所以這個旗標目前是面向未來的預留管線——`BackgroundVADHeadBCELoss` 與其測試已經會用到它，只是還沒有真正部署的 head 會產生對應的輸出。

`uses_background_vad_logits` 用的 `bg_target`，若 `batch.get("background_vad_target")` 存在就直接用；但如果這個 key 不存在、且 `background_vad_logits is not None`，就會合成一個 `torch.zeros_like(background_vad_logits)`，而不是讓 `None` 直接傳下去。原因是：一個背景完全靜音的 batch，dataset/collate 根本不會產生任何 `background_vad_reference`/`_target`（collate 只有在 batch 裡*某些*列真的有背景語音時，才會幫缺的列補零）——這是一個真實、合法的 batch 狀態，不是錯誤——所以 `compute_loss` 會把它當成「沒有背景活動」處理，而不是讓 `BackgroundVADHeadBCELoss` 在 `None` 上崩潰（`test/test_system/test_siso_compute_loss.py` 裡有針對這個 regression 的測試；這個缺口曾經真的弄垮過一個 DDP job）。前景（foreground）的對應情況則**不會**用同樣的方式被拯救：若某個設定了 `uses_vad_target=True` 的 loss 缺少 `vad_target`，仍然會浮現一個 `ValueError`（由 `VADHeadBCELoss` 自己拋出，不是 `compute_loss`）——前景標記缺失被視為設定錯誤，而不是一個合法的靜音狀態。

### `training_step` / `validation_step` / `test_step` / `predict_step`

四者開頭都會先做 `batch = self.ensure_vad_targets(batch)`（見 [base.zh-TW.md](base.zh-TW.md)）。

- **`training_step`** — `forward(noisy_speech)` → `compute_loss(..., batch=batch)` → 可選的 channel-consistency 項 → 記錄 `train_step_loss`（`sync_dist=False`；一個做過同步的 progress-bar metric 會讓 DDP 死鎖，因為 progress bar 的刷新——以及它觸發的 collective——並非跨 rank 同步）〔若 `verbose` 則額外記錄每一項〕→ 累積 `epoch_train_loss` → 回傳 `{"loss": total_loss}`。

  Channel-consistency 這一項，若有設定，會依一個**跨 rank 同步的排程**觸發：`(batch_idx % period) < round(prob * period)`——純粹是 `batch_idx` 的函式，絕不是每個 rank 各自的隨機抽樣。這是硬性的 DDP 安全需求，不是風格選擇：多出來的這次 forward 會觸發 SyncBatchNorm 的 all-gather，如果各 rank 各自用隨機抽樣決定要不要觸發，它們的 collective 順序就會失去同步，整個 job 會卡住，直到被 NCCL watchdog 殺掉。觸發時，加上去的 loss 是 `weight * L1(mask(perturbed_subbatch), mask(clean_subbatch).detach())`，記錄為 `train_step_cons_loss`；`max_rows` 限制了這個子 batch 的大小，避免多出來的這次 forward 的 activation（疊加在還沒 backward 的主 forward 之上）讓峰值記憶體加倍。

- **`validation_step`** — `forward` → `compute_loss(..., batch=batch)` → 記錄 `valid_step_loss(_i)`（`sync_dist=True`）。validation 時沒有 channel-consistency 項。
- **`test_step`** — 對每個註冊的 `_metrics_func`，若與 `batch["sr"]` 不同，就先把 `clean_speech`/`enhanced_speech` 重新取樣到該 metric 宣告的 sample rate（`wav_resampling(..., backend="sox")`），再評分並累積進 `puresound_logging`。
- **`predict_step`** — `forward` → 用 `AudioIO.save` 把強化後的 wav 存到 `{eval_output_folder_path}/{batch['name'][0]}.wav`。沒有 embedding 匯出（相對地，下面的 `EncPredClassBase.predict_step`，以及 `miso.EncDecCondMaskBase.predict_step`，都還會另外寫一個 `.txt` embedding）。

### `get_total_param_groups()`

若 `train_vad_head_only`：只回傳 `{"gate_head": {"params": backbone.vad_head.parameters(), "lr_factor": gate_head_lr_factor}}`。否則：`{"encoder": ..., "feats": ..., "backbone": ...}`，每個都是 `{"params": <module>.parameters(), "lr_factor": <module>_lr_factor}`。直接餵給 [`system.optim.create_optimizer_and_scheduler`](optim.zh-TW.md)。

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

對 `loss_func_list` / `loss_func_list_w` 做單純的加權加總——沒有屬性旗標分派（不同於 `EncDecMaskBase.compute_loss`）。

### `training_step` / `validation_step` / `test_step` / `predict_step`

和 `EncDecMaskBase` 一樣的總 loss 記錄模式（`train_step_loss` 搭配 `sync_dist=False`，`valid_step_loss(_i)` 搭配 `sync_dist=True`）。`test_step` 直接呼叫 `_metrics_func[name]["func"](pred, target)`——沒有 per-metric 重新取樣（因為沒有波形輸出需要重新取樣）。`predict_step` 會把 embedding 做 L2 normalize、若 batch 內超過一筆就取平均池化，再透過 `np.savetxt` 寫進 `{eval_output_folder_path}/{batch['name'][0]}.txt`——沒有 wav 輸出。

### `get_total_param_groups()`

`encoder` / `feats` / `backbone` 群組，**再加上 `loss_func_list` 裡每一項各一個 `loss{i}` 群組**（`lr_factor: 1.0`）——帶參數的 loss（例如一個有可學習類別中心的 margin-based 分類 loss）本身就帶有需要進 optimizer 的可訓練權重。`EncDecMaskBase.get_total_param_groups()` 從不會這樣做。

## Example

```python
import torch
from puresound.system.siso import EncDecMaskBase
from puresound.nnet.lobe.encoder import ConvEncDec
from puresound.nnet import FeatureEncoder, DPCRN
from puresound.nnet.loss.sdr import SDRLoss

encoder  = ConvEncDec(fft_length=512, win_length=512, hop_length=160, fmin=0, fmax=8000, sr=16000, trainable=False)
feats    = FeatureEncoder(feats_type="complex", drop_stft_first_bin=True, trainable=False)
backbone = DPCRN(input_dim=256, channels=(2, 32, 64, 128), rnn_hidden=96)

model = EncDecMaskBase(encoder=encoder, feats=feats, backbone=backbone, mask_type="complex")
model.register_loss_func(torch.nn.ModuleList([SDRLoss()]), [1.0])
```

任何暴露相同 `forward(features) -> mask` 介面的 backbone 都能用同樣的方式接上——例如 `DPRNN`（`puresound/nnet/dprnn.py`；實際的建構參數是 `input_size, hidden_size, output_size, n_blocks=2, seg_size=20, seg_overlap=False, causal=True, embed_dim=0, ...`，**不是** `in_channel`/`hid_channel`/`out_channel`/`num_layers`）：

```python
from puresound.nnet import DPRNN

backbone = DPRNN(input_size=256, hidden_size=64, output_size=256, n_blocks=6)
```
