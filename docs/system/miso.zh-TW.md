# puresound.system.miso

English version: [`miso.md`](miso.md)

> **Status: legacy** — 維持可運作但已凍結：不會有新功能，也不會重寫。

Multi-Input Single-Output（MISO）PyTorch Lightning 訓練模組。「Multiple input」指的是在 noisy mixture 之外，還有第二個*聲音*輸入：一段 conditioning waveform（例如一段 enrollment utterance），會經過自己的前端（`c_encoder` / `c_feats` / `c_backbone`）產生一個 embedding 來引導 mask。若模型是以*非聲音*的純量（例如一個 distance query）作為條件，那並不算 MISO——那種情況屬於 [`siso.EncDecMaskBase`](siso.zh-TW.md)，它會把純量以 FiLM bias 的形式直接餵給 backbone，完全不需要第二個前端。

Use case：Personalized Speech Enhancement / Target Speaker Extraction（TSE）。

## Class: `EncDecCondMaskBase`

> 從 `puresound.system` 匯出為 `EncDecCondMaskBase`。

### Architecture

```
Noisy Mixture Waveform                 Enrollment (Conditioning) Waveform
  └─ encoder                             └─ c_encoder (siamese copy of encoder, or separate module)
       └─ feats                               └─ c_feats (siamese copy of feats, or separate module)
            └─ backbone(features, c_features) ◄──── c_backbone
                 └─ Mask Estimation
                      └─ Mask Application (complex/deepfilter/wiener/mvdr/mapping)
                           └─ Decoder
                                └─ (Enhanced Waveform, c_features)
```

### Constructor

```python
EncDecCondMaskBase(
    encoder: nn.Module,
    feats: nn.Module,
    backbone: nn.Module,
    c_backbone: nn.Module,
    jointed_trained: bool = True,
    siamese_encoder: bool = True,
    siamese_feats: bool = True,
    c_encoder: Optional[nn.Module] = None,
    c_feats: Optional[nn.Module] = None,
    mask_type: str = "complex",
    encoder_lr_factor: float = 1.0,
    feats_lr_factor: float = 1.0,
    backbone_lr_factor: float = 1.0,
    c_encoder_lr_factor: float = 1.0,
    c_feats_lr_factor: float = 1.0,
    c_backbone_lr_factor: float = 1.0,
    verbose: bool = False,
)
```

**Parameters:**
- `encoder` / `feats` / `backbone` — 主要 enhancement 路徑（角色與 `siso.EncDecMaskBase` 相同）。
- `c_backbone` — conditioning 路徑的 backbone，把編碼後的 conditioning features 轉成一個 embedding（例如一個 ECAPA-TDNN speaker extractor）；一定要提供，且不論 `siamese_*` 為何都一定會使用。
- `siamese_encoder` / `siamese_feats` — 各自獨立控制 `c_encoder`/`c_feats` 要用 `encoder`/`feats` 的 `deepcopy`（預設 `True`），還是用透過 `c_encoder=`/`c_feats=` 明確傳入的模組（`False`——此時該引數就是必要的）。這裡沒有單一合併的 `siamese` 旗標。
- `c_encoder` / `c_feats` — 只有在對應的 `siamese_*` 為 `False` 時才會被參考。
- `jointed_trained`（注意確切拼法——不是 `joint_training`）— 若為 `False`，`c_backbone`/`c_encoder`/`c_feats` 會在 `__init__` 裡被設為 `.eval()`，且 `forward()` 會在 `torch.no_grad()` 底下執行它們。真正無條件擋住梯度的是 `torch.no_grad()`；單靠 `.eval()` 這個呼叫本身，會在 Lightning 每個 epoch 正常呼叫 `.train()` 時被還原回去——因為不同於 `siso.EncDecMaskBase` 的 gate-only 模式，這個類別並沒有覆寫 `train()` 來把這些 submodule 釘死在 eval。
- `mask_type` — 會轉小寫，但不會提早驗證。`forward()` 只處理 `"complex"`、`"deepfilter"`、`"wiener"`、`"mvdr"`、`"mapping"`；任何其他值（包括早期文件暗示可用的 `"real"`/`"magnitude"`）在第一次執行 `forward()` 時都會落入一個單純的 `raise NameError`。
- `*_lr_factor` — 六個各自獨立的模組級學習率倍率，對應 `get_total_param_groups()` 裡的每個元件群組。

### `register_loss_func`（覆寫版本）

```python
register_loss_func(
    loss_func_list: nn.ModuleList,
    loss_func_list_weights: List,
    c_loss_func_list: Optional[nn.ModuleList] = None,
    c_loss_func_list_weights: Optional[List] = None,
)
```

在 base 版本的 registry（[base.zh-TW.md](base.zh-TW.md)）之上，多加了一組可選的第二組 loss，用於 conditioning embedding，由 `compute_loss2` 使用。

### `forward(wav, conditional_wav) -> Tuple[Tensor, Tensor]`

回傳的是 **`(enh, c_features)`**——不是單一個 tensor。`enh` 是強化後的波形（clamp 到 `[-1, 1]`）；`c_features` 是經過 `c_backbone` 之後的 conditioning embedding，會被一併傳出去，讓呼叫端（`training_step`、`compute_loss2`、`predict_step`）可以監督或匯出它。

### `compute_loss(enhanced, target, vad_target=None)`

長度對齊的作法與 `siso.EncDecMaskBase.compute_loss` 相同，但分派邏輯比較窄：每個註冊的 loss，若設定了 `uses_vad_target = True`，會以 `loss_func(enhanced, target, vad_target=vad_target)` 呼叫，否則就單純呼叫 `loss_func(enhanced, target)`。它**沒有** `siso.EncDecMaskBase.compute_loss` 裡的 `uses_vad_logits` / `uses_background_vad_logits` / `uses_dist_preds` / `uses_batch` / `uses_inactive_labels` 這幾條分派分支——那些是比較新、只有 SISO 才有的新增功能。

### `compute_loss2(pred, target)`

`assert self.c_loss_func_list is not None`，然後對 `c_loss_func_list` / `c_loss_func_list_w` 做單純的加權加總（完全沒有屬性旗標分派）。用來監督 conditioning embedding（`pred`）對上一個 target（batch 裡的 `conditional_target`，例如一個 speaker-ID 標籤）。

### `training_step` / `validation_step` / `test_step` / `predict_step`

- **`training_step`** — 讀取 `noisy_speech`、`clean_speech`、`conditional_speech`、`conditional_target`；`forward` → `compute_loss`；**若** `self.jointed_trained and self.c_loss_func_list is not None`，還會執行 `compute_loss2(embedding, conditional_target)` 並加進總 loss。記錄 `train_step_loss`（`sync_dist=False`——一個做過同步的 progress-bar metric 會讓 DDP 死鎖，因為觸發這次讀取的 progress bar 刷新本身並非跨 rank 同步）。
- **`validation_step`** — 同樣是 `forward` + `compute_loss`，但**原始碼裡 `compute_loss2` 的部分是被註解掉的**，所以即使 `jointed_trained=True`，validation loss 也永遠不包含 conditioning-embedding 這一項。記錄 `valid_step_loss(_i)`，`sync_dist=True`。
- **`test_step`** — 對每個註冊的 `_metrics_func`，在評分前把 `clean_speech`/`enhanced_speech` 重新取樣到該 metric 宣告的 sample rate（`wav_resampling(..., backend="sox")`）——與 `siso.EncDecMaskBase.test_step` 的作法一致。
- **`predict_step`** — 透過 `AudioIO.save` 存下強化後的 wav，**並且**把 embedding 做 L2 normalize、若 batch 內超過一筆就取平均池化，再透過 `np.savetxt` 寫進一個同名的 `.txt` 檔——這和 `EncPredClassBase.predict_step`（見 siso.zh-TW.md）的作法一致，而不是 `EncDecMaskBase.predict_step`（只存 wav）。

### `get_total_param_groups()`

一定會回傳 `encoder` / `feats` / `backbone` 群組；**只有在 `self.jointed_trained` 為真時**才會加上 `c_encoder` / `c_feats` / `c_backbone` 群組——在凍結 conditioning 的模式下，conditioning 路徑的參數完全不會進到 optimizer 裡，這和 `jointed_trained=False` 同時在 `forward()` 裡擋住它們梯度的行為是一致的。

## Example

改寫自實際的 recipe，`egs/target_speaker_extraction/config/default_config.yaml`：

```python
from puresound.system.miso import EncDecCondMaskBase
from puresound.nnet import ConvEncDec, DPCRN, EcapaTdnnExtractor, FeatureEncoder

encoder    = ConvEncDec(fft_length=512, win_length=512, hop_length=160, fmin=0, fmax=8000, sr=16000, trainable=False)
c_encoder  = ConvEncDec(fft_length=512, win_length=512, hop_length=160, fmin=0, fmax=8000, sr=16000, preemphasis=0.97, trainable=False)
feats      = FeatureEncoder(feats_type="complex", drop_stft_first_bin=True, trainable=False)
c_feats    = FeatureEncoder(feats_type="fbank80_16k", normalized_mode="all_feature", trainable=False)
backbone   = DPCRN(input_dim=256, dvec_dim=192, channels=(2, 32, 32, 32, 64, 128))
c_backbone = EcapaTdnnExtractor(input_size=80, embedding_size=192, model_scale=8, ndim=1024, att_size=1536)

model = EncDecCondMaskBase(
    encoder=encoder,
    feats=feats,
    backbone=backbone,
    c_backbone=c_backbone,
    c_encoder=c_encoder,
    c_feats=c_feats,
    siamese_encoder=False,
    siamese_feats=False,
    jointed_trained=True,
    mask_type="complex",
)
```
