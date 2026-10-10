# puresound.nnet.loss.asr_feature

English version: [asr_feature.md](asr_feature.md)

透過凍結的自監督語音 encoder（torchaudio wav2vec2 / HuBERT / WavLM）做
feature matching 的 loss。它在 encoder 的中間層比較 enhanced 輸出與乾淨
target，是可微分的可懂度代理指標。

## Class：`ASRFeatureLoss`

### 計算內容

```
F_l(x) = output of transformer layer l of the frozen encoder
loss   = mean over l in layers of  d(F_l(enhanced), F_l(target))
d      = L1 | MSE | mean(1 - cosine similarity over the feature axis)
```

target 的 feature 在 `no_grad` 下計算；梯度只回傳到 `enhanced`。

### Constructor

```python
ASRFeatureLoss(
    bundle: str = "HUBERT_BASE",       # torchaudio.pipelines 的 bundle 名稱
    layers=(6, 9),                     # 要比對輸出的 transformer layer index
    loss: str = "l1",                  # "l1" | "mse" | "cosine"
    crop_seconds: float | None = None, # 評分此長度的隨機視窗；None = 整個 row
)
```

- `bundle`：例如 `HUBERT_BASE`、`WAV2VEC2_BASE`、`WAV2VEC2_ASR_BASE_960H`、
  `WAVLM_BASE_PLUS`（皆為 16 kHz）。loss 不做 resample；輸入必須是該 bundle
  的取樣率。
- `layers`：encoder 會跑到第 `max(layers) + 1` 層。
- `loss`：其他值會 raise `NotImplementedError`。
- `crop_seconds`：row 比它長時，每次呼叫只評分一個此長度的隨機視窗（兩個
  訊號用同一個視窗）。

### 輸入

`forward(enhanced, target) -> Tensor`（scalar）。waveform 可為 `[B, T]`、
`[B, 1, T]` 或 `[T]`；兩者截到較短的長度。沒有宣告 `required_inputs`，因此
拿到的是 `("enhanced", "target")`。

### Config usage

```yaml
loss_func:
  - type: ASRFeatureLoss          # 一般的音素內容
    weighted: 1.0
    args: {bundle: HUBERT_BASE, layers: [6, 9], loss: cosine, crop_seconds: 6.}
  - type: ASRFeatureLoss          # 對噪音/重疊穩健的 feature
    weighted: 1.0
    args: {bundle: WAVLM_BASE_PLUS, layers: [6, 9], loss: cosine, crop_seconds: 6.}
```

### 設計說明

- 訊號類 loss（SDR、MR-STFT）獎勵壓制干擾，但不懲罰破壞可懂度，只用它們訓練
  的模型可能過度壓制語音。以大量真實語音訓練的 SSL encoder 能在各種條件下
  穩健地編碼音素內容；對齊它們的 feature 就是對準辨識器需要的內容。WER 本身
  不可微分。
- 中間層帶有最多的音素資訊，所以預設是 `(6, 9)`。
- 可以疊兩個 bundle，因為它們失效的方式不同：HuBERT 帶一般的音素內容；WavLM
  以去噪與模擬重疊語音預訓練，對噪音與重疊條件有反應。`cosine` 與尺度無關，
  所以 `weighted` 相同的疊加項影響力相同；用 `l1` 時各 bundle 的 feature 量級
  不同，權重需要另外平衡。
- encoder 是凍結的，並且放在 list 裡、不註冊為 submodule，因此不會存進
  checkpoint、不被 DDP 同步、也不會被 optimizer 看到。第一次使用時搬到輸入的
  device，並在關閉 autocast 的 fp32 下執行。
- 有 `crop_seconds` 是因為這些 encoder 是 transformer：記憶體隨 row 長度的平方
  成長，其他項都是線性成長。這個 loss 是逐 frame 的距離，隨機視窗與整個 row
  意義相同，一個 epoch 下來也會涵蓋整個 row。
