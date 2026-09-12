# puresound.nnet.loss.asr_feature

English version: [asr_feature.md](asr_feature.md)

ASR-aware 的感知型 loss。訊號類 loss（SI-SDR / MR-STFT）獎勵的是「把干擾壓下去」，
但不會懲罰「把可懂度也一起破壞掉」，所以只靠這類 loss 調出來的模型可能會
over-suppress 真正的語音、把 WER 拉高。這個 loss 把 enhanced 輸出的 features
對齊到 clean target 的 features，兩者都取自同一個**凍結的 self-supervised
speech encoder**（torchaudio 的 wav2vec2 / HuBERT / WavLM，16 kHz，可微分），
藉此作為不可微分的 WER 的一個可微分 proxy。

## Class: `ASRFeatureLoss`

### Constructor

```python
ASRFeatureLoss(
    bundle: str = "HUBERT_BASE",   # any torchaudio.pipelines bundle name
    layers=(6, 9),                 # encoder layers whose features are matched
    loss: str = "l1",              # "l1" | "mse" | "cosine"
)
```

**Parameters:**
- `bundle` – torchaudio 的 pipeline bundle（`HUBERT_BASE`、`WAVLM_BASE_PLUS`、...）。
  疊兩個不同 bundle 的 term 可以涵蓋不同的失效模式：HuBERT 帶的是一般性的
  phonetic content，WavLM（預訓練時就有 denoising 與模擬重疊語音）則對
  deletion 容易出現的噪音/重疊情境比較敏感。
- `layers` – 要拿哪幾層 transformer 的 features 進 loss。
- `loss` – feature 距離的量法。`cosine` 是 scale-invariant 的，所以多個疊加的
  term 用相同權重就能得到相同影響力；用 `l1` 的話,不同 bundle 之間的權重就需要
  重新調整。

### `forward(enhanced, target) -> Tensor`

輸入 waveform，輸出 scalar loss。Encoder 是凍結的，並且刻意**放在 module
registry 之外**（用 list 包起來），因此既不會被存進 checkpoint,也不會被 DDP
同步;它會延遲搬到輸入所在的 device,並以 fp32 執行（關閉 autocast）。

### Config usage

```yaml
loss_func:
  - type: ASRFeatureLoss
    weighted: 1.0
    args: {bundle: HUBERT_BASE, layers: [6, 9], loss: cosine}
  - type: ASRFeatureLoss
    weighted: 1.0
    args: {bundle: WAVLM_BASE_PLUS, layers: [6, 9], loss: cosine}
```
