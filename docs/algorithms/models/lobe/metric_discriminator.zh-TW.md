# puresound.nnet.lobe.metric_discriminator

English version: [metric_discriminator.md](metric_discriminator.md)

Metric discriminator：一個從（clean, enhanced）配對學習預測 PESQ 的小網路，在
MetricGAN 式訓練中當作 PESQ 的可微分替身（Fu et al., "MetricGAN: Generative
Adversarial Networks based Black-box Metric Scores Optimization for Speech
Enhancement", ICML 2019；Fu et al., "MetricGAN+: An Improved Version of MetricGAN
for Speech Enhancement", Interspeech 2021）。架構沿用 CMGAN 的 discriminator
（Cao et al., "CMGAN: Conformer-based Metric GAN for Speech Enhancement",
Interspeech 2022），MP-SENet 也用同一個。

只在訓練時使用：它的權重會存進訓練 checkpoint，但推論與串流匯出都不會讀它。

## Class: `MetricDiscriminator`

```
M(w)  = |STFT(w)|^0.3                                    # Hann window、fp32、關閉 autocast
pair  = stack(M(reference), M(estimate))                 # [B, 2, F, frames]
h     = 4 x [SN-Conv2d 4x4 stride 2 -> InstanceNorm2d -> PReLU]   # ndf、2ndf、4ndf、8ndf 個 channel
v     = AdaptiveMaxPool2d(1)(h)                          # [B, 8ndf]
score = LearnableSigmoid(SN-Linear(PReLU(Dropout0.3(SN-Linear(v)))))   # 8ndf -> 4ndf -> n_outputs
```

`SN` 是 spectral normalisation。（clean, enhanced）配對的目標是正規化的 PESQ-WB
`(pesq - 1) / 3.5`，裁到 `[0, 1]`；（clean, clean）的目標是 1。

```python
MetricDiscriminator(
    n_fft: int = 400,      # 讀入波形的 STFT
    hop: int = 100,
    ndf: int = 16,         # 第一層 conv 的寬度；每層加倍
    beta: float = 1.2,     # 輸出 sigmoid 的上限
    n_outputs: int = 1,    # 每列預測的指標數
)
```

- `forward(reference [B, T], estimate [B, T]) -> [B]`，`n_outputs > 1` 時為
  `[B, n_outputs]`。兩個波形都裁到較短的長度。
- `magnitude(wav [B, T]) -> [B, n_fft // 2 + 1, frames]`，fp32 的 `|STFT|^0.3`。

`n_outputs > 1` 時最後一層從同一個共享主幹預測多個指標，每個指標有自己的輸出單元與
sigmoid 斜率。

## Class: `LearnableSigmoid`

```python
LearnableSigmoid(features: int = 1, beta: float = 1.2)   # beta * sigmoid(slope * x)，slope 可學習
```

## 在訓練中的用法

`EncDecMaskBase` 在 `metric_gan` 區塊啟用時建立 discriminator，傳入 `n_fft`、`hop`、
`ndf`（`beta` 與 `n_outputs` 維持預設）：

```yaml
model:
  lightning_module:
    type: EncDecMaskBase
    module_args:
      metric_gan:
        enabled: true
        weight: 5.0
        warmup_steps: 1000
        lr_factor: 5.0
        pesq_workers: 4
        rows_per_step: 12
        buffer_rows: 256
        d_rows: 12
```

| key | 預設 | 意義 |
|---|---|---|
| `enabled` | `false` | 建立 discriminator 並加入兩個損失項 |
| `weight` | `0.0` | generator 項的權重；`0` 表示只訓練 discriminator |
| `warmup_steps` | `1000` | generator 項開啟前的 optimizer step 數 |
| `n_fft`、`hop`、`ndf` | `400`、`100`、`16` | discriminator 的 STFT 與寬度 |
| `lr_factor` | `1.0` | discriminator 參數群組的 learning-rate 倍率 |
| `pesq_workers` | `4` | 每個 rank 的 PESQ worker process 數；`0` 表示同步計算 |
| `rows_per_step` | `12` | 每個 batch 送去算 PESQ 的列數 |
| `buffer_rows` | `256` | replay buffer 容量（列） |
| `d_rows` | `12` | 每次 discriminator 更新重播的列數 |
| `sample_rate` | `16000` | 給 PESQ 的取樣率 |

每個訓練 step 在一般損失上再加兩項（`puresound/system/metric_gan.py`）：

```
L_D = (D(c, c) - 1)^2  +  (D(c', ê') - q')^2        # c'、ê'、q'：重播的列與其 PESQ 標籤
L_G = weight * (D(c, ê) - 1)^2                      # D 的參數凍結；warmup_steps 之後才開
```

目標不存在的列（clean reference 為靜音）沒有 PESQ，兩項都排除。系統端見
[EncDecMaskBase](../../../architecture/system/siso.zh-TW.md)。

## 設計說明

- **壓縮過的 magnitude 輸入。** `|X|^0.3` 近似 PESQ 運作的響度域。
- **`beta` 大於 1** 讓輸出能達到 clean 對 clean 的目標 1.0，而不必把 logit 推向無限大
  （MetricGAN+ 用 1.2）。
- **非同步標籤。** PESQ 在 CPU 上跑，一整個 batch 的量會主導 step 時間，所以標籤由背景
  process pool 產生、放進 replay buffer，discriminator 從過去的強化結果學習（這正是
  MetricGAN+ 刻意保留的 replay）。
- **每一項只作用在一邊。** generator 項執行時 discriminator 參數凍結，discriminator
  項讀的是 detach 過的音訊，所以同一個 optimizer 可以同時更新兩者。clean 對 clean 那項
  每個 step 都執行，讓每個 rank 上 discriminator 的每個參數都有梯度，這是 DDP 的要求。
