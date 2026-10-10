# puresound.nnet.loss.residual

English version: [residual.md](residual.md)

僅限訓練期的背景訊號 accounting：監督模型拿掉了什麼。把 mixture 減去
enhanced 輸出後剩下的殘差，拿來與 batch 裡的參考殘差比較。

## Class：`ResidualReferenceLoss`

### 計算內容

```
residual_pred = batch["noisy_speech"] - enhanced
residual_ref  = batch[reference_key]         # default consistency_noise = noisy - clean
loss          = d(residual_pred, residual_ref),   d = L1 | MSE | SDRLoss(**sdr_args)
```

### Constructor

```python
ResidualReferenceLoss(
    reference_key: str = "consistency_noise",  # 存放參考殘差的 batch key
    loss: str = "l1",                          # "l1" | "mse" | "sdr"
    target_present_only: bool = False,         # 只評分 target_present > 0.5 的 row
    sdr_args: dict | None = None,              # loss="sdr" 時轉給 SDRLoss
)
```

其他的 `loss` 值會 raise `NotImplementedError`。

### 輸入

`forward(enhanced, target, batch) -> Tensor`。
`required_inputs = ("enhanced", "target", "batch")`。它讀取
`batch["noisy_speech"]` 與 `batch[reference_key]`（缺少時 raise `KeyError`），
開啟 `target_present_only` 時再讀每個 row 的 `target_present` scalar。各 tensor
之間的單一 channel 軸會對齊，三者都截到最短的長度。若 `target_present_only`
篩完沒有剩下任何 row，loss 為 `enhanced.sum() * 0.0`，一個保留 graph edge
的零。

noise-suppression 與 voice-isolation 的 dataset 會產生 `consistency_noise`
（`noisy_speech - clean_speech`）；voice-isolation 的 dataset 會產生
`target_present`。

### Config usage

```yaml
loss_func:
  - type: ResidualReferenceLoss
    weighted: 0.2
    args: {reference_key: consistency_noise, loss: l1, target_present_only: True}
```

### 設計說明

部署的模型維持單一輸出；這一項只是在訓練時多一個「被壓掉的能量跑去哪了」的
訊號，直接監督被拿掉的部分，而不只監督被保留的部分。`target_present_only` 跳過
target-absent row：它們的參考殘差就是整個 mixture，這一項會退化成「輸出為零」，
而 [`SDRLoss`](sdr.zh-TW.md) 的 target-absent 部分已經在要求這件事。
