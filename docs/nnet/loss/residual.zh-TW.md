# puresound.nnet.loss.residual

English version: [residual.md](residual.md)

僅限訓練期的背景訊號 accounting:把 mixture 減去 enhanced target 之後剩下的
殘差拿來監督。部署的模型維持單一輸出,但訓練時多了一個「被壓掉的能量跑去
哪了」的輔助訊號。

## Class: `ResidualReferenceLoss`

```
residual_pred = noisy_speech - enhanced
residual_ref  = batch[reference_key]        # e.g. consistency_noise = noisy - clean
loss          = distance(residual_pred, residual_ref)
```

### Constructor

```python
ResidualReferenceLoss(
    reference_key: str = "consistency_noise",  # batch key holding the reference residual
    loss: str = "l1",                          # "l1" | "mse" | "sdr"
    target_present_only: bool = False,         # skip target-absent rows
    sdr_args: dict | None = None,              # forwarded to SDRLoss when loss="sdr"
)
```

會設定 `uses_batch = True`,因此訓練系統會把整個 batch dict 路由進來
（這個 loss 會讀取 `reference_key`,若開了 `target_present_only`,還會讀
voice-isolation dataset 產生的 `target_present` scalar）。

### Config usage

```yaml
loss_func:
  - type: ResidualReferenceLoss
    weighted: 0.2
    args: {reference_key: consistency_noise, loss: l1, target_present_only: True}
```
