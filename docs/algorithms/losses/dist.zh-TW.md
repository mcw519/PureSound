# puresound.nnet.loss.dist

English version: [dist.md](dist.md)

把 backbone `DistHead` 的 utterance-level 輸出（見
[nnet.lobe.heads](../models/lobe/heads.zh-TW.md)）對 dataset 產生的距離標籤做
NaN-masked regression。

## Class：`DistHeadRegressionLoss`

### 計算內容

target 依 head 的輸出順序：

| index | target | batch key |
|---|---|---|
| 0 | `foreground_drr / drr_scale` | `foreground_drr`（dB） |
| 1 | `log10(clamp(foreground_distance, min_dist, max_dist))` | `foreground_distance`（公尺） |
| 2 | `log10(clamp(nearest_interferer_distance, min_dist, max_dist))` | `nearest_interferer_distance`（公尺） |

```
valid = isfinite(target)                               # entrywise, [N, 3]
loss  = SmoothL1(dist_preds[valid], target[valid], beta)
```

batch 裡沒有的 key 視為全部 NaN。任何標籤在任何 row 上都可能是 NaN：真實錄音
有距離但沒有 DRR，沒有干擾者的 row 沒有干擾者距離。完全沒有有效項時 loss 為
`dist_preds.sum() * 0.0`，一個保留 graph edge 的零，讓 DDP 不會把 head 視為
未使用。

### Constructor

```python
DistHeadRegressionLoss(
    drr_scale: float = 10.0,   # DRR 標籤的除數（dB / drr_scale）
    min_dist: float = 0.1,     # 取 log10 前的距離 clamp，單位公尺
    max_dist: float = 30.0,
    beta: float = 1.0,         # SmoothL1 的 beta
)
```

### 輸入

`forward(dist_preds, batch) -> Tensor`。
`required_inputs = ("dist_preds", "batch")`：`dist_preds` 是 backbone 的
`last_dist_preds` `[N, 3]`，標籤是 batch dict 裡每個 row 的 scalar。若 backbone
沒有產生 `last_dist_preds`，loss 會 raise `ValueError`，提示要開啟
`backbone_args.dist_head`。

### Config usage

```yaml
model:
  backbone:
    backbone_args:
      dist_head: {enabled: True, hidden: 128}
loss_func:
  - type: DistHeadRegressionLoss
    weighted: 0.3
    args: {drr_scale: 10.0, min_dist: 0.1, max_dist: 30.0}
```

### 設計說明

輔助的 multi-task 監督：從 bottleneck 回歸物理上的遠近（DRR、log 距離），
迫使 bottleneck 編碼這些線索，讓近/遠判斷可以依據它們，而不是依據近場訓練
row 的錄音鏈特徵。log 距離與縮放後的 DRR 讓三個 target 落在相近的範圍。
這個 head 只用於訓練：推論不會讀它，streaming export 也不受影響。
