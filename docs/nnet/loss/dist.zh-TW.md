# puresound.nnet.loss.dist

English version: [dist.md](dist.md)

對 backbone `DistHead` 輸出的 utterance-level 結果做 NaN-masked regression
（參見 [nnet.lobe.heads](../lobe/heads.md)）。這是一種輔助性的多工監督：把
bottleneck 對齊到資料集本來就免費附帶的物理鄰近度標籤上,把 near/far 的判斷
往 distance/DRR 這類線索推,而不是依賴訓練資料中 near-field rows 的 capture-chain
特徵。

## Class: `DistHeadRegressionLoss`

Target,依 head 輸出的順序:

| index | target | populated on |
|---|---|---|
| 0 | `foreground_drr / drr_scale` | 僅限 simulated rows |
| 1 | `log10(foreground_distance)` | simulated + real-near rows |
| 2 | `log10(nearest_interferer_distance)` | simulated + real-far rows |

每一列的每個 label 都可能是 NaN（真實錄音有距離但沒有 DRR;沒有 interferer 的
row 就沒有 interferer distance）。Masking 是逐項（entrywise）進行的,若整個
batch 沒有任何有效項,就回傳一個 loss 值為零、但仍然掛在計算圖上的 tensor,
確保 DDP 不會偵測到某個 head 沒被用到。

### Constructor

```python
DistHeadRegressionLoss(
    drr_scale: float = 10.0,   # DRR label scaling (dB / drr_scale)
    min_dist: float = 0.1,     # distance clamp range before log10
    max_dist: float = 30.0,
    beta: float = 1.0,         # SmoothL1 beta
)
```

會設定 `uses_dist_preds = True`,因此訓練系統會把 `backbone.last_dist_preds`
連同 batch 裡的 scalar labels 一起路由進來。

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

僅限訓練期使用:推論階段不會讀這個 head,streaming export 也不受影響。
