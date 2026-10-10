# puresound.nnet.loss.dist

繁體中文版本：[dist.zh-TW.md](dist.zh-TW.md)

NaN-masked regression of the backbone `DistHead`'s utterance-level outputs
(see [nnet.lobe.heads](../models/lobe/heads.md)) against the proximity labels
the dataset emits.

## Class: `DistHeadRegressionLoss`

### What it computes

Targets, in the head's output order:

| index | target | batch key |
|---|---|---|
| 0 | `foreground_drr / drr_scale` | `foreground_drr` (dB) |
| 1 | `log10(clamp(foreground_distance, min_dist, max_dist))` | `foreground_distance` (m) |
| 2 | `log10(clamp(nearest_interferer_distance, min_dist, max_dist))` | `nearest_interferer_distance` (m) |

```
valid = isfinite(target)                               # entrywise, [N, 3]
loss  = SmoothL1(dist_preds[valid], target[valid], beta)
```

A missing batch key counts as all-NaN. Any label can be NaN on any row: real
recordings carry a distance but no DRR, rows without an interferer carry no
interferer distance. With no valid entry the loss is `dist_preds.sum() * 0.0`,
a zero that keeps a graph edge, so DDP never sees the head as unused.

### Constructor

```python
DistHeadRegressionLoss(
    drr_scale: float = 10.0,   # DRR label divisor (dB / drr_scale)
    min_dist: float = 0.1,     # distance clamp before log10, metres
    max_dist: float = 30.0,
    beta: float = 1.0,         # SmoothL1 beta
)
```

### Inputs

`forward(dist_preds, batch) -> Tensor`.
`required_inputs = ("dist_preds", "batch")`: `dist_preds` is the backbone's
`last_dist_preds` `[N, 3]`, and the labels are the per-row scalars in the batch
dict. If the backbone produced no `last_dist_preds` the loss raises
`ValueError` telling you to enable `backbone_args.dist_head`.

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

### Design notes

Auxiliary multi-task supervision: regressing physical proximity (DRR,
log-distance) from the bottleneck pushes it to encode those cues, so the
near/far decision can rest on them rather than on the capture-chain signature
of the near-field training rows. Log distance and scaled DRR put the three
targets on similar ranges. The head is training-only: inference never reads it
and the streaming export is unchanged.
