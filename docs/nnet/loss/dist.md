# puresound.nnet.loss.dist

NaN-masked regression on the backbone `DistHead`'s utterance-level outputs
(see [nnet.lobe.heads](../lobe/heads.md)). Auxiliary multi-task supervision:
the bottleneck is regressed against physical proximity labels the dataset
already emits for free, pushing the near/far decision toward distance/DRR cues
rather than the capture-chain signature of the training near-field rows.

## Class: `DistHeadRegressionLoss`

Targets, in the head's output order:

| index | target | populated on |
|---|---|---|
| 0 | `foreground_drr / drr_scale` | simulated rows only |
| 1 | `log10(foreground_distance)` | simulated + real-near rows |
| 2 | `log10(nearest_interferer_distance)` | simulated + real-far rows |

Every label can be NaN for any given row (real recordings carry a distance but
no DRR; no-interferer rows carry no interferer distance). Masking is entrywise,
and a batch with zero valid entries contributes a zero loss that still carries a
graph edge, so DDP never sees an unused head.

### Constructor

```python
DistHeadRegressionLoss(
    drr_scale: float = 10.0,   # DRR label scaling (dB / drr_scale)
    min_dist: float = 0.1,     # distance clamp range before log10
    max_dist: float = 30.0,
    beta: float = 1.0,         # SmoothL1 beta
)
```

Sets `uses_dist_preds = True`, so the training system routes
`backbone.last_dist_preds` plus the batch's scalar labels in.

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

Training-only: inference never reads the head and the streaming export is
untouched.
