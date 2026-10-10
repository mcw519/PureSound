# puresound.nnet.loss.residual

繁體中文版本：[residual.zh-TW.md](residual.zh-TW.md)

Training-only background accounting: supervises what the model removed. The
residual left after subtracting the enhanced output from the mixture is compared
with a reference residual from the batch.

## Class: `ResidualReferenceLoss`

### What it computes

```
residual_pred = batch["noisy_speech"] - enhanced
residual_ref  = batch[reference_key]         # default consistency_noise = noisy - clean
loss          = d(residual_pred, residual_ref),   d = L1 | MSE | SDRLoss(**sdr_args)
```

### Constructor

```python
ResidualReferenceLoss(
    reference_key: str = "consistency_noise",  # batch key holding the reference residual
    loss: str = "l1",                          # "l1" | "mse" | "sdr"
    target_present_only: bool = False,         # score only rows with target_present > 0.5
    sdr_args: dict | None = None,              # forwarded to SDRLoss when loss="sdr"
)
```

Any other `loss` raises `NotImplementedError`.

### Inputs

`forward(enhanced, target, batch) -> Tensor`.
`required_inputs = ("enhanced", "target", "batch")`. It reads
`batch["noisy_speech"]` and `batch[reference_key]` (a missing key raises
`KeyError`) and, with `target_present_only`, the per-row `target_present`
scalar. A singleton channel axis is reconciled between the tensors, and all
three are cut to the shortest length. When `target_present_only` leaves no row,
the loss is `enhanced.sum() * 0.0`, a zero that keeps a graph edge.

The noise-suppression and voice-isolation datasets emit `consistency_noise`
(`noisy_speech - clean_speech`); the voice-isolation dataset emits
`target_present`.

### Config usage

```yaml
loss_func:
  - type: ResidualReferenceLoss
    weighted: 0.2
    args: {reference_key: consistency_noise, loss: l1, target_present_only: True}
```

### Design notes

The deployed model stays single-output; this term only adds a training signal
for where the suppressed energy went, supervising the removed part directly
instead of only the kept part. `target_present_only` skips target-absent rows:
their reference residual is the whole mixture, so the term would reduce to
"output zero", which the target-absent part of [`SDRLoss`](sdr.md) already
asks for.
