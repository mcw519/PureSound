# puresound.nnet.loss.residual

Training-only background accounting: supervise the residual left after
subtracting the enhanced target from the mixture. The deployed model stays
single-output while training gets an auxiliary "where did the suppressed energy
go?" signal.

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

Sets `uses_batch = True`, so the training system routes the whole batch dict in
(the loss reads `reference_key` and, with `target_present_only`, the
`target_present` scalar the voice-isolation dataset emits).

### Config usage

```yaml
loss_func:
  - type: ResidualReferenceLoss
    weighted: 0.2
    args: {reference_key: consistency_noise, loss: l1, target_present_only: True}
```
