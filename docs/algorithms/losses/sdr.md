# puresound.nnet.loss.sdr

繁體中文版本：[sdr.zh-TW.md](sdr.zh-TW.md)

Time-domain SDR family losses. One class, `SDRLoss`, covers SI-SNR, SD-SDR,
plain SDR, SA-SDR and t-SDR through four flags, and scores target-absent rows
(silent reference) with a separate energy term instead of the undefined ratio.

## Class: `SDRLoss`

### What it computes

Per active row, after optional zero-mean:

```
s_target    = <s1, s2> / (<s2, s2> + eps) * s2      if scaled          else s2
e_noise     = s1 - s2                               if scale_dependent else s1 - s_target
target_norm = <s_target, s_target>
noise_norm  = <e_noise, e_noise>   (+ tau * target_norm when sdr_max is set,
                                      tau = 10 ** (-sdr_max / 10))
loss        = -10 * log10(target_norm / (noise_norm + eps) + eps)
```

`<a, b>` is `sum(a * b, dim=-1)`. With `source_aggregated=True` the energies are
summed over the source axis before the ratio. The SI-SNR case
(`scaled=True, scale_dependent=False`):

$$\mathcal{L} = -10 \log_{10} \frac{\|\alpha s\|^2}{\|\hat{s} - \alpha s\|^2}, \qquad \alpha = \frac{\langle \hat{s}, s \rangle}{\langle s, s \rangle}$$

with $\hat s$ = `s1` (enhanced) and $s$ = `s2` (reference).

### Constructor

```python
SDRLoss(
    scaled: bool = True,                # project the reference onto the estimate (scale-invariant)
    scale_dependent: bool = False,      # residual against the raw reference; acts only when scaled=True
    zero_mean: bool = True,             # remove per-row DC before anything else
    source_aggregated: bool = False,    # SA-SDR; input must be [N, S, L]
    sdr_max: int = None,                # soft ceiling in dB (t-SDR)
    eps: float = 1e-8,
    reduction: bool = True,             # bool: True -> scalar mean, False -> per-row [rows, 1]
    threshold: Optional[float] = None,  # drop active rows whose loss is <= threshold
    inactive_mode: str = "absolute",    # "absolute" | "mean" | "relative", see below
)
```

- `scale_dependent` has no effect when `scaled=False`: `s_target` is then `s2`
  and `e_noise` is `s1 - s2` either way.
- `sdr_max` bounds the loss below by `-sdr_max`. A `threshold` equal to
  `-sdr_max` therefore removes nothing; a higher `threshold` drops rows that are
  already easy. If every active row would be dropped, the filter is skipped.
- `source_aggregated=True` asserts 3-D input `[N, S, L]`; otherwise the input
  must be 2-D `[N, L]`.
- Any other `inactive_mode` raises `ValueError`.

### Inputs

`forward(s1, s2, inactive_labels=None, batch=None) -> Tensor`

`required_inputs` is `("enhanced", "target", "inactive_labels")`, extended with
`"batch"` when `inactive_mode="relative"`. The training module resolves these
names through `invoke_loss` (see [system/base](../../architecture/system/base.md));
`inactive_labels` is `target.abs().amax(-1) == 0`, one bool per row.

### Target-absent rows: `inactive_sdr_loss`

The ratio is undefined when the reference is silent. When `inactive_labels`
marks any row, `forward` splits the batch: active rows take the formula above,
inactive rows take `inactive_sdr_loss(s1, s2, mode=inactive_mode)`, and the two
per-row results are concatenated before the reduction (so with `reduction=False`
the rows come back active first, then inactive, not in batch order). Inactive
rows are scored, not dropped. Both signals are zero-meaned first.

| `inactive_mode` | per-row value |
| --- | --- |
| `absolute` (default) | `10*log10(sum(s1^2) + 0.01*sum(s2^2) + 1e-8)` |
| `mean` | `10*log10(mean(s1^2) + 0.01*mean(s2^2) + 1e-8)` |
| `relative` | `10*log10(mean(s1^2) + 1e-10) - 10*log10(mean(noisy^2) + 1e-10)`, with `noisy = batch["noisy_speech"]` |

All three are "smaller is better" and push the output toward silence. They
differ in what they reward:

- `absolute` puts the floor on a sum, so in mean-power terms the floor moves
  with row length (by `10*log10(L)`), and a quieter input scores lower without
  any suppression.
- `mean` is the same objective on mean power, so the floor is length-invariant.
- `relative` scores the suppression achieved against the row's own mixture, in
  dB, independent of level and length. It raises `ValueError` if the batch has
  no `noisy_speech`.

### Presets: `init_mode`

`SDRLoss.init_mode(loss_func="sisnr", reduction=True, threshold=None)` builds a
named variant (with `zero_mean=True`, `eps=1e-8`). It is a Python-side
constructor; recipes pass the flags to `SDRLoss` directly. Any other name raises
`NameError`.

| name | `scaled` | `scale_dependent` | `source_aggregated` | `sdr_max` |
|---|---|---|---|---|
| `sisnr` | True | False | False | None |
| `sdsdr` | True | True | False | None |
| `sdr` | False | False | False | None |
| `tsdr` | False | False | False | 30 |
| `tsdr50` | False | False | False | 50 |
| `sasdr` | False | False | True | None |
| `sasisnr` | True | False | True | None |
| `satsdr` | False | False | True | 30 |

### Config usage

```yaml
loss_func:
  - type: SDRLoss
    weighted: 1.
    args:
      scaled: False
      scale_dependent: False
      zero_mean: True
      source_aggregated: False
      sdr_max: 50
      threshold: -50
```

### Design notes

- Level supervision. With `scaled=True` the loss is blind to output gain, so
  the output level is left free. The shipped recipes use `scaled: False` (plain
  SDR) so the level is supervised as well; treat `scaled` and
  `scale_dependent` as a pair and change them together.
- The soft ceiling (`sdr_max`) keeps rows that are already near-perfect from
  dominating the gradient as their residual goes to zero.
- Target-absent rows get their own term because the training data contains
  rows whose correct output is silence, and a ratio against a zero reference has
  no finite value.

## Module-level helpers

Not reachable from `loss_func[].type`.

- `si_snr(s1, s2, eps=1e-8, reduction=True)` — SI-SNR as a metric (positive dB,
  not negated). Used by `puresound/metrics.py` and the evaluation tools.
- `inactive_sdr_loss(s1, s2, reduction=True, mode="absolute", noisy=None)` —
  the target-absent term above.
- `l2_norm(s1, s2)` — `sum(s1 * s2, dim=-1, keepdim=True)`.
- `attenuation_ratio(s1, s2, mask, reduction=True)` — per row, the dB ratio of
  the unprocessed signal's energy (`s2`) to the output's energy (`s1`) over the
  samples where `mask == 0`: how far the output was suppressed where it should
  be silent.
