# puresound.nnet.loss.sdr

繁體中文版本：[sdr.zh-TW.md](sdr.zh-TW.md)

Time-domain SDR / SI-SNR family loss. Regressing waveforms directly with L1/MSE
lets a model "win" by matching the reference's overall level rather than its
shape, which is not the thing speech enhancement/separation actually cares
about. `SDRLoss` normalizes that away: `zero_mean` removes DC offset and
`scaled` (see below) projects the reference onto the estimate's optimal scale
before measuring the residual, so the loss is invariant to a constant gain
mismatch between `s1` and `s2`. The `scaled` / `scale_dependent` /
`source_aggregated` / `sdr_max` flags below select the SI-SNR / SD-SDR / SA-SDR
/ t-SDR family of variants out of one implementation.

## Class: `SDRLoss`

### Constructor

```python
SDRLoss(
    scaled: bool = True,
    scale_dependent: bool = False,
    zero_mean: bool = True,
    source_aggregated: bool = False,
    sdr_max: int = None,
    eps: float = 1e-8,
    reduction: bool = True,
    threshold: Optional[float] = None,
)
```

**Parameters:**
- `scaled` – if `True`, project the reference onto the estimate's best-fit
  scale before computing the residual (see `forward` below). This is what
  makes the loss scale-*invariant* (the "SI" in SI-SNR).
- `scale_dependent` – if `True`, measure the residual against the raw
  (unscaled) reference instead of the scaled one. Only has an effect when
  `scaled=True` — see the note under `forward`.
- `zero_mean` – remove the per-utterance mean (DC offset) from both signals
  before anything else. Required for the projection to mean "shape", not
  "shape + a constant".
- `source_aggregated` – aggregate energies across a source axis before taking
  the ratio (SA-SDR). Needs a 3-D input (`[batch, num_sources, length]`)
  instead of the usual 2-D `[batch, length]` — enforced by `check_input_shape`.
- `sdr_max` – if not `None`, a soft ceiling ("t-SDR"): adds `tau * target_norm`
  (`tau = 10 ** (-sdr_max / 10)`) to the noise energy before the ratio, so the
  reported SDR asymptotically saturates near `sdr_max` dB instead of diverging
  as the residual goes to zero.
- `eps` – division-by-zero guard.
- `reduction` – **a bool, not a string.** `True` returns `torch.mean(...)` (a
  scalar); `False` returns the un-reduced per-row loss tensor.
- `threshold` – hard-example filter; see below.

`SDRLoss.__init__` also sets `self.uses_inactive_labels = True`. That flag is
the dispatch hook `EncDecMaskBase.compute_loss` (`puresound/system/siso.py`)
looks for: any loss with `uses_inactive_labels = True` gets called with an
extra `inactive_labels` tensor derived as `target.abs().amax(dim=-1) == 0` —
i.e. rows whose reference is fully silent (target-absent training rows). See
*Inactive rows* below for what `SDRLoss` does with it.

### `forward(s1, s2, inactive_labels=None) -> Tensor`

- `s1` – the estimated/enhanced signal, `[batch, length]`, or
  `[batch, num_sources, length]` if `source_aggregated=True`.
- `s2` – the reference (clean target) signal, same shape as `s1`.
- `inactive_labels` – optional bool tensor `[batch]`; see *Inactive rows*.

Per (active) row, with `s1`/`s2` after the optional zero-mean step:

```
s_target = <s1, s2> / (<s2, s2> + eps) * s2   if scaled else s2
e_noise  = s1 - s2                             if scale_dependent else s1 - s_target
target_norm = <s_target, s_target>
noise_norm  = <e_noise, e_noise> [+ tau * target_norm, if sdr_max is set]
snr = -10 * log10(target_norm / (noise_norm + eps) + eps)     # negated: this is a loss
```

(`<a, b>` above is the module's `l2_norm(a, b)` helper — despite the name it's
just `sum(a * b, dim=-1, keepdim=True)`, an inner product; it only becomes a
squared L2 norm when called as `l2_norm(x, x)`.) With `source_aggregated=True`
the same formula runs on energies summed over the source axis first
(`target_norm.sum(dim=-1)` / `noise_norm.sum(dim=-1)`) instead of per-source.

**`scaled` vs `scale_dependent`:** `scale_dependent` only changes anything when
`scaled=True`. When `scaled=False`, `s_target` already equals `s2`, so
`e_noise` is `s1 - s2` regardless of `scale_dependent`'s value — the flag is
inert. This matters when reading a config: `egs/voice_isolate/config/train_dpcrn.yaml`
sets `scaled: False, scale_dependent: True`, and per this rule the
`scale_dependent: True` there has no effect; the behavior is determined by
`scaled: False` alone (see *Config usage*).

**Returns:** the negative SDR/SI-SNR metric (minimizing the loss maximizes the
metric), reduced to a scalar mean if `reduction=True`, else the per-row tensor.

### Inactive rows: `inactive_sdr_loss`

Plain SDR is undefined on a row whose reference is silent — `target_norm` is
(numerically) zero, so `10 * log10(target_norm / noise_norm)` diverges to
`-inf`. Rather than special-casing that inside the main formula, `forward`
splits the batch when `inactive_labels` marks at least one row: active rows go
through the pipeline above; inactive rows are scored by the module-level
`inactive_sdr_loss(s1, s2, reduction=False)` function instead, and the two
per-row results are concatenated (`torch.cat([snr, inactive_loss], dim=0)`)
before the top-level `reduction` is applied.

**This is not exclusion** — a previous version of this doc described inactive
rows as excluded from the loss; they are not. They are scored by a different
formula and folded back into the same mean.

```python
def inactive_sdr_loss(s1, s2, reduction=True):
    # zero-mean, then:
    return 10 * log10(<s1, s1> + 0.01 * <s2, s2> + eps)
```

Since `s2` is silent by construction on these rows, this is effectively a
direct penalty on the enhanced signal's own energy `<s1, s1>` — minimizing it
pushes the model toward silence exactly where it should be silent, instead of
computing the undefined ratio. `0.01 * <s2, s2>` is a floor that keeps the log
finite even when both signals are exactly zero. Note this value is **not**
negated like the active-row `snr` — it is already framed as "smaller is
better" (less leaked energy), so it combines correctly with the negated active
term under the same `mean`.

### Hard-example threshold (`threshold`)

When set, rows whose per-row loss is already **below** `threshold` are dropped
before reduction:

```python
snr_to_keep = snr[snr > self.threshold]
if snr_to_keep.nelement() > 0:
    snr = snr_to_keep.view(-1, 1)
```

i.e. gradient is only spent on rows still worth pushing on. If *every* row
already clears the threshold, `snr_to_keep` is empty and the filter is skipped
— `forward` falls back to the full, unfiltered batch rather than reducing an
empty tensor.

### Common variants and `init_mode`

```python
@classmethod
def init_mode(cls, loss_func: str = "sisnr", reduction: bool = True, threshold: Optional[float] = None) -> "SDRLoss"
```

A named-preset alternate constructor. It is **not** reachable via a recipe's
`loss_func[].type` (only the raw `SDRLoss` class is importable from the
package; `init_mode` is a method on it, not a top-level name), and no recipe
in this repo calls it — grep finds it referenced only inside `sdr.py` itself.
Recipes construct `SDRLoss(...)` directly with explicit keyword arguments
instead (see *Config usage*). The preset table, traced from the current source:

| `loss_func` name | `scaled` | `scale_dependent` | `source_aggregated` | `sdr_max` |
|---|---|---|---|---|
| `sisnr` | `True` | `False` | `False` | `None` |
| `sdsdr` | `True` | `True` | `False` | `None` |
| `sdr` | `True` ¹ | `False` | `False` | `None` |
| `tsdr` | `False` | `False` | `False` | `30` |
| `tsdr50` | `False` | `False` | `False` | `50` |
| `sasdr` | `False` | `False` | `True` | `None` |
| `sasisnr` | `False` ¹ | `False` | `True` | `None` |
| `satsdr` | `False` | `False` | `True` | `30` |

¹ **Known quirk, tracked separately** — not fixed by this doc pass. `init_mode`
decides `scaled` with:

```python
if loss_func == "sisnr" or loss_func in "sdsdr" or loss_func == "sasisdr":
    scaled = True
```

Two things here don't do what the names suggest, and both trace back to this
one line:

- `loss_func in "sdsdr"` is a **reversed containment check**: Python reads it
  as "is the string `loss_func` a substring of the literal `'sdsdr'`", not
  "does `loss_func` equal `'sdsdr'`". Since `"sdr"` happens to be a substring
  of `"sdsdr"` (`"sdsdr"[2:5] == "sdr"`), the `sdr` preset silently gets
  `scaled=True` too — so today `SDRLoss.init_mode("sdr")` and
  `SDRLoss.init_mode("sisnr")` build identical objects instead of `"sdr"`
  being a plain, non-scale-invariant SDR.
- The third clause compares against the literal `"sasisdr"`, not the real mode
  name `"sasisnr"` (the `d`/`n` are transposed). `"sasisdr"` was never a valid
  `loss_func` value — it isn't in the allow-list `init_mode` checks first — so
  this clause can never fire and is dead code for the mode it was presumably
  meant to catch. Net effect: `SDRLoss.init_mode("sasisnr")` gets
  `scaled=False` rather than the `True` its "SI" implies, making it currently
  identical to `SDRLoss.init_mode("sasdr")`.

This only affects the `init_mode(...)` convenience constructor described in
this section; it is documented here so the preset table above isn't a
mystery, not as a description of intended design.

### Config usage

```yaml
# egs/voice_isolate/config/train_dpcrn.yaml (the active recipe)
loss_func:
  - type: SDRLoss
    weighted: 1.
    args:
      scaled: False
      scale_dependent: True   # inert here -- scaled is False, see note above
      zero_mean: True
      source_aggregated: False
      sdr_max: 50              # soft ceiling ~50 dB
      threshold: -50           # stop pushing on rows already past that ceiling
```

Built with the raw constructor and explicit kwargs, a `(scaled=False,
scale_dependent=True)` combination that no `init_mode` preset above produces —
this recipe's behavior does not depend on the `init_mode` auto-defaults or
their quirk.

## SI-SNR formula (the `scaled=True, scale_dependent=False` case)

$$\text{loss} = -10 \log_{10} \frac{\|\text{proj}(\hat{s})\|^2}{\|\hat{s} - \text{proj}(\hat{s})\|^2}, \qquad \text{proj}(\hat{s}) = \frac{\langle \hat{s}, s \rangle}{\langle s, s \rangle}\, s$$

where $\hat s$ = `s1` (enhanced) and $s$ = `s2` (reference), both after the
optional zero-mean step.

## Module-level helpers

None of these are reachable via `loss_func[].type` — only `SDRLoss` is
imported into `puresound/nnet/loss/__init__.py`.

- `si_snr(s1, s2, eps=1e-8, reduction=True)` – the same computation as
  `SDRLoss(scaled=True, scale_dependent=False, source_aggregated=False)`, as a
  standalone function. Live code: `puresound/metrics.py` imports it directly
  for the `sisnr` / `sisnr_imp` (SI-SNR improvement over the noisy mixture)
  **evaluation** metrics.
- `inactive_sdr_loss(s1, s2, reduction=True)` – see above.
- `l2_norm(s1, s2)` – the inner-product helper used throughout this module.
- `attenuation_ratio(s1, s2, mask, reduction=True)` – per-utterance dB
  attenuation measured only on the non-target portion of the signal
  (`mask == 0`), i.e. how well the system suppresses where it should be
  silent. Not currently called anywhere else in the repo.

## Example

```python
from puresound.nnet.loss.sdr import SDRLoss

sisnr_loss = SDRLoss(scaled=True, scale_dependent=False, zero_mean=True)
loss = sisnr_loss(enhanced_wav, clean_wav)

# Target-absent rows are scored by inactive_sdr_loss, not dropped:
inactive_labels = clean_wav.abs().amax(dim=-1) == 0
loss = sisnr_loss(enhanced_wav, clean_wav, inactive_labels=inactive_labels)
```
