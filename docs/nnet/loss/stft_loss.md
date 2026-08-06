# puresound.nnet.loss.stft_loss

繁體中文版本：[stft_loss.zh-TW.md](stft_loss.zh-TW.md)

STFT-domain losses for spectral fidelity in speech enhancement. Only
`MultiResolutionSTFTLoss`, `SpectralLoss`, and `OverSuppressionLoss` are
imported into `puresound/nnet/loss/__init__.py` and reachable via a recipe's
`loss_func[].type`. `STFTLoss`, `SpectralConvergengeLoss`,
`LogSTFTMagnitudeLoss`, and the module-level `stft` / `as_complex` / `angle`
helpers are internal building blocks — reachable only via
`puresound.nnet.loss.stft_loss.X`, not through the package's top level.

## Helper: `stft(x, fft_size, hop_size, win_length, window) -> Tensor`

Internal helper used by `STFTLoss` and `OverSuppressionLoss`: runs
`torch.stft(..., return_complex=True)` and returns the **magnitude**
spectrogram, transposed to `[B, #frames, fft_size // 2 + 1]`. `window` is an
already-materialized window tensor (e.g. `torch.hann_window(win_length)`),
moved to `x`'s device on every call.

## Helper: `as_complex(x: Tensor) -> Tensor`

Takes a **single** tensor (not a `(re, im)` pair): if `x` is already complex,
returns it unchanged; otherwise requires `x.shape[-1] == 2` (real and
imaginary parts stacked in the last axis, the convention used elsewhere in
this codebase for "complex-as-real-pair" tensors) and converts via
`torch.view_as_complex`. Used by `SpectralLoss` to accept either
representation.

## Helper: `angle` (`torch.autograd.Function`)

Phase angle with a numerically-stable gradient, invoked as **`angle.apply(x)`**
— not `angle(x)` (`angle` is a `torch.autograd.Function` subclass; the static
`forward`/`backward` pair must be called through `.apply`). Forward is plain
`atan2(x.imag, x.real)`; backward clamps the `1 / |x|^2` term in the chain
rule to `1e-10` so the gradient does not blow up for near-zero-magnitude bins
— the problem with differentiating `torch.angle` directly. Used inside
`SpectralLoss` (see below) when it needs to rebuild a gamma-compressed complex
tensor from a compressed magnitude and the original phase.

## Class: `SpectralConvergengeLoss`

Frobenius-norm spectral divergence between magnitude spectrograms.

$$\mathcal{L}_{sc} = \frac{\||\hat{S}| - |S|\|_F}{\||S|\|_F}$$

`forward(x_mag, y_mag) -> Tensor` — `x_mag` = predicted, `y_mag` = reference.

## Class: `LogSTFTMagnitudeLoss`

L1 loss on log-compressed magnitude spectra.

$$\mathcal{L}_{log} = \| \log|\hat{S}| - \log|S| \|_1$$

`forward(x_mag, y_mag) -> Tensor`.

## Class: `STFTLoss`

Single-resolution building block combining the two losses above at one
`(fft_size, hop, win_length)` setting.

```python
STFTLoss(
    fft_size=1024,
    shift_size=120,
    win_length=600,
    window="hann_window",   # any torch.*_window factory name, resolved via getattr(torch, window)
)
```

`forward(x, y) -> Tuple[Tensor, Tensor]` — **returns a 2-tuple**
`(sc_loss, mag_loss)`, not a single scalar; the caller (`MultiResolutionSTFTLoss`
below) is responsible for weighting and summing the two. Not imported into
the package `__init__.py`, so `type: STFTLoss` is not a valid `loss_func[].type`
— it only exists as a component `MultiResolutionSTFTLoss` builds one instance
of per resolution.

## Class: `MultiResolutionSTFTLoss`

Sums `STFTLoss` over several resolutions (a coarse resolution's wide frequency
bins catch broadband distortion, a fine resolution's narrow bins catch
harmonic/pitch structure).

### Constructor

```python
MultiResolutionSTFTLoss(
    fft_sizes=[1024, 2048, 512],
    hop_sizes=[120, 240, 50],
    win_lengths=[600, 1200, 240],
    window="hann_window",
    factor_sc=0.1,
    factor_mag=0.1,
)
```

Builds one `STFTLoss(fs, ss, wl, window)` per `zip(fft_sizes, hop_sizes,
win_lengths)` triple (all three lists must be the same length).
`factor_sc` / `factor_mag` weight the (mean-over-resolutions) spectral
convergence and log-magnitude terms in the final sum.

`uses_inactive_labels = True` — set for the same reason as `SDRLoss` (see
[loss/sdr](sdr.md)), but handled differently: spectral losses are undefined
against an all-zero reference (spectral convergence divides by `||ref||`,
log-magnitude hits the eps clamp), so a single silent-target row can produce a
loss orders of magnitude above the active-row level and its gradient drowns
the rest of the batch. Rather than scoring silent rows with a substitute
formula the way `SDRLoss` does, `MultiResolutionSTFTLoss` simply **drops**
them.

### `forward(x, y, inactive_labels=None) -> Tensor`

```python
if inactive_labels is None:
    active = y.abs().amax(dim=-1) > 0        # self-sufficient fallback
else:
    active = ~inactive_labels.to(torch.bool).reshape(-1)
if not bool(active.any()):
    return x.new_zeros(())                   # see note below
x, y = x[active], y[active]
```

If no `inactive_labels` is supplied, the loss derives its own active-row mask
from `y` directly, so it can be used standalone (without the training system
routing labels in). If literally every row in the batch is inactive, it
returns a plain `x.new_zeros(())` — unlike `DistHeadRegressionLoss`'s
`dist_preds.sum() * 0.0` (see [loss/dist](dist.md)), this zero is **not**
connected back to `x` in the autograd graph. In practice this only matters if
`MultiResolutionSTFTLoss` were the only loss term in a recipe; other loss
terms computed on the same `enhanced` tensor in the same step still keep the
backbone connected.

### Config usage

```yaml
# egs/voice_isolate/config/train_dpcrn.yaml (the active recipe)
- type: MultiResolutionSTFTLoss
  weighted: 0.5
  args:
    fft_sizes: [512, 1024, 256]
    hop_sizes: [128, 256, 64]
    win_lengths: [512, 1024, 256]
    window: hann_window
    factor_sc: 0.5
    factor_mag: 0.5
```

## Class: `OverSuppressionLoss`

The one-sided anti-deletion pressure referenced from [loss/index](index.md):
penalizes the enhanced magnitude falling **below** the reference, and applies
**zero** penalty when it's above (over-estimation/leakage is already covered
elsewhere — see *Config usage*). A commented-out free-function prototype of
the same idea (`over_suppression_loss`) is kept directly above this class in
the source as a historical reference; it is not live code.

### Constructor

```python
OverSuppressionLoss(
    p: float = 0.5,          # power-law compression exponent
    fft_size: int = 512,
    hop_size: int = 128,
    win_length: int = 512,   # fixed Hann window; unlike STFTLoss, not configurable
)
```

### `forward(enh, ref) -> Tensor`

```python
enh_mag = stft(enh, fft_size, hop_size, win_length, hann_window)
ref_mag = stft(ref, fft_size, hop_size, win_length, hann_window)
loss = ref_mag.pow(p) - enh_mag.pow(p)
loss = loss.clamp_min(0) ** 2      # (mask > 0, then square) -- zero wherever enh_mag >= ref_mag
return loss.mean()
```

Wherever the (power-`p`-compressed) reference magnitude exceeds the enhanced
magnitude — deletion — the squared gap is penalized; wherever the enhanced
magnitude is equal or higher, the term is masked to zero. `p=0.5` (the
default, and the value the active recipe uses) is a square-root compression
that de-emphasizes the loudest bins relative to quieter ones.

### Config usage

```yaml
# egs/voice_isolate/config/train_dpcrn.yaml
# ANTI-SUPPRESSION: pure-magnitude one-sided loss = mean(ReLU(|T|^p - |E|^p)^2).
# Penalises enhanced magnitude falling BELOW target (deletion); over-estimation is
# already covered by SD-SDR + STFT above, so this adds net asymmetric pressure toward
# preservation. `weighted` trades deletions against substitutions.
- type: OverSuppressionLoss
  weighted: 3.0
  args: {p: 0.5, fft_size: 512, hop_size: 128, win_length: 512}
```

## Class: `SpectralLoss`

Unlike every other class on this page, `SpectralLoss` does **not** take any
STFT configuration (`n_fft` / `hop_length` / `win_length`) — it operates on
spectral tensors the caller has **already computed**, not on waveforms.

### Constructor

```python
SpectralLoss(
    gamma: float = 1,             # power-law magnitude compression exponent
    factor_magnitude: float = 1,  # weight on the magnitude term
    factor_complex: float = 1,    # weight on the real+imag (phase-sensitive) term
    factor_under: float = 1,      # extra weight where enhanced undershoots reference
)
```

### `forward(enh, ref) -> Tensor`

```
enh, ref: Tensor | ComplexTensor, shape [N, *, C, T, 2] or [N, *, C, T]
```

1. Coerce both to complex via `as_complex` (accepts a native complex tensor or
   a real tensor with a trailing size-2 real/imag axis).
2. Take magnitudes; if `gamma != 1`, compress both by `.clamp_min(1e-12).pow(gamma)`.
3. Magnitude term: mean squared `(enh_abs - ref_abs)`, weighted by
   `factor_magnitude`. If `factor_under != 1`, bins where the enhanced
   magnitude undershoots the reference (`enh_abs < ref_abs` — the same
   deletion condition `OverSuppressionLoss` targets) get an extra
   `factor_under` multiplier — a one-sided emphasis folded into a combined
   loss rather than a separate pure-penalty term.
4. If `factor_complex > 0`: rebuild gamma-compressed complex tensors (using
   `angle.apply`, not `torch.angle`, for gradient stability) and add
   `factor_complex * MSE(view_as_real(enh), view_as_real(ref)) `— matching
   real and imaginary parts jointly implicitly constrains phase, not just
   magnitude. Skipped entirely when `factor_complex <= 0`.

Not currently used by any recipe in this repo.
