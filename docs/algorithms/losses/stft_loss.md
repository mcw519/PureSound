# puresound.nnet.loss.stft_loss

繁體中文版本：[stft_loss.zh-TW.md](stft_loss.zh-TW.md)

STFT-domain losses. Three classes are exported for recipes:
`MultiResolutionSTFTLoss` (spectral fidelity over several STFT resolutions),
`OverSuppressionLoss` (one-sided penalty on output magnitude below the target)
and `SpectralLoss` (magnitude and complex error on precomputed spectra).
`STFTLoss`, `SpectralConvergengeLoss`, `LogSTFTMagnitudeLoss` and the helpers
`stft`, `as_complex` and `angle` are internal building blocks, reachable only
as `puresound.nnet.loss.stft_loss.<name>`.

## Helper: `stft`

```python
stft(x, fft_size, hop_size, win_length, window, power_floor=1e-8, relative_floor=False)
# x: [B, T] -> magnitude [B, frames, fft_size // 2 + 1]
```

Runs `torch.stft(..., return_complex=True)` and returns
`sqrt(clamp(|X|^2, min=floor))`. `window` is a window tensor, moved to `x`'s
device on each call.

- `power_floor` is the clamp on `|X|^2`. The default `1e-8` is magnitude `1e-4`,
  about −80 dB relative to a full-scale bin. It is a gradient floor: every bin
  below it is invisible to both STFT terms.
- `relative_floor=True` makes `power_floor` a ratio to each row's own peak bin
  power (`floor = power_floor * max |X|^2` per row), so the floor follows the
  row's level instead of sitting at a fixed absolute value.

## Class: `STFTLoss`

One resolution: spectral convergence plus log-magnitude L1 on the magnitudes
from `stft`.

$$\mathcal{L}_{sc} = \frac{\||S| - |\hat{S}|\|_F}{\||S|\|_F}, \qquad \mathcal{L}_{mag} = \operatorname{mean}\left|\ln|S| - \ln|\hat{S}|\right|$$

$\hat S$ is the prediction, $S$ the reference. The Frobenius norms run over the
whole batch tensor, not per row.

```python
STFTLoss(fft_size=1024, shift_size=120, win_length=600, window="hann_window",
         power_floor=1e-8, relative_floor=False)
# window: a torch.*_window factory name, resolved with getattr(torch, window)
```

`forward(x, y) -> (sc_loss, mag_loss)` returns a pair; the caller weights and
sums them. `SpectralConvergengeLoss` and `LogSTFTMagnitudeLoss` are the two
terms as separate modules, each `forward(x_mag, y_mag) -> Tensor`.

## Class: `MultiResolutionSTFTLoss`

One `STFTLoss` per `(fft_size, hop_size, win_length)` triple, each term averaged
over resolutions:

$$\mathcal{L} = f_{sc} \cdot \frac{1}{R}\sum_r \mathcal{L}_{sc}^{(r)} + f_{mag} \cdot \frac{1}{R}\sum_r \mathcal{L}_{mag}^{(r)}$$

### Constructor

```python
MultiResolutionSTFTLoss(
    fft_sizes=[1024, 2048, 512],
    hop_sizes=[120, 240, 50],
    win_lengths=[600, 1200, 240],   # the three lists must have equal length
    window="hann_window",
    factor_sc=0.1,                  # f_sc
    factor_mag=0.1,                 # f_mag
    power_floor=1e-8,               # passed to every resolution, see `stft`
    relative_floor=False,
)
```

### Inputs

`forward(x, y, inactive_labels=None) -> Tensor`, waveforms `[B, T]`.
`required_inputs = ("enhanced", "target", "inactive_labels")`.

Rows marked inactive (silent reference) are dropped before any STFT. Without
`inactive_labels` the loss derives the mask itself (`y.abs().amax(-1) > 0`), so
it also works when called directly. If every row is inactive it returns
`x.new_zeros(())`, a zero with no graph edge to `x`.

### Config usage

```yaml
loss_func:
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

### Design notes

- Several resolutions because no single STFT sees both time and frequency
  detail: long windows resolve harmonics, short windows resolve onsets and
  transients.
- The two terms weight errors differently: spectral convergence works on
  linear magnitude and is dominated by loud bins; log-magnitude L1 counts a
  relative error equally across the dynamic range.
- Silent-reference rows are dropped rather than scored: spectral convergence
  divides by `||ref||`, and log-magnitude against a reference at the floor
  yields a loss far above the active-row level whose gradient would swamp the
  batch. Those rows are supervised by the target-absent term of
  [`SDRLoss`](sdr.md) and by the VAD losses.
- The term's scale next to the SDR term is set by `factor_sc`, `factor_mag` and
  the entry's `weighted`; with small factors it is nearly constant beside SDR
  and contributes little gradient.

## Class: `OverSuppressionLoss`

One-sided magnitude penalty: only bins where the output falls below the
reference count.

$$\mathcal{L} = \operatorname{mean}\left(\max\left(0,\ |S|^{p} - |\hat{S}|^{p}\right)^2\right)$$

### Constructor

```python
OverSuppressionLoss(
    p: float = 0.5,          # power-law compression exponent
    fft_size: int = 512,
    hop_size: int = 128,
    win_length: int = 512,   # Hann window of this length; the window type is fixed
)
```

`forward(enh, ref) -> Tensor`, waveforms `[B, T]`. It declares no
`required_inputs`, so it receives `("enhanced", "target")`. It uses `stft` with
the default absolute floor.

### Config usage

```yaml
loss_func:
  - type: OverSuppressionLoss
    weighted: 1.
    args: {p: 0.5, fft_size: 512, hop_size: 128, win_length: 512}
```

### Design notes

The symmetric losses (SDR, MR-STFT) charge over- and under-estimation alike.
Deletion of target speech is the costlier error for downstream ASR, so this
term adds pressure in one direction only: toward keeping target energy.
Over-estimation stays covered by the symmetric terms. `p = 0.5` compresses the
magnitudes so the loudest bins do not dominate the penalty. The entry's
`weighted` sets the trade between deletion and residual noise.

## Class: `SpectralLoss`

Magnitude and complex-spectrum MSE on spectra the caller has already computed.
It takes no STFT settings.

### Constructor

```python
SpectralLoss(
    gamma: float = 1,             # power-law compression of the magnitudes
    factor_magnitude: float = 1,  # weight of the magnitude term
    factor_complex: float = 1,    # weight of the complex term; <= 0 disables it
    factor_under: float = 1,      # extra weight on bins where |enh| < |ref|
)
```

### `forward(enh, ref) -> Tensor`

`enh`, `ref`: complex tensors `[N, *, C, T]`, or real tensors with a trailing
real/imaginary axis `[N, *, C, T, 2]` (converted by `as_complex`).

```
A_e, A_r = |enh|, |ref|                        # ** gamma if gamma != 1 (clamped at 1e-12)
mag      = mean((A_e - A_r)^2 * w) * factor_magnitude,   w = factor_under where A_e < A_r, else 1
cplx     = MSE(view_as_real(A_e * e^{j angle(enh)}), view_as_real(A_r * e^{j angle(ref)})) * factor_complex
loss     = mag + cplx                          # cplx only when factor_complex > 0
```

With `gamma == 1` the complex term compares `enh` and `ref` directly. The
phase is taken with `angle.apply`, a `torch.autograd.Function` whose backward
clamps `1 / |x|^2` at `1e-10`, so near-zero bins do not produce exploding
phase gradients.

The training module hands losses waveforms, so `SpectralLoss` is meant for code
that computes the spectra itself; configured as a `loss_func` entry it would
receive waveforms and fail in `as_complex`.

### Design notes

Power-law compression (`gamma < 1`) balances loud and quiet bins; the complex
term constrains phase as well as magnitude; `factor_under` folds a one-sided
emphasis on undershoot into the same loss.
