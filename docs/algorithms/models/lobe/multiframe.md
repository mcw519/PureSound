# puresound.nnet.lobe.multiframe

繁體中文版本：[multiframe.zh-TW.md](multiframe.zh-TW.md)

Multi-frame complex filters. A single-frame mask sets one gain per bin per
frame; a K-tap filter per bin combines that bin's recent frames, which can
resolve harmonic structure a 10 ms hop blurs (Schröter et al., "DeepFilterNet:
A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep
Filtering", ICASSP 2022).

The module holds two groups:

- **Deep-filter residual** — `deep_filter_residual` and
  `DeepFilterResidualHead`, used by `DPCRN(df_head=...)`. Real arithmetic,
  causal, streamable.
- **DeepFilterNet-style operators** — `DeepFilter`, `MultiFrameWienerFilter`,
  `MultiFrameMvdrFilter` and `DeepFilterDecoder`, in complex arithmetic, reached
  through `Masker.apply_df_on_reim` / `apply_wiener` / `apply_mvdr`
  (`mask_type: deepfilter | wiener | mvdr` in `EncDecMaskBase`).

**The two groups use different shape conventions.** The residual path uses
`[N, 2, F, T]` (re/im, frequency, time), the same order as
[`Masker`](../masker.md)'s `[N, 2, C, T]`. The DeepFilterNet-style operators use
`[B, C, T, F, 2]` (time before frequency, re/im last); `Masker` permutes into
that order at every call site.

## Function: `deep_filter_residual`

Causal multi-frame complex filter, per bin:

```
out[t] = sum_{k=0}^{K-1} coefs[k, t] * x[t - k]        (complex multiply)
```

```python
deep_filter_residual(history: Tensor, coefs: Tensor) -> Tensor
```

- `history` – `[N, 2, F, T + K - 1]` re/im spectrum, oldest frame first. The
  offline path left-pads `K - 1` zero frames; the streaming path passes its
  `K - 1` cached frames followed by the current one.
- `coefs` – `[N, K, 2, F, T]`; tap `k` multiplies frame `t - k`.
- Returns `[N, 2, F, T]`. Raises `ValueError` if `history` does not hold
  `T + K - 1` frames.

It is written with re/im pairs rather than complex tensors, so the same
function is the offline op and the per-frame streaming op, and it exports to
ONNX unchanged.

## Class: `DeepFilterResidualHead`

Predicts K-tap coefficients for the lowest `bins` fine bins from a decoder
feature map. The filter output is **added** to the complex-masked spectrum:
`Y = M * X + DF(X)` on the low band, where `DF` filters the noisy spectrum `X`.

```python
DeepFilterResidualHead(
    in_channels: int,     # channels of the tapped decoder feature map
    upsample: int,        # fine bins per tapped (coarse) bin
    bins: int = 128,      # fine bins filtered, from the bottom; a multiple of upsample
    order: int = 5,       # taps K: current frame plus K - 1 past frames
    hidden: int = 32,     # width of the pointwise projection
)
```

Structure: `Conv2d 1x1 -> PReLU -> Conv2d 1x1 (zero-initialised) -> tanh` over
the first `bins // upsample` coarse bins. The last conv emits
`upsample * 2 * order` channels, reshaped so that fine bin
`f = coarse * upsample + u`.

`forward(x [N, C, F_coarse, T]) -> [N, K, 2, bins, T]`, bounded to `(-1, 1)` by
`tanh`. Raises `ValueError` if `bins % upsample != 0`, `order < 1`, or the
feature map has fewer than `bins // upsample` bins.

### Use in DPCRN

```yaml
backbone_args:
  df_head: {bins: 128, order: 5, hidden: 32}
```

`DPCRN` builds the head with `in_channels = channels[1]` and
`upsample = stride_f[0]`, runs it on the decoder's second-to-last feature map,
and stores the result as `backbone.last_df_coefs`. `EncDecMaskBase` with
`mask_type: complex` passes it to `Masker.apply_complex_mask_with_df`; a
backbone without the head leaves it `None`, which gives the plain complex mask.
The streaming runner (`puresound/streaming/dpcrn.py`) keeps the `K - 1` most
recent noisy frames of the low band as extra state.

### Design notes

- **Residual, not replacement.** The last projection starts at zero, so at
  initialisation the model computes exactly what the mask-only network
  computes, and a checkpoint of that network warm-starts this one without
  re-initialising anything.
- **Stateless head.** Only pointwise convolutions, no time kernel, so the only
  new streaming state is the `K - 1` frames of noisy spectrum the filter reads.
- **Reads the noisy spectrum**, as DeepFilterNet's filter does, not the masked
  one.

Tests: `test/nnet/test_multiframe.py` pins the filter, its causality, the
zero start and the mask-only warm start.

---

## Class: `MultiFrameModule`

Base class of the DeepFilterNet-style operators: pads and unfolds the time axis
into windows of `frame_size` frames.

```python
MultiFrameModule(num_freqs: int, frame_size: int, lookahead: int, real: bool = False)
```

- `num_freqs` – bins the filter is applied to, from the bottom; higher bins pass
  through unchanged.
- `frame_size` – window length N in frames.
- `lookahead` – how many of the N frames are future frames
  (`frame_size - 1 - lookahead` are past); `0` is causal. No default.
- `real` – use `spec_unfold_real` (real input with an extra trailing axis)
  instead of the complex `spec_unfold`. The filters below use the complex path.

Methods:

- `spec_unfold(spec)` – complex `[B, C, T, F]` → `[B, C, T, F, N]`, padding
  `frame_size - 1 - lookahead` frames before and `lookahead` after.
- `solve(Rxx, rss, diag_eps=1e-8, eps=1e-7)` (static) – `Rxx⁻¹ rss` with
  Tikhonov regularisation.
- `apply_coefs(spec, coefs)` (static) – `einsum("...n,...n->...")`: one
  coefficient vector per window.

Module-level helpers:

- `psd(x, n)` – correlation matrix `X Xᴴ` over an `n`-frame window:
  `[B, C, T, F]` → `[B, C, T, F, n, n]`.
- `df(spec, coefs)` – the deep-filter sum-product
  `einsum("...tfn,...ntf->...tf")`, used by `DeepFilter`.
- `_tik_reg(mat, reg=1e-7, eps=1e-8)` – `mat + (trace(mat).real * reg + eps) * I`.

## Class: `DeepFilter`

Applies learned complex coefficients directly (DeepFilterNet's operator):
`Y[t, f] = sum_n c_n[t, f] * X[t - N + 1 + lookahead + n, f]`.

```python
DeepFilter(num_freqs: int, frame_size: int, lookahead: int,
           real: bool = False,
           conj: bool = False)   # conjugate the coefficients before applying
```

`forward(spec [B, C, T, F, 2], coefs [B, C*N, T, num_freqs, 2]) -> [B, C, T, F, 2]`.
Only the first `num_freqs` bins are replaced.

## Class: `MultiFrameWienerFilter`

Multi-frame Wiener filter from the noisy correlation matrix `Rxx` and the speech
inter-frame correlation (IFC) vector `rss` (Huang and Benesty, "A Multi-Frame
Approach to the Frequency-Domain Single-Channel Noise Reduction Problem", IEEE
TASLP 2012):

```
w = Rxx⁻¹ rss,        Y[t, f] = sum_n w_n[t, f] * X_window[t, f, n]
```

```python
MultiFrameWienerFilter(
    num_freqs: int,
    frame_size: int,
    lookahead: int,
    cholesky_decomp: bool = False,   # covariance input is a Cholesky factor L
    inverse: bool = True,            # covariance input is already an inverse
    enforce_constraints: bool = True,
    eps: float = 1e-8,
    dload: float = 1e-7,             # diagonal loading when inverse=False
)
```

`forward(spec [B, 1, T, F, 2], ifc [B, T, num_freqs, N*2], iRxx [B, T, num_freqs, N*N*2]) -> [B, 1, T, F, 2]`.

## Class: `MultiFrameMvdrFilter`

Multi-frame MVDR filter, distortionless towards the current frame's speech
component, from the noise correlation matrix `Rnn`:

```
w = Rnn⁻¹ r · conj(r_cur) / (rᴴ Rnn⁻¹ r)
```

This equals `Rnn⁻¹ γ / (γᴴ Rnn⁻¹ γ)` with the normalised IFC `γ = r / r_cur`,
where `r_cur` is the window's last element (the current frame when
`lookahead=0`). Same constructor as `MultiFrameWienerFilter`;
`forward(spec, ifc, iRnn)` takes the same shapes.

### Covariance input forms (both filters)

| `cholesky_decomp` | `inverse` | the network predicts | the filter |
|---|---|---|---|
| False | True | `R⁻¹` | multiplies |
| True | True | `L` with `R⁻¹ = L Lᴴ` | rebuilds, multiplies |
| False | False | `R` | loads the diagonal, `linalg.solve` |
| True | False | `L` with `R = L Lᴴ` | rebuilds, loads the diagonal, solves |

`enforce_constraints` zeroes the strict upper triangle of a Cholesky input, or
makes a plain `R` Hermitian (real diagonal, upper triangle = conjugate of the
lower). `get_r_factor()` returns a constant `f` per input form such that the
predicted matrix divided by `f` lies roughly in `[-1, 1]`, for scaling a network
output. `Masker.apply_wiener` uses the default form (`R⁻¹`);
`Masker.apply_mvdr` uses the Cholesky factor of `R⁻¹`.

## Class: `DeepFilterDecoder`

Predicts the IFC vector and covariance for the two filters above from a
bottleneck embedding and the complex spectrogram: a
[`SqueezedGRU`](group_op.md) over the embedding feeds two
[`GroupedLinear`](group_op.md) outputs, and a separable-conv pathway over the
spectrogram is added to each.

```python
DeepFilterDecoder(
    df_bins: int,                       # bins to predict for
    cpx_in_channels: int,               # channels of the spectrogram input
    emb_in_dim: int,                    # embedding width
    emb_hid_dim: int = 256,             # SqueezedGRU hidden size
    df_order: int = 3,                  # taps N
    df_n_layer: int = 3,                # SqueezedGRU layers
    df_pathway_kernel_size_t: int = 1,  # time kernel of the conv pathways
    df_lin_groups: int = 1,             # groups of the output GroupedLinear
)
```

`forward(emb [N, emb_in_dim, T], c0 [N, CH, df_bins, T]) -> (ifc [N, df_bins, T, df_order*2], cov [N, df_bins, T, df_order**2*2])`.
The outputs are frequency-before-time; `Masker.apply_wiener` /
`apply_mvdr` permute them into the filters' `[B, T, F, ...]` order.
`mask_type: wiener | mvdr` needs a backbone that returns `(mask, ifc, cov)`;
no backbone in `puresound.nnet` does.

## Wiring through `Masker`

```python
# Masker.apply_wiener, abbreviated
tf_rep = tf_rep.permute(0, 3, 2, 1).unsqueeze(1)   # [N, 2, C, T] -> [N, 1, T, C, 2]
est_ifc = est_ifc.permute(0, 2, 1, 3)              # [N, C, T, *] -> [N, T, C, *]
est_cov = est_cov.permute(0, 2, 1, 3)
wiener = MultiFrameWienerFilter(num_freqs=nbins, frame_size=order, lookahead=0)
enh = wiener(tf_rep, est_ifc, est_cov)
```

`Masker.apply_df_on_reim` does the same for `DeepFilter`, and
`Masker.apply_complex_mask_with_df` applies the residual path.

## Example

```python
from puresound.nnet.lobe.multiframe import DeepFilter

deepfilter = DeepFilter(num_freqs=257, frame_size=3, lookahead=0)
spec = torch.rand(2, 1, 100, 257, 2)     # [B, C, T, F, 2]
coefs = torch.rand(2, 3, 100, 257, 2)    # [B, N, T, F, 2]
enh = deepfilter(spec, coefs)            # [B, C, T, F, 2]
```
