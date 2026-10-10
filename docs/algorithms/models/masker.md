# puresound.nnet.masker

繁體中文版本：[masker.zh-TW.md](masker.zh-TW.md)

`Masker` applies a backbone's output (a mask, filter taps, or filter
statistics) to the mixture's time-frequency representation. It is a plain
class of `@staticmethod`s, not an `nn.Module`: the multi-frame filters it
wraps (`DeepFilter`, `MultiFrameWienerFilter`, `MultiFrameMvdrFilter`, see
[lobe/multiframe](lobe/multiframe.md)) have no learnable parameters, so each
call builds a fresh one.

**Shape convention.** Complex tensors carry an explicit real/imaginary axis,
`[N, 2, C, T]` (batch, re/im, frequency, time). `C` is the encoder's bin count,
for example 256 after `drop_stft_first_bin` on a 512-point FFT.

## Dispatch by `mask_type`

`EncDecMaskBase.forward` (`puresound/system/siso.py`) chooses the call from the
recipe's `mask_type`:

| `mask_type` | Masker call | backbone output |
| --- | --- | --- |
| `complex` | `apply_complex_mask_with_df` | `[N, 2, C, T]` complex mask (plus `last_df_coefs` when the backbone has a `df_head`) |
| `deepfilter` | `apply_df_on_reim` | `[N, 2*order, F, T]` filter taps |
| `wiener` | `apply_complex_mask_on_reim`, then `apply_wiener` on the low band | `(mask, ifc, cov)` |
| `mvdr` | `apply_complex_mask_on_reim`, then `apply_mvdr` on the low band | `(mask, ifc, cov)` |
| `mapping` | none; the output is the enhanced spectrum | `[N, 2, C, T]` |

For `deepfilter`, `F = mask.shape[2]` and `order = mask.shape[1] / 2`: the
output shape is the contract, with no separate config key. The streaming ports
(`puresound/streaming/dpcrn.py`, `dparn.py`) call `apply_complex_mask_on_reim`
per frame; the DPCRN port adds the `df_head` residual itself from a cache of
past frames. None of the backbones
in this library returns the `(mask, ifc, cov)` triple, so `wiener` and `mvdr`
are wiring for a backbone that predicts filter statistics.

## Methods

### `apply_complex_mask_on_reim(tf_rep, est_masks, postfilter=False)`

```python
apply_complex_mask_on_reim(
    tf_rep: Tensor,           # X, [N, 2, C, T]
    est_masks: Tensor,        # M, [N, 2, C, T]
    postfilter: bool = False, # envelope post-filter on M first
) -> Tensor                   # [N, 2, C, T]
```

Complex multiplication `Y = M X` in real arithmetic:
`Y_re = X_re M_re - X_im M_im`, `Y_im = X_re M_im + X_im M_re`.

### `apply_complex_mask_with_df(tf_rep, est_masks, df_coefs=None)`

```python
apply_complex_mask_with_df(
    tf_rep: Tensor,            # X, [N, 2, C, T]
    est_masks: Tensor,         # M, [N, 2, C, T]
    df_coefs: Tensor = None,   # w, [N, K, 2, F_df, T] from DeepFilterResidualHead
) -> Tensor                    # [N, 2, C, T]
```

`M X` everywhere, plus a causal deep-filter residual on the lowest `F_df` bins:

```
Y[f, t] = M[f, t] X[f, t] + sum_{k=0}^{K-1} w_k[f, t] X[f, t - k]    for f < F_df
```

The residual reads the noisy spectrum, zero-padded `K - 1` frames into the
past, through `deep_filter_residual`, which is also the per-frame streaming op.
With `df_coefs=None` the result is exactly `apply_complex_mask_on_reim`. See
the `df_head` option of [DPCRN](dpcrn.md).

### `apply_df_on_reim(tf_rep, est_masks, num_feats, order)`

```python
apply_df_on_reim(
    tf_rep: Tensor,     # [N, 2, C, T]
    est_masks: Tensor,  # [N, 2*order, num_feats, T]; channel 2k + {0, 1} = re/im of tap k
    num_feats: int,     # bins filtered, from the bottom (<= C)
    order: int,         # taps K
) -> Tensor             # [N, 2, C, T]
```

Deep filtering (Mack and Habets, "Deep Filtering: Signal Extraction and
Reconstruction Using Complex Time-Frequency Filters", IEEE SPL 2020) with
`DeepFilter(num_freqs=num_feats, frame_size=order, lookahead=0)`: each of the
lowest `num_feats` bins is replaced by a complex linear combination of its
current and `order - 1` previous frames. Bins above `num_feats` pass through
unchanged.

### `apply_wiener(tf_rep, est_ifc, est_cov, order)` / `apply_mvdr(...)`

```python
apply_wiener(
    tf_rep: Tensor,   # [N, 2, C, T]
    est_ifc: Tensor,  # [N, nbins, T, 2*order]    inter-frame correlation vector
    est_cov: Tensor,  # [N, nbins, T, 2*order^2]  (inverse) correlation matrix
    order: int,       # taps
) -> Tuple[Tensor, int]   # (enhanced [N, 2, C, T], nbins)
```

`nbins = est_cov.shape[1]`. `apply_wiener` builds
`MultiFrameWienerFilter(num_freqs=nbins, frame_size=order, lookahead=0)`;
`apply_mvdr` takes the same arguments and builds `MultiFrameMvdrFilter` with
`enforce_constraints=True, cholesky_decomp=True`. Both filter only the lowest
`nbins` bins across `order` frames and pass the rest through. The caller
splices the filtered low band into the complex-mask estimate:

```python
enh = Masker.apply_complex_mask_on_reim(tf_rep=X, est_masks=mask)
enh_filter, n_bins = Masker.apply_wiener(tf_rep=X, est_ifc=ifc, est_cov=cov, order=n_order)
enh[:, :, :n_bins, :] = enh_filter[:, :, :n_bins, :]
```

with `n_order = ifc.shape[-1] / 2`.

### `envelope_postfiltering_on_cpx_mask(tf_rep, est_masks, tau=0.02)`

```python
envelope_postfiltering_on_cpx_mask(
    tf_rep: Tensor,     # [N, 2, C, T]
    est_masks: Tensor,  # [N, 2, C, T]
    tau: float = 0.02,
) -> Tensor             # [N, 2, C, T], the post-filtered mask
```

```
g  = clamp(|est_masks| / (|tf_rep| + eps), eps, 1)
pf = (1 + tau) / (1 + tau / sin^2(pi * g / 2))
return est_masks * pf
```

`pf` is 1 at `g = 1` and falls towards 0 as `g` shrinks, so small gains are
attenuated further. It runs only through
`apply_complex_mask_on_reim(..., postfilter=True)`; no system passes that flag.

### `apply_real_on_real(tf_rep, est_masks)` / `apply_mag_mask_on_reim(tf_rep, est_masks)`

```python
apply_real_on_real(tf_rep: Tensor, est_masks: Tensor) -> Tensor      # [N, C, T] * [N, C, T]
apply_mag_mask_on_reim(tf_rep: Tensor, est_masks: Tensor) -> Tensor  # [N, 2, C, T] * [N, C, T] -> [N, 2, C, T]
```

Elementwise products for a real mask: on a real representation, or broadcast
over both re/im planes of a complex one (phase unchanged). No `mask_type` in
`EncDecMaskBase` calls them.

## Design notes

- A complex mask changes magnitude and phase together at one frame. Deep
  filtering and the multi-frame filters combine several frames per bin, which
  resolves harmonic detail a single frame blurs; they are limited to the low
  band, where the harmonic part of speech lies.
- The deep-filter residual is added to the mask estimate instead of replacing
  it, so a model with a zero-initialised `df_head` computes exactly the
  mask-only output.
