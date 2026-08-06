# puresound.nnet.masker

繁體中文版本：[masker.zh-TW.md](masker.zh-TW.md)

## Class: `Masker`

A stateless collection of `@staticmethod`s that apply a backbone-predicted
mask (or filter coefficients) to a time-frequency representation. `Masker`
is a plain class, not an `nn.Module` — the multi-frame filters it wraps
(`DeepFilter`, `MultiFrameWienerFilter`, `MultiFrameMvdrFilter`, see
[lobe/multiframe](lobe/multiframe.md)) carry no learnable parameters, so each
call constructs a fresh one instead of Masker holding submodule state.

**Shape convention:** every method here uses an explicit real/imag axis,
`[N, 2, C, T]` (batch, real-or-imag, frequency, time) — *not* a
stacked-along-channel `[N, 2*F, T]` layout. `C` is whatever frequency-bin
count the encoder produced (e.g. 256 after `drop_stft_first_bin` on a 512-pt
FFT).

**Wiring:** `Masker` is consumed by `system.siso.EncDecMaskBase.forward` and
`system.miso.EncDecCondMaskBase.forward` (the enhancement training/inference
loops) and by the streaming ports `streaming.dpcrn.*` /
`streaming.dparn.*`. All four dispatch on a `mask_type` string:

| `mask_type` | Masker call | Backbone output |
|---|---|---|
| `complex` | `apply_complex_mask_on_reim` | one `[N, 2, C, T]` complex mask |
| `deepfilter` | `apply_df_on_reim` | one `[N, 2*order, C, T]` deep-filter mask |
| `wiener` | `apply_complex_mask_on_reim` **then** `apply_wiener`, spliced together | `(mask, ifc, cov)` triple |
| `mvdr` | `apply_complex_mask_on_reim` **then** `apply_mvdr`, spliced together | `(mask, ifc, cov)` triple |
| `mapping` | none — backbone output *is* the enhanced spectrum | `[N, 2, C, T]` |

The streaming ports and `apply_real_on_real` / `apply_mag_mask_on_reim` are
the exception: no current `mask_type` branch in `EncDecMaskBase`/
`EncDecCondMaskBase` calls the latter two, so they are library-only
primitives today, kept for a real-mask (magnitude/mapping) pipeline nothing
currently assembles.

### `envelope_postfiltering_on_cpx_mask(tf_rep, est_masks, tau=0.02) -> Tensor`

```python
envelope_postfiltering_on_cpx_mask(
    tf_rep: Tensor,     # [N, 2, C, T]
    est_masks: Tensor,  # [N, 2, C, T]
    tau: float = 0.02,
) -> Tensor             # [N, 2, C, T] -- a *mask*, not an enhanced spectrum
```

Sharpens a complex mask's magnitude response (envelope post-filtering, aka
the McAulay-Malpass-style musical-noise suppressor) before it gets applied.
Computes `mask = |est_masks| / |tf_rep|` (the mask's own implied gain),
warps it through `sin(pi * mask / 2)`, and rescales `est_masks` by the ratio
— pulling near-1 gains further toward 1 and small gains further toward 0 to
sharpen the keep/suppress decision. Called from inside
`apply_complex_mask_on_reim` when `postfilter=True`; every current call site
leaves that flag at its default `False`, so this path is not currently
exercised by any recipe.

### `apply_real_on_real(tf_rep, est_masks) -> Tensor`

```python
apply_real_on_real(tf_rep: Tensor, est_masks: Tensor) -> Tensor  # [N, C, T]
```

`tf_rep * est_masks` — plain elementwise multiply for a real-valued mask on
a real-valued (already magnitude/mapping-domain) representation. No complex
axis involved.

### `apply_mag_mask_on_reim(tf_rep, est_masks) -> Tensor`

```python
apply_mag_mask_on_reim(
    tf_rep: Tensor,     # [N, 2, C, T]
    est_masks: Tensor,  # [N, C, T]  -- real-valued magnitude mask
) -> Tensor             # [N, 2, C, T]
```

Broadcasts a single real mask across both the real and imaginary planes
(`torch.stack([est_masks, est_masks], dim=1)` then multiply) — a magnitude
mask applied to a complex representation without touching phase.

### `apply_complex_mask_on_reim(tf_rep, est_masks, postfilter=False) -> Tensor`

```python
apply_complex_mask_on_reim(
    tf_rep: Tensor,     # [N, 2, C, T]
    est_masks: Tensor,  # [N, 2, C, T]
    postfilter: bool = False,
) -> Tensor             # [N, 2, C, T]
```

Complex multiplication, done as four real chunks:
`y_real = real1*real2 - imag1*imag2`, `y_imag = real1*imag2 + imag1*real2`.
This is the mask application for DPCRN/DPARN's `mask_type: complex` (the
released voice-isolate path — see
[algorithms/dpcrn](algorithms/dpcrn.md)) and for the streaming ports.

### `apply_df_on_reim(tf_rep, est_masks, num_feats, order) -> Tensor`

```python
apply_df_on_reim(
    tf_rep: Tensor,     # [N, 2, C, T]
    est_masks: Tensor,  # [N, 2*order, num_feats, T] -- deep-filter taps
    num_feats: int,     # frequency bins the filter covers (<= C)
    order: int,         # number of taps (frames of context)
) -> Tensor             # [N, 2, C, T]
```

Builds a fresh `DeepFilter(num_freqs=num_feats, frame_size=order,
lookahead=0)` (see [lobe/multiframe](lobe/multiframe.md)) each call, reshapes
`tf_rep`/`est_masks` into the `[N, 1, T, C, 2]` / `[N, order, T, C, 2]`
layout it expects, and reshapes the result back. `est_masks`' `2*order`
channel axis packs `order` complex taps (real/imag interleaved) — a
per-frequency, per-frame linear filter over `order` neighboring time frames
rather than a single-frame multiplicative mask (DeepFilterNet-style deep
filtering). Callers derive `num_feats`/`order` from the mask tensor's own
shape (`system/siso.py`: `n_filter = mask.shape[2]`,
`n_order = int(mask.shape[1] / 2)`), so the backbone's output channel count
*is* the contract — no separate config knob.

### `apply_wiener(tf_rep, est_ifc, est_cov, order) -> Tuple[Tensor, int]`

```python
apply_wiener(
    tf_rep: Tensor,   # [N, 2, C, T]
    est_ifc: Tensor,  # [N, nbins, T, 2*order]   -- inter-frame correlation vector
    est_cov: Tensor,  # [N, nbins, T, 2*order^2] -- (inverse) covariance matrix
    order: int,       # number of taps
) -> Tuple[Tensor, int]
# enh:   [N, 2, C, T] -- full spectrum, but only the first `nbins` frequency
#        bins are actually Wiener-filtered; the rest are an unfiltered copy
#        of tf_rep passed straight through the underlying MultiFrameWienerFilter.
# nbins: est_cov.shape[1] -- how many low-frequency bins were covered
```

Builds a fresh `MultiFrameWienerFilter(num_freqs=nbins, frame_size=order,
lookahead=0)`. `nbins` is read off the covariance tensor's own shape, so it
is whatever frequency width the backbone's own `ifc`/`cov` prediction head
actually produced — typically fewer than the model's full frequency count,
since covariance-based multi-frame filtering is only reliable in a
restricted low-frequency band. **None of the backbones documented in
[algorithms/](algorithms/) currently return the required `(mask, ifc, cov)`
triple** (`DPCRN`, `DPARN`, `DPRNN`, `SkiM`, `TFGridNet`, `ConvTasNet`, `Unet`
family all return a single `Tensor`), so this path — like
`apply_real_on_real` / `apply_mag_mask_on_reim` above — is reachable in
`EncDecMaskBase`/`EncDecCondMaskBase` (`mask, ifc, cov = mask` when
`mask_type in ["wiener", "mvdr"]`) but not exercised by any backbone in this
library today; it's plumbing for a model architecture nothing here
currently implements. Callers (`system/siso.py`, `system/miso.py`) always
compute the `complex`-mask enhancement first, then splice in the refined
low band:

```python
enh = Masker.apply_complex_mask_on_reim(tf_rep=features_for_enhanced, est_masks=mask)
enh_filter, n_bins = Masker.apply_wiener(
    tf_rep=features_for_enhanced, est_ifc=ifc, est_cov=cov, order=n_order
)
enh[:, :, :n_bins, :] = enh_filter[:, :, :n_bins, :]  # low band: Wiener; high band: ratio mask
```

### `apply_mvdr(tf_rep, est_ifc, est_cov, order) -> Tuple[Tensor, int]`

```python
apply_mvdr(
    tf_rep: Tensor,   # [N, 2, C, T]
    est_ifc: Tensor,  # [N, nbins, T, 2*order]
    est_cov: Tensor,  # [N, nbins, T, 2*order^2]
    order: int,
) -> Tuple[Tensor, int]  # same contract as apply_wiener
```

Same shape contract and low-band splice pattern as `apply_wiener`, but
builds `MultiFrameMvdrFilter(num_freqs=nbins, frame_size=order, lookahead=0,
enforce_constraints=True, cholesky_decomp=True)` — a distortionless-response
(spatial) filter rather than a minimum-MSE one.
