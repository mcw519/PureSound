# puresound.nnet.lobe.multiframe

繁體中文版本：`multiframe.zh-TW.md`

Multi-frame speech enhancement modules (DeepFilterNet-style): unfold a
spectrogram into small temporal context windows per (frequency, time) cell,
then apply learned or closed-form (Wiener/MVDR) filter coefficients across
that window instead of a single-frame complex mask.

**Shape convention in this module: `[B, C, T, F, ...]`** — batch, channel,
time, frequency, with an optional trailing `2` for stacked real/imag. This is
a **different, real convention** from
[`masker.Masker`](../masker.md)'s `[N, 2, C, T]` (real/imag stacked right
after batch, then frequency, then time) — the two are not the same axis
order, and `nnet/masker.py` explicitly `.permute(...)`s between them at every
call site below. Don't assume the two modules' tensors are interchangeable
without a transpose.

## Class: `MultiFrameModule`

Base class: turns a `[B, C, T, F]` spectrogram into overlapping per-step
context windows of `frame_size` frames.

```python
MultiFrameModule(num_freqs: int, frame_size: int, lookahead: int, real: bool = False)
```

**Parameters:**
- `num_freqs` – number of frequency bins the filter is actually applied to (may be less than the full spectrogram width — see `narrow` calls in the subclasses below)
- `frame_size` – context window size in frames
- `lookahead` – **required, no default** — how many of the `frame_size` frames are *future* frames (`frame_size - 1 - lookahead` are past frames); `lookahead=0` is fully causal
- `real` – selects between `spec_unfold` (complex input) and `spec_unfold_real` (real-valued input with an explicit extra trailing axis) padding/unfolding logic. **No caller in this file or elsewhere sets `real=True`** — every concrete subclass below only ever exercises the complex (`real=False`) path

### `spec_unfold(spec: Tensor) -> Tensor`

**Parameters:** `spec` – complex `[B, C, T, F]`

Pads the time axis (`frame_size - 1 - lookahead` before, `lookahead` after)
and unfolds it. **Returns:** `[B, C, T, F, N]`, `N = frame_size`.

### `spec_unfold_real(spec: Tensor) -> Tensor`

Real-valued counterpart (used only when `real=True`, which nothing in this
codebase currently triggers).

### `solve(Rxx, rss, diag_eps=1e-8, eps=1e-7)` (staticmethod)

Generic `Rxx⁻¹ @ rss` solve via `torch.inverse` + Tikhonov regularization
([`_tik_reg`](#module-level-helpers)). Defined for reuse but **not called by
any of the three concrete filters below** — they each inline their own
solve/multiply logic — nor by anything else in the repository.

### `apply_coefs(spec, coefs)` (staticmethod)

`einsum("...n,...n->...", spec, coefs)` — applies a per-window coefficient
vector to the unfolded spectrogram. Used by `MultiFrameWienerFilter` and
`MultiFrameMvdrFilter` (not `DeepFilter`, which uses the module-level `df()`
function instead).

---

## Module-level helpers

- **`psd(x: Tensor, n: int) -> Tensor`** – `X · conj(X)ᵀ` outer-product PSD/correlation matrix over an `n`-step time-unfolded window. `x`: `[B, C, T, F]` → returns `[B, C, T, F, n, n]`. Defined but not called anywhere in the repository — a library utility for computing `Rxx` from raw spectrograms, for callers that don't already have a covariance estimate from elsewhere.
- **`df(spec, coefs) -> Tensor`** – `einsum("...tfn,...ntf->...tf", spec, coefs)`, the actual deep-filtering sum-product. Used by `DeepFilter.forward`.
- **`_compute_mat_trace(input, dim1=-2, dim2=-1)`** – trace along two dims; used by `_tik_reg`.
- **`_tik_reg(mat, reg=1e-7, eps=1e-8) -> Tensor`** – Tikhonov-regularizes a correlation matrix: `mat + (trace(mat).real * reg + eps) * I`. Used by `MultiFrameWienerFilter`/`MultiFrameMvdrFilter` when their input is *not* already an inverse (`inverse=False`).

---

## Class: `DeepFilter`

Applies **learned** filter coefficients directly (no Wiener/MVDR structure) —
DeepFilterNet's core operator.

```python
DeepFilter(
    num_freqs: int,
    frame_size: int,
    lookahead: int,
    real: bool = False,
    conj: bool = False,
)
```

**Parameters:**
- `num_freqs`, `frame_size`, `lookahead`, `real` – as in `MultiFrameModule`
- `conj` – if `True`, conjugates the coefficients before applying them

**Reference:** Schröter et al., "DeepFilterNet: A Low Complexity Speech
Enhancement Framework for Full-Band Audio based on Deep Filtering," ICASSP 2022.

### `forward(spec: Tensor, coefs: Tensor) -> Tensor`

**Parameters (source docstring's own convention — see the callout above):**
- `spec` – `[B, C, T, F, 2]`
- `coefs` – `[B, N, T, F, 2]` (`N = frame_size`)

Only the first `num_freqs` bins are touched — `spec[..., :num_freqs, :]` gets
replaced by the filtered result, higher bins pass through unmodified.

**Returns:** `[B, C, T, F, 2]`, same shape as `spec`.

---

## Class: `MultiFrameWienerFilter`

Multi-frame Wiener filter from an inter-frame correlation (IFC) vector and a
noisy covariance matrix.

```python
MultiFrameWienerFilter(
    num_freqs: int,
    frame_size: int,
    lookahead: int,
    cholesky_decomp: bool = False,
    inverse: bool = True,
    enforce_constraints: bool = True,
    eps: float = 1e-8,
    dload: float = 1e-7,
)
```

**Parameters:**
- `num_freqs`, `frame_size`, `lookahead` – as in `MultiFrameModule`
- `cholesky_decomp` – if `True`, the covariance input is `L` from `Rxx = L·Lᴴ`, not `Rxx` itself
- `inverse` – if `True` (default), the covariance input is already `Rxx⁻¹`, so no `torch.linalg.solve` is needed — just a linear combination; if `False`, it's the plain `Rxx`, regularized via `_tik_reg` and solved
- `enforce_constraints` – re-imposes Hermitian symmetry (plain matrix input) or zeroes the invalid upper triangle (Cholesky input) before use
- `eps`, `dload` – regularization constants forwarded to `_tik_reg` when `inverse=False`

### `forward(spec, ifc, iRxx) -> Tensor`

**Parameters:**
- `spec` – `[B, 1, T, F, 2]`
- `ifc` – `[B, T, F, N*2]`, inter-frame speech correlation vector
- `iRxx` – `[B, T, F, (N**2)*2]`, (inverse) noisy covariance matrix or its Cholesky factor

**Returns:** `[B, C, T, F, 2]`.

---

## Class: `MultiFrameMvdrFilter`

Multi-frame MVDR (distortionless) beamformer — same constructor shape as
`MultiFrameWienerFilter`, different weight formula (normalized so the
steering direction passes distortionless).

```python
MultiFrameMvdrFilter(
    num_freqs: int,
    frame_size: int,
    lookahead: int,
    cholesky_decomp: bool = False,
    inverse: bool = True,
    enforce_constraints: bool = True,
    eps: float = 1e-8,
    dload: float = 1e-7,
)
```

### `forward(spec, ifc, iRnn) -> Tensor`

Same shapes as `MultiFrameWienerFilter.forward`, with `iRnn` playing the role
of `iRxx` (noise, rather than noisy, covariance).

---

## Class: `DeepFilterDecoder`

Predicts the `ifc`/covariance tensors that `MultiFrameWienerFilter` /
`MultiFrameMvdrFilter` need, from a bottleneck embedding plus the raw complex
spectrogram — combines a cheap [`group_op.SqueezedGRU`](group_op.md) branch
over the embedding with a small separable-conv branch directly over the
spectrogram.

```python
DeepFilterDecoder(
    df_bins: int,
    cpx_in_channels: int,
    emb_in_dim: int,
    emb_hid_dim: int = 256,
    df_order: int = 3,
    df_n_layer: int = 3,
    df_pathway_kernel_size_t: int = 1,
    df_lin_groups: int = 1,
)
```

**Parameters:**
- `df_bins` – number of frequency bins to predict coefficients for
- `cpx_in_channels` – channel count of the complex-spectrogram input branch
- `emb_in_dim` / `emb_hid_dim` / `df_n_layer` – forwarded to the internal `SqueezedGRU(emb_in_dim, emb_hid_dim, num_layers=df_n_layer, linear_groups=8)`
- `df_order` – filter tap count (`N` / `frame_size` in the filters above)
- `df_pathway_kernel_size_t` – kernel size of the two separable-conv pathways over the raw spectrogram
- `df_lin_groups` – groups used by the two output `GroupedLinear` projections (independent of the `SqueezedGRU`'s own hardcoded `linear_groups=8`)

### `forward(emb, c0) -> Tuple[Tensor, Tensor]`

**Parameters:**
- `emb` – `[N, C, T]`, bottleneck embedding
- `c0` – `[N, CH, C, T]`, complex spectrogram (`CH = cpx_in_channels`)

**Returns:** `(ifc, cov)`:
- `ifc` – `[N, C, T, df_order * 2]`
- `cov` – `[N, C, T, df_order**2 * 2]`

(here `C` is the frequency axis, `df_bins` wide — **frequency before time**,
the opposite axis order from `MultiFrameWienerFilter`/`MultiFrameMvdrFilter`'s
own `[B, T, F, ...]` input convention above). No caller wires this directly to
those two filters anywhere in the repository today (no test covers it
either), but `nnet/masker.py`'s `Masker.apply_wiener` / `Masker.apply_mvdr`
demonstrate the exact transpose needed to bridge the two conventions — see
Wiring below.

## Wiring

`nnet/masker.py`'s `Masker` static methods are the only current callers of
this module, each constructing the filter fresh per call and transposing
into its expected axis order first:

```python
# Masker.apply_wiener / Masker.apply_mvdr, abbreviated
tf_rep = tf_rep.permute(0, 3, 2, 1).unsqueeze(1)          # [N,2,C,T] -> [N,1,T,C,2]
est_ifc = est_ifc.permute(0, 2, 1, 3)                     # [N,C,T,*] -> [N,T,C,*]
est_cov = est_cov.permute(0, 2, 1, 3)                     # [N,C,T,*] -> [N,T,C,*]
wiener = MultiFrameWienerFilter(num_freqs=nbins, frame_size=order, lookahead=0)
enh = wiener(tf_rep, est_ifc, est_cov)
```

This permute is exactly what would be needed to feed `DeepFilterDecoder`'s
`(ifc, cov)` output into these filters, even though no current model does so
end-to-end. `Masker.apply_df_on_reim` is the analogous glue for `DeepFilter`.

## Example

```python
from puresound.nnet.lobe.multiframe import DeepFilter

deepfilter = DeepFilter(num_freqs=257, frame_size=3, lookahead=0)
spec = torch.rand(2, 1, 100, 257, 2)     # [B, C, T, F, 2]
coefs = torch.rand(2, 3, 100, 257, 2)    # [B, N, T, F, 2]
enh = deepfilter(spec, coefs)            # [B, C, T, F, 2]
```
