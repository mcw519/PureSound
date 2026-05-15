# puresound.nnet.masker

Mask application utilities supporting various masking strategies for speech enhancement.

## Class: `Masker`

A collection of static methods for applying different types of time-frequency masks.

### Static Methods

#### `envelope_postfiltering_on_cpx_mask(mask: Tensor, alpha: float = 0.3) -> Tensor`

Applies envelope post-filtering to a complex mask to reduce musical noise artifacts.

**Parameters:**
- `mask` – Complex mask tensor `[batch, 2*F, T]`
- `alpha` – Smoothing factor for post-filter

**Returns:** Post-filtered complex mask tensor.

---

#### `apply_real_on_real(spec: Tensor, mask: Tensor) -> Tensor`

Applies a real-valued mask to a real spectrum representation.

**Parameters:**
- `spec` – Real spectrum tensor `[batch, F, T]`
- `mask` – Real mask tensor `[batch, F, T]`

**Returns:** Masked spectrum tensor.

---

#### `apply_mag_mask_on_reim(spec: Tensor, mask: Tensor) -> Tensor`

Applies a magnitude (real-valued) mask to a complex stacked representation. The mask is applied uniformly to both real and imaginary parts.

**Parameters:**
- `spec` – Stacked real/imaginary spectrum `[batch, 2*F, T]`
- `mask` – Real magnitude mask `[batch, F, T]`

**Returns:** Masked spectrum tensor `[batch, 2*F, T]`.

---

#### `apply_complex_mask_on_reim(spec: Tensor, mask: Tensor) -> Tensor`

Applies a complex mask to a complex stacked spectrum via complex multiplication.

**Parameters:**
- `spec` – Stacked real/imaginary spectrum `[batch, 2*F, T]`
- `mask` – Complex mask (stacked real/imaginary) `[batch, 2*F, T]`

**Returns:** Masked spectrum tensor `[batch, 2*F, T]`.

---

#### `apply_df_on_reim(spec: Tensor, coeff: Tensor, frame_size: int) -> Tensor`

Applies Deep Filter coefficients for multi-frame filtering.

**Parameters:**
- `spec` – Stacked spectrum `[batch, 2*F, T]`
- `coeff` – Deep filter coefficients `[batch, 2*F*frame_size, T]`
- `frame_size` – Number of frames in the filter context

**Returns:** Filtered spectrum tensor.

---

#### `apply_wiener(spec: Tensor, coeff: Tensor, frame_size: int) -> Tensor`

Applies multi-frame Wiener filter coefficients.

**Parameters:**
- `spec` – Stacked spectrum `[batch, 2*F, T]`
- `coeff` – Wiener filter coefficients
- `frame_size` – Filter context window size

**Returns:** Filtered spectrum tensor.

---

#### `apply_mvdr(spec: Tensor, coeff: Tensor, frame_size: int) -> Tensor`

Applies multi-frame MVDR (Minimum Variance Distortionless Response) beamforming filter.

**Parameters:**
- `spec` – Stacked spectrum `[batch, 2*F, T]`
- `coeff` – MVDR filter coefficients
- `frame_size` – Filter context window size

**Returns:** Filtered spectrum tensor.

## Mask Type Summary

| Mask Type | Method | Input Spec Shape | Mask Shape |
|-----------|--------|-----------------|------------|
| Real on Real | `apply_real_on_real` | `[B, F, T]` | `[B, F, T]` |
| Magnitude on Complex | `apply_mag_mask_on_reim` | `[B, 2F, T]` | `[B, F, T]` |
| Complex on Complex | `apply_complex_mask_on_reim` | `[B, 2F, T]` | `[B, 2F, T]` |
| Deep Filter | `apply_df_on_reim` | `[B, 2F, T]` | `[B, 2F*K, T]` |
| Wiener | `apply_wiener` | `[B, 2F, T]` | varies |
| MVDR | `apply_mvdr` | `[B, 2F, T]` | varies |
