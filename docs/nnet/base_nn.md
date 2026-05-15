# puresound.nnet.base_nn

Base model classes for encoder-decoder masking pipelines.

## Class: `BaseModel`

Extends `torch.nn.Module`. Provides parameter counting utilities shared by all PureSound models.

### Properties

#### `overall_parameters -> int`

Returns the total number of parameters (trainable and frozen).

#### `overall_trainable_parameters -> int`

Returns the number of trainable parameters only.

### Methods

#### `get_state_dict() -> Dict`

Returns the model's `state_dict` for checkpointing.

---

## Class: `EncDecMaskerBaseModel`

Extends `BaseModel`. Provides mask application utilities for time-frequency domain processing.

### Methods

#### `apply_tf_masks(spec: Tensor, mask: Tensor, mask_type: str) -> Tensor`

Applies a time-frequency mask to a spectral representation.

**Parameters:**
- `spec` – Spectrum tensor `[batch, F, T]` (real) or `[batch, 2*F, T]` (stacked real/imag)
- `mask` – Mask tensor of the same shape as `spec`
- `mask_type` – Masking strategy:
  - `"complex"` – Element-wise complex multiplication
  - `"real"` – Element-wise real multiplication
  - `"polar"` – Magnitude mask applied to magnitude; phase preserved

**Returns:** Masked spectrum tensor.

---

#### `get_mask(raw: Tensor, constraint: str) -> Tensor`

Applies a non-linearity to raw network output to produce a bounded mask.

**Parameters:**
- `raw` – Raw mask logits from the network
- `constraint` – Activation constraint:
  - `"linear"` – No activation (identity)
  - `"relu"` – ReLU (non-negative mask)
  - `"sigmoid"` – Sigmoid (mask in [0, 1])

**Returns:** Bounded mask tensor.

---

#### `cpx_mul(a_re: Tensor, a_im: Tensor, b_re: Tensor, b_im: Tensor) -> Tuple[Tensor, Tensor]`

Performs complex multiplication `(a_re + j*a_im) * (b_re + j*b_im)`.

**Returns:** `(result_real, result_imag)`

## Example

```python
class MyModel(EncDecMaskerBaseModel):
    def forward(self, wav):
        spec = self.encoder(wav)
        raw_mask = self.backbone(spec)
        mask = self.get_mask(raw_mask, constraint="sigmoid")
        enhanced = self.apply_tf_masks(spec, mask, mask_type="complex")
        return self.decoder(enhanced)
```
