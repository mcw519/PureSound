# puresound.system.siso

Single-Input Single-Output (SISO) PyTorch Lightning training modules.

## Class: `EncDecMaskBase`

Full SISO speech enhancement pipeline: encodes waveform → extracts features → processes with backbone → applies TF mask → decodes to waveform.

### Architecture

```
Noisy Waveform
  └─ Encoder (FreeEncDec or ConvEncDec)
       └─ Feature Extractor (MelBank or identity)
            └─ Backbone Network (DPRNN, DPCRN, UNet, ...)
                 └─ Mask Estimation
                      └─ Mask Application (complex/real/deepfilter/wiener/mvdr)
                           └─ Decoder
                                └─ Enhanced Waveform
```

### Constructor

```python
EncDecMaskBase(
    encoder: nn.Module,
    feature_encoder: Optional[nn.Module],
    backbone: nn.Module,
    mask_type: str = "complex",
    mask_constraint: str = "linear",
    loss_fns: List[Tuple[nn.Module, float]] = [],
    hparam: Dict = {},
)
```

**Parameters:**
- `encoder` – Waveform encoder (e.g., `FreeEncDec`, `ConvEncDec`)
- `feature_encoder` – Optional feature processing layer (e.g., `MelBank`)
- `backbone` – Core sequence model
- `mask_type` – Type of mask to apply: `"complex"`, `"real"`, `"magnitude"`, `"deepfilter"`, `"wiener"`, `"mvdr"`
- `mask_constraint` – Mask non-linearity: `"linear"`, `"relu"`, `"sigmoid"`
- `loss_fns` – List of `(loss_fn, weight)` tuples
- `hparam` – Full hyperparameter config dict

### `forward(noisy_wav: Tensor) -> Tensor`

**Parameters:**
- `noisy_wav` – Noisy input waveform `[batch, 1, T]`

**Returns:** Enhanced waveform `[batch, 1, T]`.

---

## Class: `EncPredClassBase`

SISO classification pipeline for speaker embedding extraction. Encodes waveform → extracts features → backbone → pooling → embedding projection.

### Architecture

```
Input Waveform
  └─ Encoder
       └─ Feature Extractor
            └─ Backbone (ECAPA-TDNN, etc.)
                 └─ Pooling (AttentiveStatisticsPooling)
                      └─ Linear Projection
                           └─ Speaker Embedding [batch, embd_dim]
```

### Constructor

```python
EncPredClassBase(
    encoder: nn.Module,
    feature_encoder: Optional[nn.Module],
    backbone: nn.Module,
    loss_fns: List[Tuple[nn.Module, float]] = [],
    hparam: Dict = {},
)
```

### `forward(wav: Tensor) -> Tensor`

**Parameters:**
- `wav` – Input waveform `[batch, 1, T]`

**Returns:** Speaker embedding `[batch, embd_dim]`.

## Example

```python
from puresound.system.siso import EncDecMaskBase
from puresound.nnet.lobe.encoder import FreeEncDec
from puresound.nnet.dprnn import DPRNN
from puresound.nnet.loss.sdr import SDRLoss

encoder  = FreeEncDec(win=16, stride=8, out_channel=512)
backbone = DPRNN(in_channel=512, hid_channel=64, out_channel=512, num_layers=6)

model = EncDecMaskBase(
    encoder=encoder,
    feature_encoder=None,
    backbone=backbone,
    mask_type="complex",
    loss_fns=[(SDRLoss(mode="sisnr"), 1.0)],
)
```
