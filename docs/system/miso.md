# puresound.system.miso

> **Status: legacy** — kept working and frozen: no new features, no rewrites.

Multi-Input Single-Output (MISO) PyTorch Lightning training module for speaker-conditional speech enhancement and target speaker extraction.

## Class: `EncDecCondMaskBase`

Speaker-conditional enhancement pipeline. Processes a noisy mixture alongside a conditioning enrollment utterance to extract the target speaker.

> Exported from `puresound.system` as `EncDecCondMaskBase`.

### Architecture

```
Noisy Mixture Waveform               Enrollment (Conditioning) Waveform
  └─ Encoder                           └─ Cond. Encoder (shared or separate)
       └─ Feature Extractor                  └─ Cond. Feature Extractor
            └─ Backbone                           └─ Cond. Backbone
                 └─ Main Features                      └─ Speaker Embedding
                      └────────────── FiLM/Concat ──────────────┘
                                           │
                                     Mask Estimation
                                           │
                                     Mask Application
                                           │
                                     Decoder
                                           │
                                   Enhanced Waveform
```

### Constructor

```python
EncDecCondMaskBase(
    encoder: nn.Module,
    feature_encoder: Optional[nn.Module],
    backbone: nn.Module,
    cond_encoder: Optional[nn.Module],
    cond_feature_encoder: Optional[nn.Module],
    cond_backbone: nn.Module,
    siamese: bool = True,
    joint_training: bool = True,
    mask_type: str = "complex",
    mask_constraint: str = "linear",
    loss_fns: List[Tuple[nn.Module, float]] = [],
    hparam: Dict = {},
)
```

**Parameters:**
- `encoder` / `feature_encoder` / `backbone` – Main enhancement path components
- `cond_encoder` / `cond_feature_encoder` / `cond_backbone` – Conditioning (speaker embedding) path components
- `siamese` – If `True`, main and conditioning paths share encoder weights (Siamese architecture)
- `joint_training` – If `True`, both paths are trained jointly end-to-end; if `False`, the conditioning path is frozen
- `mask_type` – Masking strategy: `"complex"`, `"real"`, `"magnitude"`, `"deepfilter"`, `"wiener"`, `"mvdr"`
- `mask_constraint` – Mask non-linearity: `"linear"`, `"relu"`, `"sigmoid"`
- `loss_fns` – List of `(loss_fn, weight)` tuples for multi-objective training
- `hparam` – Full hyperparameter config dict

### Per-Component Learning Rate Scaling

`EncDecCondMaskBase` supports different learning rates per component group (encoder, feature encoder, backbone, conditioning path) via the `hparam` config:

```yaml
optim:
  lr: 0.001
  lr_factor:
    encoder: 0.1
    cond_backbone: 0.5
```

### `forward(noisy_wav: Tensor, enroll_wav: Tensor) -> Tensor`

**Parameters:**
- `noisy_wav` – Noisy mixture waveform `[batch, 1, T]`
- `enroll_wav` – Enrollment/reference waveform for the target speaker `[batch, 1, T_enroll]`

**Returns:** Enhanced waveform of the target speaker `[batch, 1, T]`.

## Example

```python
from puresound.system.miso import EncDecCondMaskBase
from puresound.nnet.lobe.encoder import FreeEncDec
from puresound.nnet.dpcrn import DPCRN
from puresound.nnet.ecapa_tdnn import EcapaTdnnExtractor

encoder  = FreeEncDec(win=16, stride=8, out_channel=512)
backbone = DPCRN(in_channel=2, out_channel=2, ..., embed_dim=192)
spk_model = EcapaTdnnExtractor(in_channel=80, embd_dim=192)

model = EncDecCondMaskBase(
    encoder=encoder,
    backbone=backbone,
    cond_backbone=spk_model,
    siamese=False,
    mask_type="complex",
)
```
