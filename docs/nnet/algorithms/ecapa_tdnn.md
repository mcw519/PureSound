# puresound.nnet.ecapa_tdnn

ECAPA-TDNN speaker embedding extractor — an enhanced x-vector architecture using multi-scale temporal convolutions, squeeze-and-excitation attention, and attentive statistics pooling.

**Reference:** Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification," Interspeech 2020.

## Class: `SEModule`

Squeeze-and-Excitation attention module that recalibrates channel-wise feature responses.

### Constructor

```python
SEModule(channels: int, bottleneck: int = 128)
```

**Parameters:**
- `channels` – Number of input/output channels
- `bottleneck` – Intermediate bottleneck dimension for the SE gating MLP

### `forward(x: Tensor) -> Tensor`

Applies global average pooling, then a 2-layer MLP with sigmoid gating to re-scale channel features.

---

## Class: `EcapaBlock`

A single ECAPA-TDNN residual block combining multi-scale dilated 1D convolutions with SE attention.

### Constructor

```python
EcapaBlock(
    in_channel: int,
    out_channel: int,
    kernel_size: int,
    dilation: int,
    scale: int = 8,
)
```

**Parameters:**
- `in_channel` – Input channel dimension
- `out_channel` – Output channel dimension
- `kernel_size` – 1D convolution kernel size
- `dilation` – Dilation factor for the temporal convolution
- `scale` – Number of feature groups for multi-scale processing

### `forward(x: Tensor, t_len: Optional[Tensor] = None) -> Tensor`

Processes input through multi-scale conv → SE attention → residual connection.

---

## Class: `EcapaTdnnExtractor`

Full ECAPA-TDNN speaker embedding extractor model.

### Constructor

```python
EcapaTdnnExtractor(
    in_channel: int,
    channel: int = 512,
    embd_dim: int = 192,
)
```

**Parameters:**
- `in_channel` – Input feature dimension (number of Mel bins or STFT bins)
- `channel` – Main channel width (default: 512)
- `embd_dim` – Output speaker embedding dimension (default: 192)

### `forward(x: Tensor) -> Tensor`

Processes an input feature sequence and returns a single speaker embedding vector.

**Parameters:**
- `x` – Input feature sequence `[batch, in_channel, T]`

**Returns:** Speaker embedding `[batch, embd_dim]`.

### Pipeline

```
Input Features [B, F, T]
  └─ Frame-level TDNN layers with SE attention
       └─ Multi-layer feature aggregation
            └─ Attentive Statistics Pooling [B, 2*channel]
                 └─ Linear projection → Speaker Embedding [B, embd_dim]
```

## Example

```python
from puresound.nnet.ecapa_tdnn import EcapaTdnnExtractor

model = EcapaTdnnExtractor(in_channel=80, channel=512, embd_dim=192)

mel_feat = mel_bank(stft_mag)          # [B, 80, T]
embedding = model(mel_feat)             # [B, 192]
```
