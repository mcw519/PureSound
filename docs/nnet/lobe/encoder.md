# puresound.nnet.lobe.encoder

Audio encoding and decoding modules for transforming waveforms to latent feature representations.

## Class: `FreeEncDec`

Learnable waveform encoder/decoder using Conv1d analysis and transposed Conv1d synthesis. Also known as a "free" encoder as it learns the analysis filters end-to-end.

> Exported from `puresound.nnet` as `FreeEncDec`.

### Constructor

```python
FreeEncDec(
    win: int,
    stride: int,
    out_channel: int,
    bias: bool = False,
)
```

**Parameters:**
- `win` – Analysis window length (filter kernel size)
- `stride` – Encoder hop size
- `out_channel` – Number of encoder output channels (learned filters)
- `bias` – If `True`, adds a learnable bias to the encoder filters

### Methods

#### `encode(wav: Tensor) -> Tensor`

Encodes a waveform to a latent feature representation.

**Parameters:**
- `wav` – Input waveform `[batch, 1, T]`

**Returns:** Feature tensor `[batch, out_channel, T//stride]`.

#### `decode(feat: Tensor, original_length: Optional[int] = None) -> Tensor`

Decodes a feature tensor back to a waveform.

**Parameters:**
- `feat` – Feature tensor `[batch, out_channel, T_feat]`
- `original_length` – If provided, trims/pads output to this length

**Returns:** Reconstructed waveform `[batch, 1, T]`.

---

## Class: `ConvEncDec`

STFT-based encoder/decoder with configurable frequency scaling (linear or log-scale).

> Exported from `puresound.nnet` as `ConvEncDec`.

### Constructor

```python
ConvEncDec(
    n_fft: int,
    hop_length: int,
    win_length: int,
    freq_scale: str = "linear",
    out_channel: Optional[int] = None,
)
```

**Parameters:**
- `n_fft` – FFT size
- `hop_length` – STFT hop size
- `win_length` – Analysis window length
- `freq_scale` – Frequency axis scaling:
  - `"linear"` – Standard linear STFT bins
  - `"log"` – Log-compressed frequency bins
- `out_channel` – Optional linear projection of frequency bins after STFT

### Methods

#### `encode(wav: Tensor) -> Tuple[Tensor, Dict]`

Computes STFT and returns features with metadata.

**Parameters:**
- `wav` – Input waveform `[batch, 1, T]`

**Returns:** `(features, stft_meta)` where `features` is `[batch, C, T_frame]`.

#### `decode(feat: Tensor, meta: Dict, original_length: Optional[int] = None) -> Tensor`

Reconstructs waveform from STFT features using inverse STFT.

**Returns:** Waveform `[batch, 1, T]`.

---

## Class: `ConvSTFT`

Trainable STFT/iSTFT implemented via convolutional kernels (no autograd through `torch.stft`). Enables end-to-end gradient flow through the analysis-synthesis filterbank.

### Constructor

```python
ConvSTFT(
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str = "hann",
    trainable: bool = False,
)
```

**Parameters:**
- `n_fft` – FFT size
- `hop_length` – Frame hop size
- `win_length` – Window length
- `window` – Window function type: `"hann"`, `"hamming"`, etc.
- `trainable` – If `True`, analysis kernels are learnable parameters

### Methods

#### `forward(wav: Tensor) -> Tuple[Tensor, Tensor]`

Computes STFT and returns real and imaginary parts separately.

**Returns:** `(real, imag)` each of shape `[batch, n_fft//2+1, T_frame]`.

## Example

```python
from puresound.nnet.lobe.encoder import FreeEncDec, ConvEncDec

# Learnable encoder
enc = FreeEncDec(win=16, stride=8, out_channel=512)
feat = enc.encode(wav)
wav_out = enc.decode(feat, original_length=wav.shape[-1])

# STFT encoder
enc_stft = ConvEncDec(n_fft=512, hop_length=128, win_length=512)
feat, meta = enc_stft.encode(wav)
wav_out = enc_stft.decode(feat, meta)
```
