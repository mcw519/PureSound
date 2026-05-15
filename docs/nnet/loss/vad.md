# puresound.nnet.loss.vad

Voice Activity Detection (VAD) loss function for training models with implicit speech activity awareness.

## Class: `VADActivityLoss`

A differentiable VAD loss that supervises the model based on frame-level speech activity, derived from the power level of reference frames.

### Constructor

```python
VADActivityLoss(
    n_fft: int,
    hop_length: int,
    win_length: int,
    threshold_db: float = -40.0,
    reduction: str = "mean",
)
```

**Parameters:**
- `n_fft` – FFT size for frame-level power computation
- `hop_length` – Frame hop size in samples
- `win_length` – Analysis window length in samples
- `threshold_db` – Power threshold in dBFS below which a frame is considered inactive/silence (default: -40 dB)
- `reduction` – Loss reduction: `"mean"` or `"sum"`

### `forward(est: Tensor, ref: Tensor) -> Tensor`

Computes VAD activity loss, penalizing the model for producing output during silent frames (over-generation) and for suppressing output during active speech frames (over-suppression).

**Parameters:**
- `est` – Estimated (enhanced) waveform `[batch, T]`
- `ref` – Reference (clean) waveform `[batch, T]`

**Returns:** Scalar VAD activity loss.

### Behavior

1. Computes frame-level power of `ref` using STFT
2. Labels each frame as active (speech) or inactive (silence) based on `threshold_db`
3. Penalizes `est` for:
   - Having energy in frames labeled as silence (reduces residual noise)
   - Lacking energy in frames labeled as active speech (preserves speech)

## Use Case

`VADActivityLoss` is used as an auxiliary loss alongside SDR or spectral losses to discourage speech suppression artifacts and encourage clean noise suppression.

## Example

```python
from puresound.nnet.loss.vad import VADActivityLoss

vad_loss = VADActivityLoss(n_fft=512, hop_length=128, win_length=512, threshold_db=-40.0)
loss = vad_loss(enhanced_wav, clean_wav)
```
