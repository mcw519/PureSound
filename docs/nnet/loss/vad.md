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

## Class: `VADHeadBCELoss`

Frame-level BCE-with-logits against the dataset's `vad_target`, for training a
backbone `VADHead` (see [nnet.lobe.heads](../lobe/heads.md)). Sets
`uses_vad_logits = True` so the training system routes
`backbone.last_vad_logits` in. `false_positive_weight` up-weights far/absent
frames wrongly scored active; `balance_per_batch` reweights the two classes to
equal mass per batch (batches with a single class fall back to plain BCE).

```python
VADHeadBCELoss(false_positive_weight: float = 1.0, false_negative_weight: float = 1.0, balance_per_batch: bool = False)
```

## Class: `BackgroundVADHeadBCELoss`

`VADHeadBCELoss` variant supervised against `background_vad_target`
(interferer-speech activity) instead of the foreground labels. Batches without
any background reference contribute a zero loss that still carries a graph edge,
so DDP never sees an unused head.

## Class: `F1_loss`

Differentiable soft-F1 loss for binary frame/utterance classification.
