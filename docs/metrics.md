# puresound.metrics

Audio quality evaluation metrics for speech enhancement and source separation.

## Class: `Metrics`

A collection of static methods for objective audio quality assessment. All methods accept `torch.Tensor` or `numpy.ndarray` inputs and handle shape normalization internally.

### Static Methods

#### `check_shape(est: Tensor, ref: Tensor) -> Tuple[ndarray, ndarray]`

Normalizes tensors and aligns shapes between estimated and reference signals.

- Converts tensors to numpy arrays
- Clips to the shorter length if lengths differ

---

#### `pesq_wb(est, ref, sr: int = 16000) -> float`

Computes PESQ (Perceptual Evaluation of Speech Quality) in wide-band mode.

- Requires `sr = 16000` Hz
- Score range: -0.5 to 4.5 (higher is better)

---

#### `pesq_nb(est, ref, sr: int = 8000) -> float`

Computes PESQ in narrow-band mode.

- Requires `sr = 8000` Hz
- Score range: 1.0 to 4.5

---

#### `stoi(est, ref, sr: int) -> float`

Short-Time Objective Intelligibility (STOI) score.

- Score range: 0 to 1 (higher is better)

---

#### `estoi(est, ref, sr: int) -> float`

Extended STOI (ESTOI) score, better for very low SNR conditions.

- Score range: 0 to 1

---

#### `bss_sdr(est, ref) -> float`

BSS Eval Signal-to-Distortion Ratio (SDR).

- Uses `mir_eval.separation.bss_eval_sources`
- Returns SDR in dB

---

#### `sisnr(est, ref) -> float`

Scale-Invariant Signal-to-Noise Ratio (SI-SNR).

- Higher values indicate better enhancement
- Returns value in dB

---

#### `sisnr_imp(est, ref, noisy) -> float`

SI-SNR improvement over the noisy baseline (SI-SNRi).

- `SI-SNRi = SI-SNR(est, ref) - SI-SNR(noisy, ref)`
- Returns improvement in dB

---

#### `f1_score(est, ref) -> Tuple[float, float, float, float]`

Binary classification metrics.

**Returns:** `(f1, precision, recall, accuracy)`

---

#### `noise_reduction(est, ref) -> float`

Computes the reduction in noise power between a noisy reference and enhanced output.

- Returns noise reduction in dB

## Example

```python
from puresound.metrics import Metrics

sisnr_val = Metrics.sisnr(enhanced_wav, clean_wav)
pesq_val  = Metrics.pesq_wb(enhanced_wav, clean_wav, sr=16000)
stoi_val  = Metrics.stoi(enhanced_wav, clean_wav, sr=16000)
```
