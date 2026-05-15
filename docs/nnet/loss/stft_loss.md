# puresound.nnet.loss.stft_loss

STFT-domain losses for spectral fidelity in speech enhancement.

## Helper Functions

### `angle(cpx: Tensor) -> Tensor`

Gradient-robust phase angle computation from a complex tensor.

Avoids gradient instability near zero amplitude using a numerically stable atan2 implementation.

---

### `as_complex(re: Tensor, im: Tensor) -> Tensor`

Converts stacked real/imaginary tensors to a PyTorch complex tensor.

---

## Class: `SpectralConvergengeLoss`

Measures the Frobenius-norm spectral divergence between estimated and reference magnitude spectra.

$$\mathcal{L}_{sc} = \frac{\||\hat{S}| - |S|\|_F}{\||S|\|_F}$$

### `forward(est_mag: Tensor, ref_mag: Tensor) -> Tensor`

**Returns:** Scalar spectral convergence loss.

---

## Class: `LogSTFTMagnitudeLoss`

L1 loss on log-compressed magnitude spectra.

$$\mathcal{L}_{log} = \| \log|\hat{S}| - \log|S| \|_1$$

### `forward(est_mag: Tensor, ref_mag: Tensor) -> Tensor`

**Returns:** Scalar log-magnitude loss.

---

## Class: `MultiResolutionSTFTLoss`

Combines spectral convergence and log-magnitude losses at multiple STFT resolutions for richer spectral supervision.

### Constructor

```python
MultiResolutionSTFTLoss(
    fft_sizes: List[int] = [512, 1024, 2048],
    hop_sizes: List[int] = [120, 240, 480],
    win_lengths: List[int] = [512, 1024, 2048],
    sc_weight: float = 1.0,
    mag_weight: float = 1.0,
)
```

**Parameters:**
- `fft_sizes` – List of FFT sizes for each resolution
- `hop_sizes` – Corresponding hop sizes
- `win_lengths` – Corresponding window lengths
- `sc_weight` – Weight for spectral convergence term
- `mag_weight` – Weight for log-magnitude term

### `forward(est: Tensor, ref: Tensor) -> Tensor`

**Parameters:**
- `est` – Estimated waveform `[batch, T]`
- `ref` – Reference waveform `[batch, T]`

**Returns:** Sum of multi-resolution STFT losses.

---

## Class: `SpectralLoss`

Combined magnitude + phase loss operating on complex STFT representations.

### Constructor

```python
SpectralLoss(
    n_fft: int,
    hop_length: int,
    win_length: int,
    mag_weight: float = 1.0,
    phase_weight: float = 0.0,
)
```

### `forward(est: Tensor, ref: Tensor) -> Tensor`

Computes weighted combination of magnitude and phase loss.

---

## Class: `OverSuppressionLoss`

Penalizes over-suppression artifacts by measuring when the enhanced signal falls below the clean reference in the spectral domain.

### Constructor

```python
OverSuppressionLoss(n_fft: int, hop_length: int, win_length: int)
```

### `forward(est: Tensor, ref: Tensor) -> Tensor`

**Returns:** Scalar over-suppression penalty loss.

## Example

```python
from puresound.nnet.loss.stft_loss import MultiResolutionSTFTLoss

mr_loss = MultiResolutionSTFTLoss(
    fft_sizes=[512, 1024, 2048],
    hop_sizes=[128, 256, 512],
    win_lengths=[512, 1024, 2048],
)
loss = mr_loss(enhanced_wav, clean_wav)
```
