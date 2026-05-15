# puresound.nnet.features

Feature processing layers for extracting audio representations from STFT spectra.

## Class: `MelBank`

Converts a linear STFT magnitude or power spectrum into a Mel-scale filterbank representation.

### Constructor

```python
MelBank(
    n_mels: int,
    sr: int,
    n_fft: int,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    trainable: bool = False,
    log_output: bool = True,
)
```

**Parameters:**
- `n_mels` – Number of Mel filter channels
- `sr` – Sample rate in Hz
- `n_fft` – FFT size (determines number of linear frequency bins)
- `f_min` – Minimum frequency for Mel filters (Hz)
- `f_max` – Maximum frequency for Mel filters (Hz); defaults to `sr / 2`
- `trainable` – If `True`, the filterbank weights are learnable parameters
- `log_output` – If `True`, applies log compression to Mel outputs

### `forward(spec: Tensor) -> Tensor`

Applies the Mel filterbank to a linear spectrum.

**Parameters:**
- `spec` – Linear magnitude spectrum `[batch, F, T]`

**Returns:** Mel-scale feature tensor `[batch, n_mels, T]`.

---

## Class: `WeightedSum`

Computes a learnable weighted average over multiple input feature representations.

### Constructor

```python
WeightedSum(num_inputs: int)
```

**Parameters:**
- `num_inputs` – Number of input feature streams to combine

### `forward(features: List[Tensor]) -> Tensor`

Combines a list of feature tensors via learned scalar weights (softmax-normalized).

**Parameters:**
- `features` – List of `num_inputs` tensors with identical shape `[batch, C, T]`

**Returns:** Weighted sum tensor `[batch, C, T]`.

## Example

```python
from puresound.nnet.features import MelBank, WeightedSum

mel = MelBank(n_mels=80, sr=16000, n_fft=512, trainable=False, log_output=True)
mel_feat = mel(linear_spec)  # [batch, 80, T]

ws = WeightedSum(num_inputs=3)
combined = ws([feat1, feat2, feat3])
```
