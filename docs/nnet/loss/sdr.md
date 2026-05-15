# puresound.nnet.loss.sdr

SDR-family loss functions for time-domain speech enhancement.

## Class: `SDRLoss`

Implements multiple variants of Signal-to-Distortion Ratio (SDR) based losses.

### Constructor

```python
SDRLoss(
    mode: str = "sisnr",
    pit: bool = False,
    reduction: str = "mean",
    zero_mean: bool = True,
    eps: float = 1e-8,
)
```

**Parameters:**
- `mode` – Loss variant (see table below)
- `pit` – If `True`, enables Permutation-Invariant Training (PIT) for multi-speaker scenarios
- `reduction` – Loss reduction: `"mean"` or `"sum"`
- `zero_mean` – If `True`, removes mean (DC offset) before computing SDR (required for SI-SNR)
- `eps` – Small constant for numerical stability

### `forward(est: Tensor, ref: Tensor, inactive: Optional[Tensor] = None) -> Tensor`

**Parameters:**
- `est` – Estimated (enhanced) waveform `[batch, T]` or `[batch, num_spk, T]`
- `ref` – Reference (clean) waveform, same shape as `est`
- `inactive` – Optional boolean mask `[batch]` indicating inactive (silent) samples to exclude from loss

**Returns:** Scalar loss (negative SDR, so minimizing loss maximizes SDR).

### Loss Modes

| Mode | Description | Reference |
|------|-------------|-----------|
| `"sisnr"` | Scale-Invariant SNR (SI-SNR) | Luo & Mesgarani, 2018 |
| `"sdsdr"` | Scale-Dependent SDR | — |
| `"tsdr"` | Soft-max t-SDR | Roux et al., 2019 |
| `"sasdr"` | Source-Aggregated SDR (for multi-speaker) | Drude et al., 2021 |

### Factory Method

#### `SDRLoss.init_mode(mode: str, **kwargs) -> SDRLoss`

Creates an `SDRLoss` instance configured for the given mode.

```python
loss_fn = SDRLoss.init_mode("sisnr", pit=False)
```

## SI-SNR Formula

$$\text{SI-SNR} = 10 \log_{10} \frac{\|\text{proj}(\hat{s})\|^2}{\|\hat{s} - \text{proj}(\hat{s})\|^2}$$

where $\text{proj}(\hat{s}) = \frac{\langle \hat{s}, s \rangle}{\|s\|^2} s$ is the projection of the estimate onto the reference.

## Example

```python
from puresound.nnet.loss.sdr import SDRLoss

sisnr_loss = SDRLoss(mode="sisnr", zero_mean=True)
loss = sisnr_loss(enhanced_wav, clean_wav)

# Ignore silent samples
loss = sisnr_loss(enhanced_wav, clean_wav, inactive=silence_mask)
```
