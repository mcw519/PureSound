# puresound.nnet.loss

Loss functions for speech enhancement, speaker verification, and voice activity detection.

## Sub-modules

| Module | Description |
|--------|-------------|
| [loss.metrics](metrics.md) | GE2E speaker embedding loss and basic time-domain losses |
| [loss.sdr](sdr.md) | SDR-family losses (SI-SNR, scale-dependent SDR, t-SDR, SA-SDR) |
| [loss.spk](spk.md) | Speaker classification losses (AAM-softmax, SphereFace2) |
| [loss.stft_loss](stft_loss.md) | STFT-domain losses (multi-resolution, spectral convergence) |
| [loss.vad](vad.md) | Voice Activity Detection loss |

## Top-level Exports

```python
from puresound.nnet.loss import (
    SDRLoss,
    MultiResolutionSTFTLoss,
    SpectralLoss,
    AAMsoftmax,
    GE2ELoss,
    VADActivityLoss,
)
```
