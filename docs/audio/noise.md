# puresound.audio.noise

Noise injection and signal mixing utilities for data augmentation.

## Functions

### `add_bg_noise(speech: Tensor, noise: Tensor, snr_db: float, sr: int) -> Tensor`

Mixes background noise into a speech signal at a specified SNR level.

**Parameters:**
- `speech` – Clean speech waveform tensor
- `noise` – Noise waveform tensor (will be cropped or repeated to match `speech` length)
- `snr_db` – Target Signal-to-Noise Ratio in dB
- `sr` – Sample rate (used for RMS computation)

**Returns:** Noisy speech waveform tensor (same length as `speech`).

**Notes:**
- Noise is rescaled so that `SNR = 10 * log10(P_speech / P_noise) = snr_db`
- If `noise` is shorter than `speech`, it is tiled (repeated) to fill the length

---

### `add_bg_white_noise(speech: Tensor, snr_db: float) -> Tensor`

Adds zero-mean Gaussian white noise to a speech signal at a specified SNR.

**Parameters:**
- `speech` – Clean speech waveform tensor
- `snr_db` – Target Signal-to-Noise Ratio in dB

**Returns:** Noisy speech waveform tensor.

**Notes:**
- Noise is generated fresh per call from `torch.randn`
- Useful as a fast alternative to loading noise files

## Example

```python
from puresound.audio.noise import add_bg_noise, add_bg_white_noise

noisy = add_bg_noise(speech, noise_wav, snr_db=10.0, sr=16000)
noisy_white = add_bg_white_noise(speech, snr_db=5.0)
```
