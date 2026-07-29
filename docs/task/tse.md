# puresound.task.tse

> **Status: legacy** — kept working and frozen: no new features, no rewrites.

Target Speaker Extraction (TSE) dataset with dynamic, multi-source augmentation.

## Class: `TargetSpeakerExtractDataset`

Extends `DynamicBaseDataset`. Constructs multi-source mixtures containing a target speaker (foreground), interference speakers, and background noise — along with a clean enrollment utterance for speaker conditioning.

### Constructor

```python
TargetSpeakerExtractDataset(
    metafile: str,
    min_utt_length: int = 0,
    min_spk_count: int = 2,
    target_sr: int = 16000,
    utt_length: int = 32000,
    num_interferers: int = 1,
    sir_range: Tuple[float, float] = (-5.0, 20.0),
    snr_range: Tuple[float, float] = (0.0, 30.0),
    role_based_rir: bool = False,
    **augment_kwargs,
)
```

**Parameters:**
- `metafile` – Path to CSV metafile describing the speaker corpus
- `min_utt_length` – Minimum utterance length in samples
- `min_spk_count` – Minimum number of speakers needed (at least 2 for target + interferer)
- `target_sr` – Target sample rate
- `utt_length` – Fixed output utterance length in samples
- `num_interferers` – Number of interfering speakers to include in the mixture
- `sir_range` – Speech Interference Ratio range (dB); controls relative interference level
- `snr_range` – Signal-to-Noise Ratio range (dB) for background noise
- `role_based_rir` – If `True`, uses different RIR pools for foreground/interferer/background roles
- `**augment_kwargs` – Augmentation config forwarded to `init_augmentor()`

### `__getitem__(idx: int) -> Dict`

Returns a sample dict:

| Key | Type | Description |
|-----|------|-------------|
| `noisy_speech` | `Tensor [1, T]` | Mixture of target + interferers + noise (reverberant) |
| `clean_speech` | `Tensor [1, T]` | Clean target speaker utterance (direct path only) |
| `enroll_speech` | `Tensor [1, T_enroll]` | Clean enrollment utterance from the same target speaker |
| `sr` | `int` | Sample rate |

### Augmentation Pipeline

1. **Target (foreground)**: Sample target speaker utterance, apply foreground RIR
2. **Interference**: Sample `num_interferers` utterances from different speakers, apply interferer RIRs, mix at random SIR
3. **Background noise**: Load or generate noise, mix at random SNR
4. **Final mixture**: `noisy = reverb(target) + Σ reverb(interferer_i) + noise`
5. **Clean reference**: Direct-path target (before reverberation)
6. **Enrollment**: Separate clean utterance from the same target speaker (no mixing)

### Role-Based RIR

When `role_based_rir=True`:
- Foreground uses `foreground_distance_range` RIRs (near-field, short distance)
- Interferers use `interferer_distance_range` RIRs (far-field, longer distance)

This simulates the target speaker being closer to the microphone than interference.

## Example

```python
from puresound.task.tse import TargetSpeakerExtractDataset
from torch.utils.data import DataLoader

dataset = TargetSpeakerExtractDataset(
    metafile="data/librispeech_meta.csv",
    min_spk_count=2,
    target_sr=16000,
    utt_length=48000,
    num_interferers=2,
    sir_range=(-5, 15),
    snr_range=(5, 30),
    role_based_rir=True,
    noise_folder="/data/musan",
    rir_folder="/data/rirs",
)

loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
for batch in loader:
    noisy   = batch["noisy_speech"]   # [8, 1, 48000]
    clean   = batch["clean_speech"]   # [8, 1, 48000]
    enroll  = batch["enroll_speech"]  # [8, 1, T_enroll]
```
