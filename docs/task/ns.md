# puresound.task.ns

Noise Suppression (NS) dataset with dynamic augmentation.

## Class: `NoiseSuppressionDataset`

Extends `DynamicBaseDataset`. Constructs noisy speech samples on-the-fly by augmenting clean speech with background noise, reverberation, and other effects.

### Constructor

```python
NoiseSuppressionDataset(
    metafile: str,
    min_utt_length: int = 0,
    target_sr: int = 16000,
    utt_length: int = 32000,
    source_rir: bool = False,
    **augment_kwargs,
)
```

**Parameters:**
- `metafile` – Path to the CSV metafile describing the clean speech corpus
- `min_utt_length` – Minimum utterance length in samples; shorter utterances are filtered
- `target_sr` – All audio is resampled to this sample rate
- `utt_length` – Fixed output utterance length in samples (shorter utterances are padded)
- `source_rir` – If `True`, applies a source RIR to the clean speech before mixing noise (simulates recording at a distance)
- `**augment_kwargs` – Augmentation configuration forwarded to `init_augmentor()`

### `__getitem__(idx: int) -> Dict`

Returns a sample dict:

| Key | Type | Description |
|-----|------|-------------|
| `noisy_speech` | `Tensor [1, T]` | Augmented (noisy) waveform |
| `clean_speech` | `Tensor [1, T]` | Original clean waveform |
| `sr` | `int` | Sample rate |

### Augmentation Pipeline

The augmentation is applied dynamically each epoch:

1. Load clean utterance (random crop to `utt_length`)
2. (Optional) Apply source RIR via `apply_rir(mode="direct")`
3. Mix background noise at random SNR
4. Apply room reverberation via room simulator or pre-loaded RIRs
5. Apply random volume perturbation
6. Apply additional effects: speed perturbation, IIR coloring, HPF, SRC, clipping

## Example

```python
from puresound.task.ns import NoiseSuppressionDataset
from torch.utils.data import DataLoader

dataset = NoiseSuppressionDataset(
    metafile="data/train_meta.csv",
    target_sr=16000,
    utt_length=48000,
    noise_folder="/data/musan",
    rir_folder="/data/rirs",
    snr_range=(-5, 20),
    source_rir=False,
)

loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
```
