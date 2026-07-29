# puresound.task.sv

> **Status: legacy** — kept working and frozen: no new features, no rewrites.

Speaker Verification and Speaker Embedding dataset with dynamic augmentation.

## Class: `SpeakerEmbeddingDataset`

Extends `DynamicBaseDataset`. Constructs samples containing a target speaker utterance alongside optional interfering speaker utterances and background noise.

### Constructor

```python
SpeakerEmbeddingDataset(
    metafile: str,
    min_utt_length: int = 0,
    min_spk_count: int = 1,
    target_sr: int = 16000,
    utt_length: int = 32000,
    num_interferers: int = 0,
    add_noise_to_enroll: bool = False,
    add_noise_to_test: bool = True,
    **augment_kwargs,
)
```

**Parameters:**
- `metafile` – Path to CSV metafile describing the speaker corpus
- `min_utt_length` – Minimum utterance length in samples
- `min_spk_count` – Minimum number of utterances per speaker (speakers with fewer are filtered)
- `target_sr` – Target sample rate
- `utt_length` – Fixed output length in samples
- `num_interferers` – Number of interfering speaker utterances to mix in
- `add_noise_to_enroll` – If `True`, applies augmentation to the enrollment utterance
- `add_noise_to_test` – If `True`, applies augmentation to the test utterance
- `**augment_kwargs` – Augmentation config forwarded to `init_augmentor()`

### `__getitem__(idx: int) -> Dict`

Returns a sample dict:

| Key | Type | Description |
|-----|------|-------------|
| `noisy_speech` | `Tensor [1, T]` | Augmented test utterance (noisy mixture if interferers present) |
| `clean_speech` | `Tensor [1, T]` | Clean target speaker utterance |
| `sr` | `int` | Sample rate |
| `spk_label` | `int` | Integer speaker ID for classification loss |

### Augmentation Pipeline

1. Sample target speaker utterance (random crop)
2. Sample `num_interferers` utterances from other speakers and mix at a random SIR
3. (Optional) Add background noise to the mixture
4. Apply reverb, volume perturbation, and other effects
5. Separately augment the enrollment utterance (if `add_noise_to_enroll=True`)

## Use Cases

- **Speaker identification**: Classify utterances to speaker IDs using `AAMsoftmax` or `GE2ELoss`
- **Speaker verification**: Train embeddings used for cosine similarity scoring
- **Data augmentation for pre-training**: Used before fine-tuning on downstream TSE tasks

## Example

```python
from puresound.task.sv import SpeakerEmbeddingDataset

dataset = SpeakerEmbeddingDataset(
    metafile="data/voxceleb2_meta.csv",
    min_spk_count=5,
    target_sr=16000,
    utt_length=32000,
    num_interferers=1,
    add_noise_to_test=True,
    noise_folder="/data/musan",
    snr_range=(0, 20),
)
```
