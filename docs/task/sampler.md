# puresound.task.sampler

N-way K-shot speaker sampler for episode-based training (metric learning, prototypical networks).

## Class: `SpeakerSampler`

Implements N-way K-shot episode sampling: selects N speakers and K utterances per speaker, producing batches suitable for metric learning training.

### Constructor

```python
SpeakerSampler(
    spk2utt: Dict[str, List[str]],
    n_way: int,
    k_shot: int,
    n_episodes: int,
    target_sr: Optional[int] = None,
    sr2spk: Optional[Dict[int, List[str]]] = None,
)
```

**Parameters:**
- `spk2utt` – Mapping from speaker ID to list of utterance IDs (from `DynamicBaseDataset`)
- `n_way` – Number of speakers (classes) per episode
- `k_shot` – Number of utterances per speaker per episode
- `n_episodes` – Total number of episodes to generate per epoch
- `target_sr` – If provided, filters speakers to only those with matching sample rate (requires `sr2spk`)
- `sr2spk` – Mapping from sample rate to list of speaker IDs (optional, for SR-filtered sampling)

### `__iter__() -> Iterator[List[str]]`

Returns an iterator over episodes. Each episode yields a list of `n_way * k_shot` utterance IDs:

```
Episode = [spk1_utt1, spk1_utt2, ..., spk1_uttK,
           spk2_utt1, ..., spk2_uttK,
           ...
           spkN_utt1, ..., spkN_uttK]
```

### `__len__() -> int`

Returns `n_episodes`.

## Use Cases

- **Prototypical Networks**: Each support set is N speakers × (K-1) utterances; query is the remaining 1 utterance per speaker
- **GE2E Training**: `N = num_speakers_per_batch`, `K = utterances_per_speaker` in each GE2E batch
- **Contrastive Learning**: Sample positive/negative pairs for speaker metric learning

## Example

```python
from puresound.task.sampler import SpeakerSampler
from puresound.dataset.dynamic_base import DynamicBaseDataset
from torch.utils.data import DataLoader

# Initialize base dataset to get spk2utt
base_ds = DynamicBaseDataset(metafile="train_meta.csv", target_sr=16000)
base_ds.init_necessary()

sampler = SpeakerSampler(
    spk2utt=base_ds.spk2utt,
    n_way=64,        # 64 speakers per batch
    k_shot=2,        # 2 utterances per speaker
    n_episodes=1000,
)

# Use with a custom collate function for episode batching
for episode_utt_ids in sampler:
    # episode_utt_ids: list of 64*2 = 128 utterance IDs
    ...
```
