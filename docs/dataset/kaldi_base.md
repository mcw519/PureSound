# puresound.dataset.kaldi_base

> **Status: legacy** — kept working and frozen: no new features, no rewrites.

Kaldi-format dataset reader using `.scp` manifest files.

## Class: `KaldiFormBaseDataset`

Extends `torch.utils.data.Dataset`. Reads audio data from Kaldi-style `wav.scp` manifest files, optionally with clean reference and enrollment speaker mappings.

### Constructor

```python
KaldiFormBaseDataset(
    wav2scp: str,
    wav2ref: Optional[str] = None,
    wav2enroll: Optional[str] = None,
    target_sr: int = 16000,
    mode: str = "eval",
    chunk_length: Optional[int] = None,
)
```

**Parameters:**
- `wav2scp` – Path to `wav.scp`-style file: `<uttid> <wav_path>` per line
- `wav2ref` – Optional path to clean reference mapping: `<uttid> <ref_path>` per line
- `wav2enroll` – Optional path to enrollment speaker mapping: `<uttid> <enroll_path>` per line
- `target_sr` – Target sample rate; all audio is resampled to this rate
- `mode` – Dataset split mode:
  - `"train"` / `"dev"` – Training or validation mode
  - `"eval"` – Evaluation mode (no augmentation)
- `chunk_length` – If set, long utterances are split into chunks of this length (in samples)

### `__getitem__(idx) -> Dict`

Returns a sample dict:

| Key | Description |
|-----|-------------|
| `noisy_speech` | Noisy/input waveform tensor |
| `clean_speech` | Clean reference waveform (if `wav2ref` provided), else `None` |
| `conditional_speech` | Enrollment waveform (if `wav2enroll` provided), else `None` |
| `sr` | Sample rate |
| `name` | Utterance ID |

### Manifest File Formats

**wav.scp** (required):
```
utt001 /path/to/noisy/utt001.wav
utt002 /path/to/noisy/utt002.wav
```

**wav2ref.txt** (optional):
```
utt001 /path/to/clean/utt001.wav
utt002 /path/to/clean/utt002.wav
```

**wav2enroll.txt** (optional):
```
utt001 /path/to/enroll/spk001.wav
utt002 /path/to/enroll/spk001.wav
```

## Example

```python
from puresound.dataset.kaldi_base import KaldiFormBaseDataset
from torch.utils.data import DataLoader

dataset = KaldiFormBaseDataset(
    wav2scp="data/test/wav.scp",
    wav2ref="data/test/wav2ref.txt",
    target_sr=16000,
    mode="eval",
)

loader = DataLoader(dataset, batch_size=1, shuffle=False)
for batch in loader:
    noisy = batch["noisy_speech"]
    clean = batch["clean_speech"]
```
