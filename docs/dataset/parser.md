# puresound.dataset.parser

CSV metafile parsing for dataset construction.

## Class: `MetafileParser`

Parses structured CSV metafiles that describe audio datasets.

### Constructor

```python
MetafileParser(
    use_speaker_as_key: bool = False,
    insert_corpus_root_path: Optional[str] = None,
    with_label_column: bool = False,
    with_start_time_column: bool = False,
)
```

**Parameters:**
- `use_speaker_as_key` – If `True`, the returned dict is keyed by speaker ID; otherwise by utterance ID
- `insert_corpus_root_path` – Optional prefix prepended to all file paths in the metafile
- `with_label_column` – Whether the metafile contains a `label_path` column
- `with_start_time_column` – Whether the metafile contains a `start_time` column

### Methods

#### `read_from_metafile(file_path: str) -> Dict`

Parses a CSV metafile into a Python dictionary.

**Parameters:**
- `file_path` – Path to the CSV metafile

**Returns:** Dictionary mapping utterance IDs (or speaker IDs) to metadata dicts.

### Expected Metafile Format

The CSV file must have a header row with the following columns (in order):

| Column | Description |
|--------|-------------|
| `uttid` | Unique utterance identifier |
| `spkid` | Speaker identifier |
| `gender` | Speaker gender (`m` / `f`) |
| `path` | Relative or absolute path to the audio file |
| `length` | Duration in samples |
| `sr` | Sample rate |
| `channels` | Number of audio channels |
| `label_path` *(optional)* | Path to label/annotation file |
| `start_time` *(optional)* | Start time offset in seconds |

### Example

```python
from puresound.dataset.parser import MetafileParser

parser = MetafileParser(
    insert_corpus_root_path="/data/librispeech",
    with_label_column=False,
)
metadata = parser.read_from_metafile("train_meta.csv")

# metadata["utt001"] = {
#     "spkid": "spk001",
#     "gender": "m",
#     "path": "/data/librispeech/audio/utt001.wav",
#     "length": 32000,
#     "sr": 16000,
#     "channels": 1,
# }
```
