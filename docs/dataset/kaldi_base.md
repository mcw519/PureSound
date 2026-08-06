# puresound.dataset.kaldi_base

繁體中文版本：[kaldi_base.zh-TW.md](kaldi_base.zh-TW.md)

> **Status: legacy** — kept working and frozen: no new features, no rewrites.
> Used by the frozen SV ([task.sv](../task/sv.md)) and TSE
> ([task.tse](../task/tse.md)) recipes.

Kaldi-style dataset reader: reads a **folder** of `<uttid> <path>` manifest
files, not individual file-path constructor arguments.

## Class: `KaldiFormBaseDataset`

Extends `torch.utils.data.Dataset`.

### Constructor

```python
KaldiFormBaseDataset(
    folder,
    resample_to: Optional[int] = None,
    mode: str = "train",
    audio_gain_normalized_to: Optional[int] = None,
    split_to_chunks_with_size: Optional[float] = None,
)
```

- `folder` – a directory expected to contain hardcoded manifest filenames
  (see [Manifest files](#manifest-files) below) — **not** individual file
  paths
- `resample_to` – if set, all opened audio is resampled to this rate
- `mode` – `"train"` (default) / `"dev"` / `"eval"` (anything else fails an
  `assert`); `"eval"` skips loading the clean reference and enables
  chunking; `"train"`/`"dev"` load the clean reference and never chunk
- `audio_gain_normalized_to` – target dBFS forwarded to
  `AudioIO.open(target_lvl=...)`
- `split_to_chunks_with_size` – **seconds**, not samples (multiplied by the
  opened file's sample rate internally: `chunk_length = int(sr *
  split_to_chunks_with_size)`). Only takes effect when `mode == "eval"`.

### Manifest files

`_folder_content` hardcodes two filenames looked up inside `folder`:

| Key | Filename | Required? |
|---|---|---|
| `wav2scp` | `wav2scp.txt` | yes — raises `FileNotFoundError` if missing |
| `wav2ref` | `wav2ref.txt` | only actually needed if `mode != "eval"` (see gotcha below) |

Both are space-separated `<uttid> <value>` files, loaded with
[`load_text_as_dict`](../utils.md). Add more files (e.g. `wav2enroll.txt`)
via the `folder_content` setter — see below.

**wav2scp.txt**:
```
utt001 /path/to/noisy/utt001.wav
utt002 /path/to/noisy/utt002.wav
```

**wav2ref.txt** (clean reference, train/dev only):
```
utt001 /path/to/clean/utt001.wav
utt002 /path/to/clean/utt002.wav
```

**wav2enroll.txt** (optional enrollment/conditional speech — only read once
you add `"wav2enroll"` via the `folder_content` setter):
```
utt001 /path/to/enroll/spk001.wav
```

### `__getitem__(idx) -> Dict`

| Key | Contents |
|---|---|
| `noisy_speech` | waveform tensor, always present |
| `clean_speech` | opened from `wav2ref` when `mode != "eval"`; **`torch.empty(0)`** otherwise (not `None`) |
| `conditional_speech` | opened from `wav2enroll` **only if** that key was added via `folder_content` and is present for this row; **`torch.empty(0)`** otherwise |
| `sr` | the opened noisy waveform's sample rate |
| `name` | the uttid (manifest key) |

Two more behaviors worth knowing:

- If `_sr` (the clean reference's native rate) differs from `sr` (the noisy
  waveform's), the reference is resampled to `sr` via the Sox backend — a
  `print()` warning is emitted, no exception.
- In `mode="eval"`, if `split_to_chunks_with_size` is set and the utterance
  is longer than the resulting `chunk_length`, `noisy_speech` is reframed
  into overlapping chunks with `torch.nn.functional.unfold` (50% hop:
  `stride=chunk_length // 2`), producing a 2-D `[n_chunks, chunk_length]`
  tensor instead of a 1-D waveform. `clean_speech`/`conditional_speech` are
  never chunked.

**Gotcha**: in `mode="train"`/`"dev"`, `__getitem__` reads
`self.df[key]["wav2ref"]` unconditionally — unlike `wav2enroll`, there is no
presence check first. If `wav2ref.txt` is missing from `folder`, `_load_df`
only prints a warning at construction time; the resulting `KeyError` does
not surface until the first `__getitem__` call.

### `folder_content` property — read-resets-to-default footgun

```python
@property
def folder_content(self):
    self._folder_content = {"wav2scp": "wav2scp.txt", "wav2ref": "wav2ref.txt"}
    return self._folder_content

@folder_content.setter
def folder_content(self, dct):
    self._folder_content.update(dct)
    self._load_df(self.folder)   # reloads immediately
```

The **setter** is the intended way to add manifest files — it merges `dct`
into the existing map and reloads `self.df`:

```python
dataset.folder_content = {"wav2enroll": "wav2enroll.txt"}
```

But the **getter silently resets** `self._folder_content` back to the
hardcoded `{wav2scp, wav2ref}` default on *every read* — including a read
that looks harmless, like printing it to inspect the current config:

```python
dataset.folder_content = {"wav2enroll": "wav2enroll.txt"}   # now tracks 3 files
print(dataset.folder_content)   # prints {wav2scp, wav2ref} -- wav2enroll is GONE
# any later code path that triggers _load_df() again (e.g. calling the
# setter a second time) reloads WITHOUT wav2enroll.
```

This is real, current behavior — not a bug slated for a fix — so treat the
getter as unsafe to read back: once you've customized `folder_content` via
the setter, never read the property again just to introspect it.

## Example

```python
from puresound.dataset.kaldi_base import KaldiFormBaseDataset
from torch.utils.data import DataLoader

# folder must contain wav2scp.txt (+ wav2ref.txt for train/dev mode)
dataset = KaldiFormBaseDataset("data/test", mode="dev")
loader = DataLoader(dataset, batch_size=1, shuffle=False)
for batch in loader:
    noisy, clean = batch["noisy_speech"], batch["clean_speech"]

# add an enrollment/conditional-speech manifest on top of the defaults
dataset.folder_content = {"wav2enroll": "wav2enroll.txt"}
```
