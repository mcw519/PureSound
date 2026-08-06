# puresound.dataset.parser

繁體中文版本：[parser.zh-TW.md](parser.zh-TW.md)

CSV metafile parsing for dataset construction.

## Class: `MetafileParser`

**Not instantiable in the usual sense** — `MetafileParser` has no `__init__`
at all; every method is a `@staticmethod`. Call methods directly on the
class: `MetafileParser.read_from_metafile(...)`, never
`MetafileParser(...)`.

### Metafile format

Comma-separated, 7 positional columns (no column names required):

```
uttid, spkid, gender, path, length, sr, channels
116-288045-0000, 116, M, dev-other/116/288045/116-288045-0000.flac, 170400, 16000, 1
```

- A header row is **optional** and is auto-detected/skipped wherever it
  appears (not only as the very first line): any row whose column 0 is
  `uttid`, column 1 is `spkid`, and column 4 is `length`
  (case-insensitive, whitespace-stripped) is treated as a header and
  skipped.
- Blank rows are skipped.
- Every other row must have exactly the expected number of columns, or
  `read_from_metafile` raises `ValueError` naming the offending line number.
- `length`, `sr`, and `channels` come back as **strings**, not ints — nothing
  in the parser casts them (callers such as
  [`dynamic_base.gen_meta()`](dynamic_base.md) explicitly `float(...)` them
  before use).

Two optional trailing columns extend the row:

```
uttid, spkid, gender, path, length, sr, channels, label_path
uttid, spkid, gender, path, length, sr, channels, label_path, start_time
```

**`with_start_time_column=True` requires `with_label_column=True`.** The
column-count check adds one expected column per flag independently, but the
row-unpacking branch only reads `start_time` when `with_label_column` is
also `True`. Setting `with_start_time_column=True` alone raises a "too many
values to unpack" `ValueError` on every row (the count check expects 8
columns, but the 7-name unpack in the `not with_label_column` branch runs
regardless). If you need a start-time column, always pass both flags
together.

### Methods

#### `read_from_metafile(f_path, use_speaker_as_key=False, insert_corpus_root_path=None, with_label_column=False, with_start_time_column=False) -> Dict`

- `f_path` – path to the CSV metafile
- `use_speaker_as_key` – changes the return shape (see below)
- `insert_corpus_root_path` – if set, prepended to `path` (and to
  `label_path` when `with_label_column`) via `os.path.join`
- `with_label_column` / `with_start_time_column` – see above

**`use_speaker_as_key=False`** (default) — flat dict keyed by `uttid`:

```python
{
    "utt001": {"spkid": ..., "gender": ..., "path": ..., "length": ...,
               "sr": ..., "channels": ..., # + "label_path"/"start_time" if requested
               },
    ...
}
```

**`use_speaker_as_key=True`** — nested dict keyed by `spkid`, with
`gender`/`channels` recorded from the first utterance seen for that
speaker:

```python
{
    "spk001": {
        "gender": ...,
        "channels": ...,
        "utts": {
            "utt001": {"path": ..., "length": ..., "channels": ..., "sr": ...,
                       # + "label_path"/"start_time" if requested
                       },
            ...
        },
    },
    ...
}
```

This is the mode [`dynamic_base.gen_meta()`](dynamic_base.md) uses.

---

#### `create_scp_files(metafile_path, out_folder, insert_corpus_root_path=None, with_label_column=False, with_start_time_column=False, rename_uttid=False, add_prefix=None) -> None`

The parser → [`kaldi_base`](kaldi_base.md) bridge: reads a metafile with
`read_from_metafile(use_speaker_as_key=False, ...)` and writes it back out as
Kaldi-style space-separated manifest files in `out_folder`:

| Always written | Only if `with_label_column` | Only if `with_start_time_column` |
|---|---|---|
| `wav2scp.txt`, `wav2spk.txt`, `wav2gender.txt`, `wav2duration.txt` | `wav2label.txt` | `wav2start.txt` |

- `wav2duration.txt` holds `length / sr` in **seconds** (computed here, not
  copied from the metafile)
- `rename_uttid=True` replaces each uttid with a zero-padded sequential
  index (`00000`, `00001`, ...); `add_prefix` (applied after any renaming)
  prepends `{prefix}_` to every uttid
- **Does not write `wav2ref.txt`.** `wav2scp.txt` alone satisfies
  [`KaldiFormBaseDataset`](kaldi_base.md)'s one required file, but if you
  need `mode="train"`/`"dev"` (which reads `wav2ref`), supply or rename your
  own clean-reference manifest into `wav2ref.txt` separately — this function
  doesn't produce one.

## Example

```python
from puresound.dataset.parser import MetafileParser

metadata = MetafileParser.read_from_metafile(
    "train_meta.csv",
    insert_corpus_root_path="/data/librispeech",
)
# metadata["utt001"] = {
#     "spkid": "spk001", "gender": "m",
#     "path": "/data/librispeech/audio/utt001.wav",
#     "length": "32000", "sr": "16000", "channels": "1",
# }

MetafileParser.create_scp_files("train_meta.csv", out_folder="data/train")
```
