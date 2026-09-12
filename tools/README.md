# Repository tools

These are standalone development and evaluation tools. They are kept in the
repository because they support reproducible maintenance, but they are not
imported by `puresound` and are not part of the runtime package.

| Tool | Use |
|---|---|
| [`audio_annotator.html`](audio_annotator.html) | Label time ranges in audio and export evaluation windows |
| [`rng_fingerprint.py`](rng_fingerprint.py) | Check that a synthesis refactor preserves seeded outputs |

## Audio annotator

`audio_annotator.html` is a local browser application for marking labelled
time ranges on a waveform or spectrogram. It has no build step and does not
upload audio.

Open it directly:

```bash
xdg-open tools/audio_annotator.html
```

Chromium-based browsers support opening a folder and writing annotation
sidecars back to it. Other browsers can open or drag files and download the
result.

### Basic workflow

1. Open one or more audio files, or a folder.
2. Drag over the waveform or spectrogram to select a range.
3. Choose or create a tag.
4. Press `Enter` to add the span.
5. Save sidecars or export the required format.

When opening a folder, existing `<audio-name>.spans.json` or
`<audio-name>.spans.csv` files are loaded automatically.

### Export formats

| Format | Contents |
|---|---|
| JSON | File metadata and labelled spans |
| CSV | Start, end, duration, tag, and label |
| `windows.json` | Per-file `keep` and `suppress` intervals for evaluation |
| Audacity labels | Tab-separated start, end, and label for the current file |

For `windows.json`, tags containing `keep`, `near`, or `double` become
`keep` intervals. Tags containing `far` or `suppress` become `suppress`
intervals. Other tags are excluded from scoring.

### Useful controls

| Key | Action |
|---|---|
| `Space` | Play or pause |
| `Shift+Space` | Play the selection |
| `Enter` | Add the selected span |
| `S` / `E` | Set selection start or end at the playhead |
| `Backspace` / `Delete` | Delete the selected span |
| `+` / `-` / `0` | Zoom in, zoom out, or fit |
| `[` / `]` | Previous or next file |
| `Ctrl/Cmd+S` | Save sidecars into the opened folder |
| `Ctrl/Cmd+E` | Download the selected export |
| `?` | Show all shortcuts |

Use both the waveform and spectrogram when placing evaluation boundaries.
A short, loud mislabeled region can dominate an energy-weighted score.

## RNG fingerprint

`rng_fingerprint.py` compares dataset synthesis before and after a refactor.
It is intended for changes that must preserve generated samples exactly, such
as moving augmentation code, changing defaults, or splitting a processing
stage.

Unit tests verify that the current implementation is deterministic. A
before/after fingerprint adds a different guarantee: the refactor did not move
all deterministic outputs to a new result.

### Capture and compare

Capture a baseline from the original tree:

```bash
uv run python tools/rng_fingerprint.py path/to/config.yaml before.json
```

Apply the refactor and capture the same recipe again:

```bash
uv run python tools/rng_fingerprint.py path/to/config.yaml after.json
```

Compare the files:

```bash
uv run python tools/rng_fingerprint.py --compare before.json after.json
```

The command exits with status 0 when all recorded hashes match and status 1
when any output changed.

Use `--n-items` to change the default sample count of 32:

```bash
uv run python tools/rng_fingerprint.py \
  path/to/config.yaml fingerprint.json \
  --n-items 64
```

### Scope and limitations

- Use a real recipe that enables every block whose behavior must be preserved.
- The recipe's train metafile and referenced audio must be available.
- A fingerprint only covers the sampled items and enabled pipeline.
- Store temporary before/after JSON outside version control.
- An intentional behavior change should update tests or release notes instead
  of forcing the old fingerprint to pass.
