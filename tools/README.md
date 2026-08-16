# tools/

Standalone utilities that are not tied to a recipe or to the Python package.
Nothing here is imported by `puresound/`; each file runs on its own.

## `audio_annotator.html` - Span

Span is a browser-based audio span annotator for the labelling step in front of
span-scored evaluations. Mark time ranges against the waveform and
spectrogram, assign semantic tags, and export annotations for review or
scoring.

Open it directly in a browser. There is no build step, server, or dependency:
audio is decoded locally with the Web Audio API and is never uploaded.

```bash
xdg-open tools/audio_annotator.html      # or double-click the file
```

Chromium-based browsers provide the full folder workflow, including writing
sidecar files back to the selected directory. Other browsers can still use
file selection, drag-and-drop, import, export, and download.

### Loading audio and annotations

Span supports individual files, multiple files, folders, and drag-and-drop.
Supported audio extensions are:

`wav`, `wave`, `flac`, `mp3`, `ogg`, `oga`, `opus`, `m4a`, `mp4`, `aac`,
`aif`, `aiff`, `aifc`, `caf`, and `webm`.

When a folder is opened or dropped, Span builds a file rail and keeps each
file's annotations separate. Use `[` and `]` to move between files. Audio is
decoded on demand, so a large folder does not need to be decoded in full before
you can start working.

Existing annotations can be loaded in either of these ways:

* Place `<audio-name>.spans.json` or `<audio-name>.spans.csv` beside an audio
  file. Matching sidecars are picked up automatically when the folder is
  opened or the audio and sidecar are dropped together.
* Drop or choose `windows.json`, a CSV file, an Audacity label file, or a JSON
  span file and use **File > Import spans**. JSON, CSV, TSV, and TXT files are
  accepted. If a file already has annotations, choose whether to add the
  imported spans or replace the existing ones.

With a Chromium folder handle opened for read/write, **File > Save spans into
folder** or **Save all** writes `<audio-name>.spans.json` next to every
annotated audio file. A folder selected through the read-only fallback cannot
be written; use **Download** in that case.

### Workspace

The main view contains a resizable file rail, a shared time ruler, the current
audio view, and the span lane:

* **Waveform** shows the min/max signal envelope for the visible time range.
* **Spectrogram** uses a 1024-point Hann FFT and displays 0-8 kHz. Switch
  between waveform, waveform plus spectrogram, and spectrogram-only modes from
  **View**.
* **Span lane** displays all annotations on the same time axis. Drag a span's
  body to move it, or drag either edge to resize it.
* The timeline scrollbar and sideways wheel scrolling make zoomed-in files
  easy to navigate. The mouse wheel zooms around the pointer; `Shift`+wheel
  pans.

Pane gutters resize the file rail, dock, spectrogram, and span lane. Sizes are
remembered in the browser and can be restored with **View > Reset pane sizes**.
The **View** menu also provides light/dark theme switching and a file-list
toggle.

### Playback and display controls

The transport supports play/pause, stop, click-to-seek, play selection, loop
selection, 0.25-2x playback rate, volume, and a millisecond clock.

* **Boost** adds playback gain from 0 to 48 dB before a compressor/soft limiter.
  **Auto** calculates the largest clip-free boost for the current file, and
  the `LIM` indicator shows when limiting is active.
* **Amp** changes waveform and spectrogram display gain only. It never changes
  the audio or exported annotations.
* **Zoom**, **Fit**, and **->Sel** control the visible time range. Zoom can go
  down to a 20 ms window for precise boundaries.

The metadata row reports the source format, channel count, duration, peak
level, selected range, and current view. For WAV files, the exported sample
rate comes from the file header rather than the browser's decoded context rate.

### Creating and editing spans

1. Drag across the waveform or spectrogram to select a range.
2. Choose a tag in the tag column, or add a new tag. Tags are user-defined and
   receive a distinct color.
3. Press `Enter` to add the selected range as a span.

Selections can also be adjusted from the playhead with `S` and `E`. Existing
   spans can be selected, re-tagged, re-labelled, moved, resized, or deleted
   from the span lane or table. The table preserves editable labels and shows
   start, end, and duration to millisecond precision. A selected span can be
   played directly by double-clicking it in the lane.

### Export formats

Choose a format in the export bar or from the **Export** menu. All exported
times are in seconds and rounded to three decimal places.

| Format | Output |
|---|---|
| `JSON` | Single-file `{file, sample_rate, duration_s, spans:[{start_s, end_s, duration_s, tag, label}]}`. For multiple files, includes `folder`, `annotated_files`, `spans_total`, and a `files` array with each file's `path` and spans. |
| `CSV` | `start_s,end_s,duration_s,tag,label`; multi-file exports add a leading `file` column. |
| `windows.json` | Scorecard input with per-file `keep` and `suppress` arrays. |
| `Audacity` | Current file only: `start<TAB>end<TAB>label`. |

The `windows.json` exporter maps tags by name:

* A tag containing `keep`, `near`, or `double` becomes a `keep` interval.
* A tag containing `far` or `suppress` becomes a `suppress` interval.
* Tags matching neither, such as `exclude`, are omitted from both arrays.

This keeps evidence visible in Span while excluding intentionally unscored
regions from the scorecard. Exported `windows.json` can be used with:

```bash
uv run python egs/voice_isolate/scripts/eval_realcase.py \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt \
    --cases-dir <your cases dir> --device cpu --dry-blend 0.9
```

Before downloading, you can **Preview** the current export in a new tab, or
use **Copy** to place it on the clipboard. The preview tab also has its own
copy and download controls and stays synchronized while annotations change.

### Keyboard shortcuts

Keys are ignored while typing in an input, textarea, or select.

| Keys | Action |
|---|---|
| `Space` | Play or pause |
| `Shift+Space` | Play the selection |
| `L` | Toggle loop selection |
| `Left` / `Right` | Nudge the playhead by 1 second, or 0.1 second with `Shift` |
| `Enter` | Add a span from the selection |
| `1`-`9` | Select the corresponding tag |
| `S` / `E` | Set the selection start/end at the playhead |
| `Backspace` / `Delete` | Delete the selected span |
| `Escape` | Clear the active selection |
| `+` / `-` / `0` | Zoom in / out / fit the whole file |
| `Shift`+wheel | Pan the timeline |
| `wheel` | Zoom around the pointer |
| `\\` | Toggle the file rail |
| `[` / `]` | Previous / next file |
| `Cmd/Ctrl+O` | Open audio files |
| `Cmd/Ctrl+Shift+O` | Open a folder |
| `Cmd/Ctrl+I` | Import spans |
| `Cmd/Ctrl+S` | Save spans into the folder |
| `Cmd/Ctrl+E` | Download the selected export |
| `G` | Auto boost without clipping |
| `?` | Open the shortcut list |

Span scores are energy-weighted. A mislabelled second of loud near speech
inside a far span can swamp the rest of the interval, so drawing boundaries
against both the waveform and spectrogram is preferable to inferring them from
an energy heuristic alone.
