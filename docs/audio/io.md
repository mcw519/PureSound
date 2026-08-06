# puresound.audio.io

繁體中文版本：[`io.zh-TW.md`](io.zh-TW.md)

Audio file I/O: load + resample + level, save, and a length-normalizing
crop. Every method is a `@staticmethod` on `AudioIO`; nothing in this repo
ever instantiates the class — always call `AudioIO.open(...)`, not
`AudioIO().open(...)`.

## Class: `AudioIO`

### Constructor

```python
AudioIO(verbose: bool = False)
```

Exists but is vestigial: `self.verbose` is stored and never read by any
method (they're all `@staticmethod`s and take their own `verbose` argument
where relevant). No call site in this repo constructs an `AudioIO` instance.

### Static Methods

#### `audio_info(f_path: str) -> Tuple[int, int, float, int]`

Reads a file's metadata via `torchaudio.info` **without loading the
waveform**. Returns a **plain tuple**, not a dict — order matters:

```python
sample_rate, total_samples, total_seconds, num_channels = AudioIO.audio_info(f_path)
```

`total_seconds` is `round(num_frames / sample_rate, 2)`.

---

#### `open(f_path: str, resample_to: Optional[int] = None, normalized: bool = False, target_lvl: Optional[float] = None, verbose: bool = False) -> Tuple[Tensor, int]`

Loads a file with `torchaudio.load`, optionally resamples via
[`dsp.wav_resampling(..., backend="sox")`](dsp.md), then optionally levels
it. `resample_to` happens *before* levelling, so `target_lvl` is measured on
the already-resampled signal.

> **Levelling has a sharp edge: `normalized=True` alone does nothing.** The
> actual logic is
> ```python
> if normalized:
>     if target_lvl is not None and verbose:
>         wav = normalize_waveform(wav=wav, amp_type="avg")
> elif target_lvl is not None:
>     wav = rescale_waveform(wav=wav, target_lvl=target_lvl, amp_type="rms", scale="dB")
> ```
> so peak/avg normalization only fires when `normalized=True` **and**
> `target_lvl` is also set **and** `verbose=True`. Every call site in this
> repo avoids `normalized` entirely and uses `target_lvl` instead — that is
> the path to use:

```python
wav, sr = AudioIO.open(f_path, target_lvl=-28.0)   # RMS-normalize to -28 dBFS
```

`target_lvl` alone (the common case, `normalized` left at its default
`False`) rescales via `rescale_waveform(..., amp_type="rms", scale="dB")` —
see [volume.md](volume.md). `target_lvl=None` (also the default) with
`normalized=False` performs no levelling at all — `open` just loads (and
optionally resamples).

---

#### `save(wav: Tensor, f_path: str, sr: int, **kwargs)`

**`wav` is the first positional argument, `f_path` the second** — the
reverse order from `open`/`audio_info`, which take the path first. A 1-D
`wav` is unsqueezed to `[1, L]` before saving. `**kwargs` forwards to
`torchaudio.save` (e.g. `encoding`, `bits_per_sample`).

```python
AudioIO.save(wav=wav, f_path="output.wav", sr=16000)
```

---

#### `audio_cut(wav: Tensor, sr: int, length_s: float) -> Tuple[Tensor, Tuple[int, int]]`

Convenience wrapper: `cut_audio(wav, sr, length_s, padding=True)`. Returns
`(wav, (offset, end_offset))`.

---

#### `cut_audio(wav: Tensor, sr: int, length_s: int, padding: bool = False) -> Tuple[Tensor, int, int]`

**A random-offset crop to a fixed target length** (`sr * length_s` samples)
— not a deterministic `[start, end)` slice:

| condition | result |
|---|---|
| `wav.shape[-1] > target_len` | random offset in `[0, len(wav) - target_len]`, sliced to exactly `target_len` |
| `wav.shape[-1] <= target_len` and `padding=True` | zero-padded on the end to `target_len` |
| `wav.shape[-1] <= target_len` and `padding=False` | returned unchanged — **shorter than `target_len`**, caller must handle |

Returns **3 values**, `(wav, offset, end_offset)`, not 1.

Neither `audio_cut` nor `cut_audio` has an external caller or a test in this
repo currently (grepped: none). The dataset layer's own length-alignment
utility, `align_audio_list` in `puresound/dataset/dynamic_base.py`, is a
separate, independent implementation of the same "random crop or pad to a
target length" idea — it does not call into either of these.

## Example

```python
from puresound.audio.io import AudioIO

wav, sr = AudioIO.open("speech.wav", resample_to=16000, target_lvl=-28.0)
sample_rate, total_samples, duration_s, num_channels = AudioIO.audio_info("speech.wav")
AudioIO.save(wav=wav, f_path="output.wav", sr=sr)
```
