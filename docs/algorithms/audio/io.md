# puresound.audio.io

繁體中文版本：[io.zh-TW.md](io.zh-TW.md)

`AudioIO` reads and writes audio files through `soundfile` (libsndfile), with
optional resampling and RMS levelling on load. Every method is a
`@staticmethod`: call `AudioIO.open(...)`, not `AudioIO().open(...)`. The
constructor exists but only stores a `verbose` flag that no method reads.

## `audio_info(f_path) -> (sample_rate, total_samples, total_seconds, num_channels)`

Header-only metadata from `soundfile.info`; the waveform is not read.
`total_seconds = round(frames / sample_rate, 2)`. The result is a plain tuple
in that order.

## `open(f_path, resample_to=None, normalized=False, target_lvl=None, verbose=False) -> (wav, sr)`

1. Reads the file as float32 with `always_2d=True` and returns
   `wav: [C, L]`, channels first. PCM files land in [-1, 1].
2. If `resample_to` is set and differs from the file's rate, converts with
   [`dsp.wav_resampling(..., backend="sox")`](dsp.md), the deterministic
   rate converter. The returned `sr` is then `resample_to`.
3. If `target_lvl` (dBFS) is set and `normalized` is false, rescales to that
   RMS level with [`volume.rescale_waveform`](volume.md)
   (`amp_type="rms"`, `scale="dB"`):

   ```
   wav ← wav / (rms(wav) + 1e-14) · 10^(target_lvl / 20)
   ```

   Levelling runs after resampling, so the level is measured at the output
   rate.

`normalized=True` turns the `target_lvl` path off and scales each channel to unit
average amplitude (`volume.normalize_waveform(..., amp_type="avg")`). No caller
uses it; level with `target_lvl`.

**Used by:** the dataset layer loads every speech clip with
`resample_to=dataset.target_sample_rate` and
`target_lvl=dataset.gain_normalized_to` (empty in the YAML = no levelling);
`AudioEffectAugmentor` loads noise and folder RIRs with it; evaluation systems
and inference processors load files with `resample_to`.

## `save(wav, f_path, sr, subtype="PCM_16", **kwargs)`

Writes with `soundfile.write`. Note the argument order: waveform first, path
second, the reverse of `open`. A 1-D `wav` is treated as `[1, L]`; the tensor is
detached, moved to CPU and transposed to libsndfile's `[L, C]`. `subtype`
defaults to 16-bit PCM (samples outside [-1, 1] clip); pass `"FLOAT"` to keep a
32-bit float payload. Extra keyword arguments go to `soundfile.write`.

## `cut_audio(wav, sr, length_s, padding=False) -> (wav, offset, end_offset)`

Random crop of a `[C, L]` tensor to `target_len = sr · length_s` samples:

| input length | result |
| --- | --- |
| `L > target_len` | a random offset in `[0, L - target_len]`, sliced to exactly `target_len` |
| `L <= target_len`, `padding=True` | zero-padded at the end to `target_len` |
| `L <= target_len`, `padding=False` | returned unchanged, shorter than `target_len` |

`sr · length_s` is truncated to an integer sample count. The offset is drawn from
Python's `random`.

## `audio_cut(wav, sr, length_s) -> (wav, (offset, end_offset))`

`cut_audio` with `padding=True`, after promoting a 1-D input to `[1, L]`.

Neither crop helper is used by the training datasets, which align lengths with
their own `align_audio_list` (see
[dynamic_base](../../architecture/dataset/dynamic_base.md)).

## Example

```python
from puresound.audio.io import AudioIO

wav, sr = AudioIO.open("speech.wav", resample_to=16000, target_lvl=-28.0)
sample_rate, total_samples, duration_s, num_channels = AudioIO.audio_info("speech.wav")
AudioIO.save(wav, "output.wav", sr)                   # 16-bit PCM
AudioIO.save(wav, "output_f32.wav", sr, subtype="FLOAT")
```
