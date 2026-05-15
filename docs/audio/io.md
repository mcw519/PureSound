# puresound.audio.io

Audio file I/O operations including loading, saving, slicing, and padding.

## Class: `AudioIO`

Provides static methods for reading and writing audio files.

### Static Methods

#### `audio_info(f_path: str) -> Dict`

Returns metadata about an audio file without loading the full waveform.

**Returns dictionary with:**
- `sr` – Sample rate
- `num_samples` – Number of samples
- `duration` – Duration in seconds
- `num_channels` – Number of channels

---

#### `open(f_path: str, target_sr: Optional[int] = None, normalize: bool = False, rescale_to: Optional[float] = None) -> Tuple[Tensor, int]`

Loads an audio file from disk.

**Parameters:**
- `f_path` – Path to the audio file (WAV, FLAC, etc.)
- `target_sr` – If provided, resample to this sample rate after loading
- `normalize` – If `True`, normalize waveform to [-1, 1] peak amplitude
- `rescale_to` – If provided, rescale RMS to this target level (dB)

**Returns:** `(waveform: Tensor, sample_rate: int)`

---

#### `save(f_path: str, wav: Tensor, sr: int)`

Saves a waveform tensor to a WAV file on disk.

**Parameters:**
- `f_path` – Output file path
- `wav` – Waveform tensor `[channels, samples]` or `[samples]`
- `sr` – Sample rate

---

#### `audio_cut(wav: Tensor, target_len: int, pad_mode: str = "zero") -> Tensor`

Randomly cuts a waveform to a target length. If the waveform is shorter than `target_len`, it is padded.

**Parameters:**
- `wav` – Input waveform `[channels, samples]` or `[samples]`
- `target_len` – Desired number of samples
- `pad_mode` – Padding strategy: `"zero"` (default) or `"repeat"`

**Returns:** Waveform of length `target_len`.

---

#### `cut_audio(wav: Tensor, start: int, end: int, pad: bool = False) -> Tensor`

Slices a waveform between `start` and `end` sample indices, with optional zero-padding if the slice extends past the signal boundary.

**Parameters:**
- `wav` – Input waveform
- `start` – Start sample index
- `end` – End sample index (exclusive)
- `pad` – If `True`, zero-pad to reach `end - start` samples

**Returns:** Sliced waveform tensor.

## Example

```python
from puresound.audio.io import AudioIO

wav, sr = AudioIO.open("speech.wav", target_sr=16000, normalize=True)
info = AudioIO.audio_info("speech.wav")
AudioIO.save("output.wav", wav, sr)
```
