# puresound.audio.augmentation

Composable audio augmentation pipeline supporting noise, reverb, speed, pitch, distortion, and filtering effects.

> **Note:** The source file is named `augmentaion.py` (typo preserved for backwards compatibility).

## Class: `AudioEffectAugmentor`

Manages a library of audio effects and applies them to speech waveforms.

### Constructor

```python
AudioEffectAugmentor()
```

Initializes with empty noise and RIR dictionaries. Effects are registered via the loader methods below.

### Data Loading Methods

#### `load_bg_noise_from_folder(folder: str, suffix: str = ".wav")`

Recursively scans `folder` for noise files and registers them in the internal noise pool.

**Parameters:**
- `folder` – Root directory containing noise files
- `suffix` – File extension to search for (default `".wav"`)

---

#### `load_rir_from_folder(folder: str, suffix: str = ".wav")`

Recursively scans `folder` for Room Impulse Response files and registers them.

---

#### `init_room_simulator(config: dict)`

Initializes a physics-based `RoomImpulseResponseSimulator` for on-the-fly RIR generation.

**Parameters:**
- `config` – Configuration dict passed to `RoomImpulseResponseSimulator`

---

#### `sample_room_scene() -> Optional[dict]`

Samples a random room scene from the physics-based simulator (if initialized).

**Returns:** Scene parameter dict, or `None` if no simulator is configured.

### Augmentation Methods

All augmentation methods operate on a waveform tensor and return an augmented waveform.

#### `sox_volume_perturbed(wav: Tensor, min_gain_db: float, max_gain_db: float) -> Tensor`

Applies random volume gain (in dB) via sox.

---

#### `sox_speed_perturbed(wav: Tensor, min_rate: float, max_rate: float) -> Tensor`

Applies random time-stretching (speed change). Changes both duration and pitch.

---

#### `sox_pitch_perturbed(wav: Tensor, min_semitones: float, max_semitones: float) -> Tensor`

Shifts pitch by a random amount (in semitones) without changing duration.

---

#### `add_bg_noise(wav: Tensor, snr_range: Tuple[float, float]) -> Tensor`

Randomly selects a noise file from the pool and mixes it at a random SNR within `snr_range`.

---

#### `add_bg_white_noise(wav: Tensor, snr_range: Tuple[float, float]) -> Tensor`

Adds Gaussian white noise at a randomly selected SNR from `snr_range`.

---

#### `apply_rir(wav: Tensor, mode: str = "full") -> Tensor`

Convolves the waveform with a randomly selected RIR from the pool.

**Parameters:**
- `mode` – Reverberation mode:
  - `"full"` – Full reverberant output
  - `"direct"` – Only direct path (early reflection trimmed)
  - `"early"` – Direct path + early reflections only

---

#### `apply_2nd_iir_response(wav: Tensor) -> Tensor`

Applies a random second-order IIR filter to simulate microphone or channel coloration.

---

#### `apply_gain_distortion(wav: Tensor, min_gain: float, max_gain: float) -> Tensor`

Applies random gain distortion to simulate saturation effects.

---

#### `apply_clipping_distortion(wav: Tensor, min_q: float, max_q: float) -> Tensor`

Clips the waveform at a randomly chosen quantile to simulate hard clipping.

---

#### `apply_src_effect(wav: Tensor, sr: int, target_sr_choices: List[int]) -> Tensor`

Simulates sample-rate conversion artifacts by downsampling and upsampling through a random intermediate rate.

---

#### `apply_hpf(wav: Tensor, fc_range: Tuple[float, float], sr: int) -> Tensor`

Applies a high-pass filter with a randomly chosen cutoff frequency from `fc_range`.

## Example

```python
from puresound.audio.augmentaion import AudioEffectAugmentor

aug = AudioEffectAugmentor()
aug.load_bg_noise_from_folder("/data/musan/noise")
aug.load_rir_from_folder("/data/rirs")

noisy = aug.add_bg_noise(clean_wav, snr_range=(-5, 20))
reverberant = aug.apply_rir(noisy, mode="full")
```
