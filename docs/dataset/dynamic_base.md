# puresound.dataset.dynamic_base

Base dataset class with a composable dynamic augmentation pipeline for speech tasks.

## Class: `DynamicBaseDataset`

Extends `torch.utils.data.Dataset`. Provides a configurable pipeline for loading and augmenting speech data with noise, reverberation, speed perturbation, and more.

### Constructor

```python
DynamicBaseDataset(
    metafile: str,
    min_utt_length: int = 0,
    min_spk_count: int = 1,
    target_sr: int = 16000,
    # Augmentation flags and parameters are passed as keyword arguments
    **kwargs
)
```

**Key Parameters:**
- `metafile` – Path to the CSV metafile describing the dataset
- `min_utt_length` – Minimum utterance length in samples; shorter utterances are filtered out
- `min_spk_count` – Minimum number of utterances per speaker; sparse speakers are filtered out
- `target_sr` – All waveforms are resampled to this sample rate

### Methods

#### `init_necessary()`

Initializes dataset from the metafile. Calls `gen_meta()` and organizes speaker-level indices.

---

#### `gen_meta() -> Dict`

Parses the metafile and applies length/speaker count filtering. Builds:
- `self.meta` – Full utterance metadata dict
- `self.spk2utt` – Mapping from speaker ID to list of utterance IDs
- `self.gender2spk` – Mapping from gender to list of speaker IDs
- `self.sr2spk` – Mapping from sample rate to list of speaker IDs

---

#### `init_augmentor(config: Dict)`

Sets up an `AudioEffectAugmentor` instance from the provided augmentation configuration.

**Supported augmentation types (via `config` keys):**
- `speech_augment` – Enable/disable speech augmentation
- `noise_folder` – Path to background noise folder
- `rir_folder` – Path to RIR folder
- `room_simulator_config` – Config for on-the-fly RIR simulation
- `speed_perturb` – Speed perturbation range
- `snr_range` – SNR range for noise mixing
- `apply_src` – Sample rate conversion effect
- `apply_hpf` – High-pass filter effect
- `volume_perturb` – Volume perturbation range

### Dataset Organization

After initialization:
- `self.meta` – Dict of utterance metadata keyed by utterance ID
- `self.spk2utt` – Speaker-to-utterance mapping
- `self.gender2spk` – Gender-to-speaker mapping
- `self.sr2spk` – Sample-rate-to-speaker mapping

### Notes

- Subclasses (e.g., `NoiseSuppressionDataset`, `TargetSpeakerExtractDataset`) override `__getitem__` to implement task-specific sample construction.
- The augmentor is shared across all subclass implementations.

## Example

```python
class MyDataset(DynamicBaseDataset):
    def __getitem__(self, idx):
        uttid = self.utt_list[idx]
        info = self.meta[uttid]
        wav, sr = AudioIO.open(info["path"], target_sr=self.target_sr)
        # apply augmentation...
        return {"speech": wav, "sr": sr}
```
