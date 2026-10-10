# puresound.task.sv

繁體中文版本：[`sv.zh-TW.md`](sv.zh-TW.md)

Status: frozen legacy — kept for existing recipes, not developed further.

Speaker-embedding dataset: crop one target utterance, optionally mix in
interfering speakers at a random SIR, then apply speed / reverb / noise / SRC /
IIR / HPF / volume augmentation, implemented inline in `sv.py` rather than
through the `DeviceChain` [task.ns](ns.md) uses. Produces a single augmented
waveform plus an integer speaker label -- nothing else. There is no enrollment
input; enrollment-conditioned extraction is [task.tse](tse.md)'s job, not this
module's.

## Class: `SpeakerEmbeddingDataset`

Extends [`DynamicBaseDataset`](../dataset/dynamic_base.md).

### Constructor

```python
SpeakerEmbeddingDataset(
    metafile_path: str,
    min_utt_length_in_seconds: float = 3.0,
    min_utts_in_each_speaker: int = 5,
    target_sr: Optional[int] = None,
    training_sample_length_in_seconds: float = 6.0,
    audio_gain_normalized_to: Optional[int] = None,
    dataset_role: str = "train",
    pipeline_role: Optional[str] = None,
    curriculum=None,
    **augmentation,   # the blocks named in AUGMENTATION_BLOCKS
)
```

The constructor is `DynamicBaseDataset`'s, as for
[`NoiseSuppressionDataset`](ns.md). Augmentation blocks are keyword arguments
accepted only if listed in the class attribute `AUGMENTATION_BLOCKS`; an
unlisted name raises `TypeError`. This class keeps the base set and replaces
one entry:

| keyword | config model | what it does |
| --- | --- | --- |
| `augmentation_speed_args` | `DiscreteSpeedAugmentation` | `speed_change`, a list of speed factors picked by index, and `treat_as_new_speaker` (below), in place of the base's `ContinuousSpeedAugmentation` |

`__getitem__` reads `augmentation_speech_args` (interferers; `add_n_cases` a
plain `int`), `augmentation_noise_args`, `augmentation_reverb_args`,
`augmentation_speed_args`, `augmentation_ir_response_args`,
`augmentation_src_args`, `augmentation_hpf_args` and
`augmentation_volume_args`. The base also registers
`augmentation_compressor_args` and `vad_label_args`; they are accepted but not
read, so this dataset applies no compressor and emits no VAD labels. The
noise-suppression-only blocks (`augmentation_codec_args`,
`augmentation_packet_loss_args`, `augmentation_target_absent_args`, ...) are
not registered. There is no `enroll_speech_args` parameter.

### `__getitem__(key) -> Dict`

The item key is parsed by `parse_item_key`, as in [task.ns](ns.md): a
`(speaker, sample_rate)` pair, optionally extended with the per-item seed, row
length, epoch and utterance that [`SpeakerSampler`](sampler.md) can supply.

Returns exactly two keys:

| Key | Description |
|---|---|
| `noisy_speech` | `Tensor [1, T]`, the augmented waveform |
| `speaker_id` | `int`, `self.spk2idx[target_speaker]`, optionally shifted (see below) |

There is no `clean_speech` (or any other) key -- a classification loss
(`AAMsoftmax`, GE2E, ...) only needs the augmented waveform and its integer
label.

### Augmentation order

Each block is independently gated by its own `used`/`prob`, the same
convention as `task.ns`:

crop -> interferer SIR-mix -> clip guard -> speed -> reverb -> noise -> SRC ->
2nd-order IIR -> HPF -> volume -> final crop to `training_sample_length`

Notes:
- Unlike [task.ns](ns.md)/[task.tse](tse.md), there is no source-level
  (per-source distance/room) reverb path here -- only one whole-mix RIR
  block, applied to `noisy_speech` after the interferer mix. There is no
  concept of near/far distance in this dataset.
- The interferer block mixes in exactly `augmentation_speech_args["add_n_cases"]`
  other speakers at one SIR drawn from `snr_range`. `add_n_cases` must be a
  plain `int` here -- unlike `task.ns`/`task.voice_isolation`, which also
  accept a `[low, high]` range for a random interferer count.
- Reverb/noise/SRC/IIR/HPF/volume only ever transform `noisy_speech` -- there
  is no parallel clean-speech path to keep in sync, because there is no clean
  output.

### Speed perturbation as a speaker-label trick

`augmentation_speed_args["speed_change"]` is a **list of speed factors**
(e.g. `[0.9, 1.1]`), not a `[low, high]` range like `task.ns`'s
`speed_range`. One entry is picked by index (`sp_idx`). When
`augmentation_speed_args["treat_as_new_speaker"]` is true, the returned
`speaker_id` is shifted by `(sp_idx + 1) * len(self.spk2idx)` -- i.e. each
speed variant of a speaker becomes a *distinct* classification class. This is
the Kaldi-style speed-augmentation-as-new-speaker trick; see
`egs/speaker_embedding/conf/PS-spk-v1.yaml`'s class-loss config,
`n_classes: 21615  # equal to 7205 * 3 (speed up / slow down)`.

## Class: `SpeakerEmbeddingCollateFunc`

```python
{
    "noisy_speech": Tensor,  # [N, T], padded
    "target": Tensor,        # [N], int64, renamed from speaker_id
}
```

That is the entire collated batch -- no `sr`, `length`, or waveform-target
key comes out of this collate function (contrast
[`NoiseSuppressionCollateFunc`](ns.md), which additionally renames
`audio_sr -> sr` and `audio_length -> length`).

## Recipe wiring

The reference recipe is `egs/speaker_embedding/main.py` +
`egs/speaker_embedding/conf/PS-spk-v1.yaml`, pairing this dataset with
`EncPredClassBase` ([system.siso](../system/siso.md)) and an `AAMsoftmax`
loss:

```python
from puresound.task.sv import SpeakerEmbeddingDataset, SpeakerEmbeddingCollateFunc
from puresound.task.sampler import SpeakerSampler

train_dataset = SpeakerEmbeddingDataset(
    metafile_path="data/vox12_train",
    min_utt_length_in_seconds=4.0,
    min_utts_in_each_speaker=10,
    target_sr=16000,
    training_sample_length_in_seconds=4.0,
    audio_gain_normalized_to=-22,
    augmentation_noise_args={
        "used": True, "prob": 0.5, "noise_folder": "musan_noise/",
        "snr_range": [10, 30], "prob_white_noise": 0.2,
        "white_noise_snr_range": [20, 40],
    },
    augmentation_reverb_args={
        "used": True, "prob": 0.5, "rir_folder": "rirs/",
        "target_rir_type": "full",
    },
    augmentation_speed_args={
        "used": True, "prob": 0.5, "treat_as_new_speaker": True,
        "speed_change": [0.9, 1.1],
    },
)
train_sampler = SpeakerSampler(
    data=train_dataset.meta, total_batch=4000, n_spks=64, n_per=2,
    select_by_sr_first=False,
)
```
