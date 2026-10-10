# puresound.dataset.dynamic_base

繁體中文版本：[dynamic_base.zh-TW.md](dynamic_base.zh-TW.md)

Base dataset for dynamic synthesis: every training row is built on the fly from a
metafile of clean speech plus noise, reverb and device-chain augmentation, so no
mixture is stored on disk. `puresound.task.ns.NoiseSuppressionDataset` subclasses
it directly, `puresound.task.voice_isolation.VoiceIsolationDataset` through the
noise-suppression dataset, and the legacy speaker-embedding and
target-speaker-extraction datasets (`puresound.task.sv`, `puresound.task.tse`) do
too. Each subclass implements `__getitem__`.

## Class: `DynamicBaseDataset`

Extends `torch.utils.data.Dataset`. Owns metafile loading and filtering, the
validation of augmentation blocks, the shared `AudioEffectAugmentor`, VAD-labeler
dispatch, sampler-key parsing and the curriculum hook. Subclasses implement the
per-item synthesis.

### Constructor

```python
DynamicBaseDataset(
    metafile_path: str,
    min_utt_length_in_seconds: float = 3.0,
    min_utts_in_each_speaker: int = 5,
    target_sr: Optional[int] = None,
    training_sample_length_in_seconds: float = 6.0,
    audio_gain_normalized_to: Optional[int] = None,
    dataset_role: str = "train",
    pipeline_role: str | None = None,
    curriculum: AugmentationArg = None,
    **augmentation: AugmentationArg,
)
```

`AugmentationArg` is a validated Pydantic model, a plain mapping, or `None`.
Lengths are in **seconds**, not samples.

- `metafile_path` – CSV metafile, parsed by
  [`MetafileParser.read_from_metafile`](parser.md)
- `min_utt_length_in_seconds` – utterances shorter than this are dropped in
  `gen_meta()`
- `min_utts_in_each_speaker` – speakers left with fewer utterances are dropped
- `target_sr` – if set, every waveform is resampled to it on load and
  `self.training_sample_length = int(target_sr * training_sample_length_in_seconds)`;
  if `None`, rows keep the rate of the file they were drawn from and
  `self.training_sample_length` is `None`
- `audio_gain_normalized_to` – target dBFS passed to
  `AudioIO.open(target_lvl=...)` wherever this class opens audio
- `dataset_role` – the dataset stage: `"train"`, `"validation"` or `"test"`;
  anything else raises `ValueError`
- `pipeline_role` – defaults to `dataset_role`; the source distribution that
  stage-sensitive capabilities (a pre-generated RIR bank split) draw from. Same
  allowed values.
- `curriculum` – a `CurriculumConfig` (see
  [Recipe configuration](../../usage/configuration.md)), or `None` for a dataset
  whose knobs are constants. The runner gives one to training datasets only, so
  validation loss stays comparable across epochs.
- `**augmentation` – the augmentation blocks, by keyword (below)

#### Augmentation blocks: `AUGMENTATION_BLOCKS`

The keyword arguments a dataset accepts are listed in the class attribute
`AUGMENTATION_BLOCKS`, a mapping `keyword -> config model`. The base registers:

| keyword | model (`puresound.config.augmentation`) |
|---|---|
| `augmentation_speech_args` | `SpeechAugmentation` |
| `augmentation_noise_args` | `NoiseAugmentation` |
| `augmentation_reverb_args` | `ReverbAugmentation` |
| `augmentation_speed_args` | `ContinuousSpeedAugmentation` |
| `augmentation_ir_response_args` | `SimpleProbAugmentation` |
| `augmentation_src_args` | `SourceRateAugmentation` |
| `augmentation_hpf_args` | `HighPassAugmentation` |
| `augmentation_volume_args` | `VolumeAugmentation` |
| `augmentation_compressor_args` | `CompressorAugmentation` |
| `vad_label_args` | `VadLabelConfig` |

Every registered keyword becomes an attribute, `None` when the caller did not
supply it, so read sites can write `if self.augmentation_noise_args:`. The recipe
forwards a block written with `used: false` as `None` too
(`BaseRecipe.augmentation_kwargs`). A value is passed through
`as_block(value, model)`: a model instance is kept, a mapping (or a model of
another type) is validated into the registered model. A keyword not in the
registry raises `TypeError` naming the accepted ones.

It is a registry rather than a parameter list because the recipe side derives its
half from the schema (`BaseRecipe.augmentation_kwargs`); a hand-written list
facing a derived one drifts. Subclasses extend the mapping, and may replace an
entry to bind a different dialect of the same block (the speaker-embedding
dataset takes discrete speeds where the separation tasks take a continuous range).
Validating mappings here means a dataset built by hand in a test or script gets
the same schema check a recipe does.

Only `augmentation_noise_args`, `augmentation_reverb_args` and `vad_label_args`
are consumed by the base class (in `init_augmentor()` and `init_vad_labeler()`);
the rest are read by subclasses while they synthesise a row.

### Initialization sequence

`__init__` stores the arguments, validates the blocks and roles, and calls
`self.init_necessary()`, which runs, in order:

1. `self.gen_meta(...)` — unpacked into `self.meta`, `self.gender_meta`,
   `self.gender_spks`, `self.sr_meta`
2. `self.total_spks` (sorted speaker ids) and `self.spk2idx`
3. `self.init_augmentor()` — builds `self.augmentor`
4. `self.init_vad_labeler()` — builds `self.vad_labeler` and
   `self.gating_vad_labeler`
5. `self.apply_curriculum_epoch(0)` — puts a schedule's first values in place, so
   a row drawn before any epoch is announced (a dump, an audit script, a plain
   PyTorch loop) sees the schedule's starting point rather than the file's
   constants

---

#### `parse_item_key(key) -> ItemKey`

Samplers hand the dataset a tuple that grows with what the run asked for:
`(speaker, sample_rate[, seed[, seconds[, epoch]]])` (see
[task.sampler](../task/sampler.md)). Anything else raises `TypeError`. The
method returns an `ItemKey(speaker, sample_rate, seed, seconds, epoch)` named
tuple (missing fields are `None`) and applies what every task shares, in this
order:

1. the row length for this item, which `sample_length` reads (seconds times
   `target_sr`, or, when there is no target rate, times the key's own
   `sample_rate`, since the utterance has not been opened yet);
2. this epoch's scheduled knobs (`apply_curriculum_epoch`), before
3. the per-item seed, which reseeds `random`, `numpy` and `torch`, so every draw
   the seed governs already sees the values the epoch asked for.

Every dynamic dataset's `__getitem__` starts here. Parsing the key in one place
keeps its shape from becoming task-specific: a seeded validation pass, a
mixed-length schedule and a curriculum work the same way for every task.

#### Derived audio parameters

- `audio_sr` – `target_sr` if set, else `self.ori_audio_sr`, the rate of the
  foreground utterance, which `__getitem__` records as it opens it.
- `sample_length` – the crop length in samples at `audio_sr`: the per-item
  override from `parse_item_key` if the key carried a length, else
  `training_sample_length`, else `ori_audio_sr * training_sample_length_in_seconds`.

Both fall back to the incoming file's rate, so they are only meaningful inside
item synthesis. Each DataLoader worker owns its own copy of the dataset and
synthesises one item at a time, so the per-item override is not shared.

---

#### `apply_curriculum_epoch(epoch)`

Moves the scheduled knobs into place for `epoch`; a no-op when no curriculum is
used or that epoch is already in force. It resolves the schedule, replaces each
named augmentation block with a validated copy (`with_overrides`), calls
`rebind_augmentation_blocks()` if any block changed, and hands scheduled bank
weights to the room bank's `set_weights` when the bank is a union of banks.

It never draws from the RNG: rows are seeded per item, and a draw here would
shift every later draw in the row depending on which epoch it was. A target it
cannot apply (a block this dataset does not have, bank weights without a union
bank) is logged as a warning, not raised: a recipe already refuses an unknown
target at load time (`BaseRecipe.curriculum_targets_exist`), and killing a
DataLoader worker mid-epoch would hang DDP on the next all-reduce.

#### `rebind_augmentation_blocks()`

Re-derives whatever was composed from an augmentation block. Blocks are read per
row, but the components built *from* them — a capture chain, a noise stage, a
gating helper — are composed once and hold a reference to the block they were
given. A subclass that composes any of them builds them in this method and calls
it from `__init__`, so construction and re-derivation cannot drift apart. Compose
only: no RNG, no disk. The base has nothing to re-derive.

---

#### `gen_meta(metafile_path, min_utts_in_spk=10, min_utt_length=3.0)`

`init_necessary()` always calls it with the constructor's values, so these
defaults only apply when it is called directly.

Parses the metafile with `MetafileParser.read_from_metafile(f_path=...,
use_speaker_as_key=True)`, then:

- drops utterances shorter than `min_utt_length` seconds (`length / sr`)
- drops speakers left with fewer than `min_utts_in_spk` utterances
- derives a `corpus_id` per speaker from the speaker id's prefix before the first
  `_` (metafiles name speakers `{corpus}_{speaker}`)
- buckets speakers by gender (`m`/`male` → `"m"`, `f`/`female` → `"f"`, anything
  else → `"other"`) and by sample rate

Returns a 4-tuple `(meta, gender_meta, gender_spks, sr_meta)`:

| value | shape |
|---|---|
| `meta` | `{spkid: {"gender", "channels", "corpus_id", "utts": {uttid: {"path", "length", "channels", "sr"}}}}` |
| `gender_meta` | `{"m"/"f"/"other": {corpus_id: [spkid, ...]}}` |
| `gender_spks` | `{"m"/"f"/"other": [spkid, ...]}` |
| `sr_meta` | `{sample_rate: {spkid: [uttid, ...]}}`; speakers with fewer than `min_utts_in_spk` utterances at that rate, and rates left with no speaker, are removed |

It also sets `self.all_corpus_id`, the set of corpus ids seen. Every corpus is
kept however few speakers it has: a minimum-speaker filter would change the
training distribution.

---

#### `init_augmentor()`

Takes no arguments; builds `self.augmentor = AudioEffectAugmentor()` from
`self.augmentation_noise_args` and `self.augmentation_reverb_args`.

- **Noise** (if the block is set): `noise_sources` (named, weighted folders) is
  loaded with `load_bg_noise_sources`; otherwise `noise_folder` with
  `load_bg_noise_from_folder`.
- **Reverb** (if the block is set): exactly one backend.

  | condition | backend |
  |---|---|
  | `simulator.used` and `simulator.pregenerated.used` | pre-generated RIR bank — `init_room_bank(...)` |
  | `simulator.used`, no `pregenerated` block or `pregenerated.used` false | physics room simulator — `init_room_simulator(...)` |
  | no `simulator` block, or `simulator.used` false | static RIR folder — `load_rir_from_folder(rir_folder)` |

  After the backend, `direct_smear` and `drr_contrast` sub-blocks, when used, are
  installed with `init_direct_smear` and `init_drr_contrast`.

Blocks handed to these components go through `delegated_kwargs`, which forwards
only the keys the recipe wrote, so each component's own defaults apply to the
rest.

**Pre-generated bank `usage_role`**: the bank block may pin its own
`usage_role` (`"train"`/`"validation"`/`"test"`). If it is set and differs from
this dataset's `pipeline_role`, `init_augmentor()` raises `ValueError`; if it is
unset, `pipeline_role` is filled in before `init_room_bank()`. That is what keeps
a validation dataset from drawing training RIRs.

---

#### VAD-labeler dispatch: `init_vad_labeler()`

Reads `self.vad_label_args` (a `VadLabelConfig`) and always sets three attributes:

- `self.gating_vad_labeler` – `None` when the block is absent or not `used`;
  otherwise an `EnergyVADLabeler`. It serves the coarse overlap-gating decisions
  made inside DataLoader workers, so it is always the cheap energy VAD and never
  Silero, whatever `backend` says.
- `self.defer_vad_to_gpu` – `True` only when `backend == "silero"`. Subclasses
  then skip per-row VAD labelling and emit the clean reference waveform; the
  training module labels the whole batch with Silero on the GPU, which keeps
  Silero out of the per-sample worker path.
- `self.vad_labeler` – the labeler subclasses call for the VAD loss target:
  `None` when VAD is disabled or deferred to the GPU, otherwise
  `create_vad_labeler(...)` from `puresound.audio.vad` (energy or Silero).

`frame_length`, `hop_length` (and `eps_mode` for the gating labeler) are read
from the block's `args` first, then from the block's own `frame_length` /
`hop_length` fields (defaults `400` / `160`); `args` is the labeler
constructor's namespace and wins.

---

### Source-level reverb helpers

These apply when `init_augmentor()` picked a room-simulator branch (pre-generated
bank or physics simulator). They let a subclass give each source its own room
channel instead of mixing pre-reverberated signals.

#### `should_apply_source_level_reverb() -> bool`

`True` with probability `augmentation_reverb_args.prob` when the reverb block is
used, its `simulator` is used and `simulator.source_level` is set. When any of
those gates is off it returns `False` without drawing from the RNG.

#### `apply_source_level_target_reverb(wav, sr, room_scene, distance_range_override=None) -> ForegroundReverb(noisy, clean, metadata)`

Applies the room scene's `"full"` RIR to `wav` as the **foreground** source to
build the noisy target, then the clean target:

- `target_rir_type == "anechoic"`: the dry `wav`
- otherwise: the **same realised RIR** (`rir_id`) rendered with
  `rir_mode=target_rir_type` (`"full"`, `"early"` or `"direct"`)

`metadata` is the realised placement (for example `source_receiver_distance`),
or `None` when the RIR came from a folder. The result unpacks as
`noisy, clean, meta = ...`.

#### `apply_source_level_interferer_reverb(wav, sr, room_scene, distance_range_override=None, source_role="interferer") -> ReverbedSource(wav, metadata)`

The same for a non-target source, tagged with `source_role` so the room scene can
place several interferers independently. The channel's `metadata` is returned
with the waveform because the caller records per-interferer placement in its RIR
lineage; reading it from the augmentor between calls would mis-attribute as soon
as anything else convolves in between.

---

### VAD target helpers

#### `create_vad_target(clean_speech, sample_rate)`

`None` when `self.vad_labeler` is `None`. An all-zero reference (a target-absent
row) returns `create_empty_vad_target(clean_speech)` without calling the labeler:
`EnergyVADLabeler` measures each frame against the utterance's own maximum, so a
silent input would otherwise read as active in every frame. Every backend thereby
reports "no activity" for an absent target.

#### `create_empty_vad_target(wav) -> Tensor`

An all-zero `float32` tensor of `frame_count(wav.shape[-1], frame_length,
hop_length)` frames (from `puresound.audio.vad`), or `None` when
`self.vad_labeler` is `None`.

---

### Utterance selection and shaping

#### `choose_an_utterance_by_speaker_name(target_speaker_name, ignoring_utt_list=None, select_channel=None, select_with_sr_as_key=None) -> (wav, sr, (speaker_name, uttid))`

Draws one utterance of `target_speaker_name` at random and opens it with
`AudioIO.open(target_lvl=self.audio_gain_normalized_to, resample_to=self.target_sr)`.

- `ignoring_utt_list` – utterance ids to exclude (for example one already used in
  the same row)
- `select_channel` – if the waveform has more channels than this index, keep only
  that channel
- `select_with_sr_as_key` – restrict the pool to
  `self.sr_meta[select_with_sr_as_key][target_speaker_name]` and assert the opened
  file has that rate. Only recipes with `target_sample_rate: null` reach this
  branch (the runner derives the sampler's `select_by_sr_first` from it).
- Both branches draw from a pool in **metafile order**, never through a `set`,
  so a fixed seed picks the same utterance in every process regardless of
  `PYTHONHASHSEED`.
- An all-silent draw is retried up to 5 times, then
  `RuntimeError("Timeout, can't find a useful utterance.")`.

#### `align_audio_list(wav_list, length, padding_type="zero") -> List[Tensor]`

Crops (random offset) or pads (a random split of the padding before and after)
every waveform to exactly `length` samples.

- A crop redraws its offset if the window is silent, at most 10 times, then
  accepts a silent window. The bound is deliberate: a genuinely silent source (a
  quiet interferer, a target-absent row) would otherwise spin forever and stall a
  DataLoader worker, which deadlocks DDP.
- `padding_type="zero"` – pad with zeros
- `padding_type="normal"` – pad, then add 40 dB-SNR white noise over the whole
  waveform (`add_bg_white_noise`)

#### `avoid_audio_clipping(wav_list) -> List[Tensor]`

If any waveform's peak exceeds 1, divides **every** waveform by the same shared
peak, which preserves the level relation between a target and its interferers.
Returns the list unchanged otherwise.

### Methods a subclass implements

- `__getitem__(key)` – the per-item synthesis; the base raises
  `NotImplementedError`
- `__len__` – the base raises `NotImplementedError`. Dynamic synthesis has no
  fixed epoch size: iteration is driven by a `batch_sampler` that carries its own
  length, and Lightning treats a dataset whose `__len__` raises
  `NotImplementedError` as unsized.
- `apply_audio_augmentation()` – raises `NotImplementedError`
- `rebind_augmentation_blocks()` – optional, see above

## Example

```python
from puresound.dataset.dynamic_base import DynamicBaseDataset

class MyDataset(DynamicBaseDataset):
    def __getitem__(self, key):
        item = self.parse_item_key(key)
        wav, self.ori_audio_sr, _ = self.choose_an_utterance_by_speaker_name(
            item.speaker, select_channel=0, select_with_sr_as_key=item.sample_rate
        )
        wav = self.align_audio_list([wav], self.sample_length)[0]
        vad_target = self.create_vad_target(wav, self.audio_sr)
        return {"speech": wav, "vad_target": vad_target}

dataset = MyDataset(
    metafile_path="data/train_speech.csv",
    target_sr=16000,
    vad_label_args={"used": True, "backend": "energy"},
)
```
