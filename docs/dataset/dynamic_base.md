# puresound.dataset.dynamic_base

繁體中文版本：[dynamic_base.zh-TW.md](dynamic_base.zh-TW.md)

Base dataset class with a composable dynamic augmentation pipeline for speech
tasks. `puresound.task.ns.NoiseSuppressionDataset` and
`puresound.task.voice_isolation.VoiceIsolationDataset` (and the legacy
`puresound.task.sv` / `puresound.task.tse` datasets) all subclass this and
override `__getitem__`.

## Class: `DynamicBaseDataset`

Extends `torch.utils.data.Dataset`. Owns metafile loading/filtering, the
shared `AudioEffectAugmentor`, and VAD-labeler dispatch; subclasses implement
the actual per-item synthesis.

### Constructor

```python
DynamicBaseDataset(
    metafile_path: str,
    min_utt_length_in_seconds: float = 3.0,
    min_utts_in_each_speaker: int = 5,
    target_sr: Optional[int] = None,
    training_sample_length_in_seconds: float = 6.0,
    audio_gain_normalized_to: Optional[int] = None,
    augmentation_speech_args: AugmentationArg = None,
    augmentation_noise_args: AugmentationArg = None,
    augmentation_reverb_args: AugmentationArg = None,
    augmentation_speed_args: AugmentationArg = None,
    augmentation_ir_response_args: AugmentationArg = None,
    augmentation_src_args: AugmentationArg = None,
    augmentation_hpf_args: AugmentationArg = None,
    augmentation_volume_args: AugmentationArg = None,
    vad_label_args: AugmentationArg = None,
    dataset_role: str = "train",
    pipeline_role: Optional[str] = None,
)
```

Every augmentation/VAD/role parameter is an explicit keyword argument with a
default — there is **no `**kwargs`** catch-all. Every length is in
**seconds**, not samples:

- `metafile_path` – path to the CSV metafile (parsed by
  [`MetafileParser.read_from_metafile`](parser.md))
- `min_utt_length_in_seconds` (default `3.0`) – utterances shorter than this
  are dropped in `gen_meta()`
- `min_utts_in_each_speaker` (default `5`) – speakers with fewer surviving
  utterances than this are dropped entirely
- `target_sr` – if set, `self.training_sample_length = int(target_sr *
  training_sample_length_in_seconds)` (samples); if `None`,
  `self.training_sample_length` stays `None`
- `training_sample_length_in_seconds` (default `6.0`)
- `audio_gain_normalized_to` – target dBFS forwarded to
  `AudioIO.open(target_lvl=...)` everywhere this base class opens audio
- `augmentation_speech_args`, `augmentation_noise_args`,
  `augmentation_reverb_args`, `augmentation_speed_args`,
  `augmentation_ir_response_args`, `augmentation_src_args`,
  `augmentation_hpf_args`, `augmentation_volume_args` – validated Pydantic
  capability models (a mapping is accepted and validated at this boundary);
  only `augmentation_noise_args` and
  `augmentation_reverb_args` are consumed here (in `init_augmentor()`) — the
  rest are read by subclasses' `__getitem__`
- `vad_label_args` – see [VAD-labeler dispatch](#vad-labeler-dispatch-init_vad_labeler) below
- `dataset_role` (default `"train"`) – the actual dataset stage.
- `pipeline_role` (defaults to `dataset_role`) – the source distribution used
  by stage-sensitive capabilities such as a pre-generated RIR bank.

### Initialization sequence

`__init__` calls `self.init_necessary()`, which runs, in order:

1. `self.gen_meta(...)` — unpacks its 4-tuple return into `self.meta`,
   `self.gender_meta`, `self.gender_spks`, `self.sr_meta`
2. builds `self.total_spks` (sorted speaker-id list) and `self.spk2idx`
3. `self.init_augmentor()` — builds `self.augmentor`
4. `self.init_vad_labeler()` — builds `self.vad_labeler` / `self.gating_vad_labeler`

---

#### `gen_meta(metafile_path, min_utts_in_spk=10, min_utt_length=3.0)`

The default values here (`min_utts_in_spk=10`) differ from the constructor's
own defaults (`min_utts_in_each_speaker=5`) — in practice this never matters,
because `init_necessary()` always calls it with explicit
`metafile_path=self.metafile_path,
min_utt_length=self.min_utt_length_in_seconds,
min_utts_in_spk=self.min_utts_in_each_speaker`, so the constructor's values
win. `gen_meta`'s own defaults only apply if you call it directly.

Parses the metafile via
`MetafileParser.read_from_metafile(f_path=metafile_path,
use_speaker_as_key=True)`, then:

- drops individual utterances shorter than `min_utt_length` seconds
  (`length / sr < min_utt_length`)
- drops speakers left with fewer than `min_utts_in_spk` utterances
- derives a `corpus_id` per speaker from the speaker id's prefix before the
  first `_` (metafiles are expected to name speakers `{corpus}_{speaker}`)
- buckets speakers by gender (`"m"` / `"f"` / `"other"`) and by sample rate

**Returns a 4-tuple** `(meta, gender_meta, gender_spks, sr_meta)`:

| Return value | Shape |
|---|---|
| `meta` | `{spkid: {"gender", "channels", "corpus_id", "utts": {uttid: {"path", "length", "channels", "sr"}}}}` |
| `gender_meta` | `{"m"/"f"/"other": {corpus_id: [spkid, ...]}}` |
| `gender_spks` | `{"m"/"f"/"other": [spkid, ...]}` |
| `sr_meta` | `{sample_rate: {spkid: [uttid, ...]}}` (a `defaultdict`) |

**Side effect**: also sets `self.all_corpus_id` (the set of corpus ids seen)
directly on `self` — in *addition to* the 4-tuple return. Calling `gen_meta` as
a bound method (as `init_necessary()` does) therefore has an effect beyond its
return value. Note there is no corpus-level filter: every corpus id reaches
`all_corpus_id`. One used to be written here but its condition was `len(...) < 0`,
which is never true, so it never removed anything; it was deleted rather than
repaired, because a real min-speaker filter would change the training
distribution. Speaker- and utterance-level filtering (`min_utts_in_spk`,
`min_utt_length`) is real and happens above.

---

#### `init_augmentor()`

Takes **no arguments** — reads `self.augmentation_noise_args` and
`self.augmentation_reverb_args` (set in `__init__`) and builds
`self.augmentor = AudioEffectAugmentor()`.

- If `augmentation_noise_args` is set:
  `self.augmentor.load_bg_noise_from_folder(augmentation_noise_args["noise_folder"])`
- If `augmentation_reverb_args` is set, a **three-way branch** picks exactly
  one reverb backend:

  | Condition on `augmentation_reverb_args["simulator"]` | Backend |
  |---|---|
  | `simulator.used` and `simulator.pregenerated.used` | pre-generated RIR bank — `self.augmentor.init_room_bank(pregenerated_config)` |
  | `simulator.used`, and no `pregenerated` block (or `pregenerated.used` false) | physics-based room simulator — `self.augmentor.init_room_simulator(simulator_args)` |
  | no `simulator` key, or `simulator.used` falsy | static RIR folder — `self.augmentor.load_rir_from_folder(augmentation_reverb_args["rir_folder"])` |

  **Pre-generated bank `usage_role` contract**: the pregenerated config may set
  its own `usage_role` (e.g. to pin a bank split to
  `"train"`/`"validation"`/`"test"`). If it is set and does not match this
  dataset instance's `pipeline_role`, `init_augmentor()` raises `ValueError`.
  If it is unset, it is set to `self.pipeline_role` before `init_room_bank()`.
  The shared runner passes distinct train/validation dataset roles and reads
  their pipeline roles from the typed `dataset` config.

---

#### VAD-labeler dispatch: `init_vad_labeler()`

Reads `self.vad_label_args` (`cfg`) and always sets three attributes:

- `self.gating_vad_labeler` – `None` if `cfg` is falsy or `cfg["used"]` is
  falsy; otherwise always an `EnergyVADLabeler(frame_length=...,
  hop_length=...)` (cheap energy-based VAD). Meant for coarse overlap-gating
  decisions made **inside DataLoader workers**, so it deliberately never uses
  Silero regardless of `cfg["backend"]`.
- `self.defer_vad_to_gpu` – `True` only when `cfg["backend"] == "silero"`
  (case-insensitive); tells subclasses to skip per-sample VAD labeling in the
  worker and instead emit the raw clean waveform, leaving a batched GPU
  Silero pass (owned by the training module) to compute the loss's VAD
  target.
- `self.vad_labeler` – the labeler subclasses call for the actual VAD loss
  target:
  - `None` if VAD is disabled, **or** if `backend == "silero"` (deferred to
    GPU, see above)
  - otherwise `create_vad_labeler(cfg)` (from `puresound.audio.vad`), which
    re-reads `cfg["backend"]` and constructs `EnergyVADLabeler` or
    `SileroVADLabeler` accordingly

`frame_length`/`hop_length` are read from `cfg["args"]` first, falling back
to top-level `cfg["frame_length"]`/`cfg["hop_length"]`, then `400`/`160`.

---

### Source-level reverb helpers

These only make sense once `init_augmentor()` has picked the room-simulator
branch (pre-generated bank or physics simulator) — they let a subclass
reverberate one source waveform at a time instead of only mixing
pre-reverberated signals.

#### `should_apply_source_level_reverb() -> bool`

`True` only if `augmentation_reverb_args["used"]`, its `simulator` block is
used, *and* `simulator["source_level"]` is set — then draws
`torch.rand(1) < augmentation_reverb_args["prob"]`. `False`
(deterministically, no RNG draw at all) whenever any of those config gates
is off.

#### `apply_source_level_target_reverb(wav, sr, room_scene, distance_range_override=None) -> ForegroundReverb(noisy, clean, metadata)`

Applies the room scene's RIR to `wav` as the **foreground** source
(`source_role="foreground"`, `rir_mode="full"`) to build `noisy_target`. Then
builds `clean_target`:

- if `augmentation_reverb_args["target_rir_type"] == "anechoic"`:
  `clean_target = wav` (no reverb at all)
- otherwise: re-applies the **same realized RIR** (`rir_id`) with
  `rir_mode=target_rir_type` (e.g. a direct-path-only or early-reflections
  mode)

`rir_metadata` is the realized placement dict (e.g.
`source_receiver_distance`) from the simulator, or `None` when the RIR
actually came from a folder rather than the simulator.

#### `apply_source_level_interferer_reverb(wav, sr, room_scene, distance_range_override=None, source_role="interferer") -> ReverbedSource(wav, metadata)`

Same idea for a non-target source: applies the room scene's `"full"` RIR under
the given `source_role` tag (so the room scene can place multiple interferers
independently). There is no clean pair, but the channel's `metadata` comes back
alongside the waveform — the caller needs per-interferer placement for its RIR
lineage, and the only other way to get it was to read the augmentor's private
`_last_rir_meta` between calls.

---

### VAD target helpers

#### `create_vad_target(clean_speech, sample_rate)`

`None` if `self.vad_labeler` is `None`. Otherwise, if `clean_speech` is an
all-zero tensor, returns `create_empty_vad_target(clean_speech)` instead of
calling the labeler — **this sidesteps a real footgun in energy-based VAD**:
`EnergyVADLabeler` normalizes each frame's power in dB relative to *that
utterance's own maximum*, so an all-silent input has `reference == eps` and
every frame reads back as "active" (`0 dB > -40 dB` threshold). Target-absent
training rows (an intentionally all-zero clean reference) would otherwise
get a spurious all-ones VAD target.

#### `create_empty_vad_target(wav) -> Tensor`

Returns an all-zero tensor of length `frame_count(wav.shape[-1],
self.vad_labeler.frame_length, self.vad_labeler.hop_length)` (from
`puresound.audio.vad.frame_count`), or `None` if `self.vad_labeler` is
`None`.

---

### Utterance selection and shaping

#### `choose_an_utterance_by_speaker_name(target_speaker_name, ignoring_utt_list=None, select_channel=None, select_with_sr_as_key=None) -> (wav, sr, (speaker_name, uttid))`

Randomly samples one utterance for `target_speaker_name`, opened via
`AudioIO.open(target_lvl=self.audio_gain_normalized_to,
resample_to=self.target_sr)`.

- `ignoring_utt_list` – utterance ids to exclude from the draw (e.g. one
  already used elsewhere in the same sample)
- `select_channel` – if given, and the opened waveform has more channels
  than this index, keeps only that channel (forces mono)
- `select_with_sr_as_key` – if given, restricts the candidate pool to
  `self.sr_meta[select_with_sr_as_key][target_speaker_name]` instead of the
  speaker's full utterance list, and asserts the opened file's sample rate
  matches. Both branches draw from a pool in **metafile order**, which is what
  makes a fixed seed reproduce the same pick across processes; this branch used
  to route the pool through a `set` and so was `PYTHONHASHSEED`-dependent. Only
  recipes with `target_sample_rate: null` reach it (`runner.py` derives
  `select_by_sr_first` from that), so no shipped recipe was affected — see
  `test_sr_keyed_utterance_pool_keeps_metafile_order`
- if the drawn utterance is all-silent, retries recursively up to 5 times,
  then raises `RuntimeError("Timeout, can't find a useful utterance.")`

#### `align_audio_list(wav_list, length, padding_type="zero") -> List[Tensor]`

Crops (random offset) or pads (a random split of the needed padding
before/after) every waveform in `wav_list` to exactly `length` samples.

- Crop retries the random offset if the chosen window is silent, but only up
  to 10 attempts, then accepts a silent window anyway — this bound is
  deliberate, not a bug: without it, a genuinely silent source (a quiet
  interferer, or a target-absent row) would spin the retry loop forever and
  stall a DataLoader worker, which deadlocks DDP (one rank never reaches the
  next collective).
- `padding_type="zero"` – pad with zeros
- `padding_type="normal"` – pad, then add 40 dB-SNR background white noise
  over the whole padded waveform via `add_bg_white_noise`

#### `avoid_audio_clipping(wav_list) -> List[Tensor]`

If any waveform's peak absolute value exceeds 1, divides **every** waveform
in the list by the same shared peak (not each one by its own peak) — this
preserves relative level between e.g. a target and its interferers. Returns
the list unchanged if nothing clips.

### Abstract / unimplemented methods

- `__len__` – raises `NotImplementedError`; subclasses must override. Dynamic
  synthesis has no fixed epoch size, and iteration on the training path is
  driven by a `batch_sampler` that carries its own length, so nothing asks the
  dataset for one
- `__getitem__` – raises `NotImplementedError`
- `apply_audio_augmentation()` – raises `NotImplementedError`

## Example

```python
from puresound.dataset.dynamic_base import DynamicBaseDataset

class MyDataset(DynamicBaseDataset):
    def __len__(self):
        return len(self.total_spks)

    def __getitem__(self, idx):
        spk = self.total_spks[idx]
        wav, sr, (spk, uttid) = self.choose_an_utterance_by_speaker_name(spk)
        wav = self.align_audio_list([wav], self.training_sample_length)[0]
        vad_target = self.create_vad_target(wav, sr)
        return {"speech": wav, "sr": sr, "vad_target": vad_target}
```
