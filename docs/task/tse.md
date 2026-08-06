# puresound.task.tse

繁體中文版本：[`tse.zh-TW.md`](tse.zh-TW.md)

> **Status: legacy** -- kept working and frozen: no new features, no rewrites.

Target Speaker Extraction (TSE) dataset: like [task.ns](ns.md), it builds a
target-vs-interferer-vs-noise mixture, but every item also carries a
separately-augmented **enrollment** utterance (a second clip of the same
target speaker, used to condition the model on which voice to extract).

This dataset does **not** subclass [`NoiseSuppressionDataset`](ns.md) or use
its row-type-hook skeleton -- it extends `DynamicBaseDataset` directly with
its own, independent `__getitem__` (no `RowPlan`, no
overlap-gating/turn-taking, no real-recording rows). It reuses only the
low-level per-source-RIR helpers (`should_apply_source_level_reverb`,
`apply_source_level_target_reverb`, `apply_source_level_interferer_reverb`)
that `DynamicBaseDataset` also exposes to `task.ns`.

## Class: `TargetSpeakerExtractDataset`

### Constructor

```python
TargetSpeakerExtractDataset(
    metafile_path: str,
    min_utt_length_in_seconds: float = 3.0,
    min_utts_in_each_speaker: int = 5,
    target_sr: Optional[int] = None,
    training_sample_length_in_seconds: float = 6.0,
    enroll_speech_args: Optional[Dict] = None,
    audio_gain_normalized_to: Optional[int] = None,
    augmentation_speech_args=None,   # interferers: used/prob/add_n_cases(int)/snr_range/is_target
    augmentation_noise_args=None,
    augmentation_reverb_args=None,
    augmentation_speed_args=None,
    augmentation_ir_response_args=None,
    augmentation_src_args=None,
    augmentation_hpf_args=None,
    augmentation_volume_args=None,
    vad_label_args=None,
    dataset_role: str = "train",
)
```

There is **no `role_based_rir` parameter** -- the string does not occur
anywhere in this 788-line file, nor anywhere else in the repo. Foreground and
interferer reverb use the same mechanism as `task.ns`: a source-level
per-source RIR when `augmentation_reverb.simulator.source_level` is on, else
one whole-mix RIR applied after the interferer mix. `add_n_cases` is a plain
`int` here too (not the `[low, high]` range `task.ns`/`task.voice_isolation`
also accept).

### `enroll_speech_args` is required despite the `Optional` type hint

The constructor calls `self.init_enroll_augmentor()` unconditionally, which
immediately evaluates `self.enroll_speech_args["add_noise"]["used"]` --
passing the literal default, `None`, raises `TypeError: 'NoneType' object is
not subscriptable` before a single item is ever fetched. In practice
`enroll_speech_args` must always be a dict with every one of these top-level
keys present (each at least `{"used": False}` to turn it off):

| Key | Purpose |
|---|---|
| `enroll_length_seconds` | crop/pad length for the enrollment clip, in seconds |
| `gain_normalized_to` | RMS target level (dB) applied to the enrollment clip if truthy |
| `add_noise` | `{used, prob, noise_folder, snr_range, prob_white_noise, white_noise_snr_range}` |
| `add_reverb` | `{used, prob, rir_folder, target_rir_type}`, or a `simulator: {used: True, ...}` sub-block instead of `rir_folder` |
| `add_volume` | `{used, prob, clipping_prob, clipping_range: {min, max}, perturbed_range}` |
| `add_inactive_target` | `{used, prob}` -- see [below](#add_inactive_target) |

`egs/target_speaker_extraction/config/default_config.yaml`'s `enroll_speech:`
block is a complete, working example of this shape.

### `get_enroll_speech(target_speaker, batch_sr)`

Draws one utterance from `target_speaker`, then independently
crops/normalizes/reverberates/adds-noise/adjusts-volume it through its
**own** `AudioEffectAugmentor` (`self.enroll_augmentor`, loaded from
`enroll_speech_args`'s `add_noise`/`add_reverb` folders or simulator --
entirely separate from `self.augmentor`, which builds the mixture). This runs
first, before the mixture itself is built, so the enrollment clip's
augmentation draws never share RNG state with the mixture's.

### `__getitem__((speaker, sr)) -> Dict`

Only the 2-tuple form -- no seeded/deterministic 3-tuple item here (contrast
[task.ns](ns.md)); this dataset does not implement the
[`SpeakerSampler`](sampler.md) `item_seed` reseeding contract.

Order (verified top-to-bottom): fetch enrollment speech for the requested
speaker -> pick the foreground utterance for that same speaker -> optionally
replace the foreground with an unrelated speaker's speech
(`add_inactive_target`, see below) -> crop -> foreground reverb (source-level
RIR, or left dry for the whole-mix path) -> optionally mix in interferers
(each reverberated at source level, or left dry; summed and mixed in at one
SIR drawn from `augmentation_speech_args.snr_range`) -> clip guard -> speed ->
whole-mix RIR (only when the foreground reverb was *not* already
source-level) -> background noise (+ optional white noise) -> SRC ->
2nd-order IIR -> HPF -> volume -> final crop.

Returns:

| Key | Description |
|---|---|
| `noisy_speech` | `Tensor [1, T]`, the full mixture |
| `clean_speech` | `Tensor [1, T]`, the target's direct-path reference (all-zero on `add_inactive_target` rows, see below) |
| `enroll_speech` | `Tensor [1, T_enroll]`, the separately-augmented enrollment clip |
| `added_noise` | background noise actually added, or `None` |
| `consistency_noise` | `noisy_speech - clean_speech` |
| `speaker_id` | `self.spk2idx[target_speaker]` -- always the real enrolled speaker, including on `add_inactive_target` rows |
| `audio_sr`, `audio_length` | as in `task.ns` |
| `vad_target` | present only if `vad_label_args` is configured |

### `add_inactive_target`

When `enroll_speech_args["add_inactive_target"]["used"]` fires (probability
`prob`), the dataset swaps the foreground utterance for an unrelated,
randomly-chosen speaker's speech -- modeling "the enrolled voice never speaks
in this clip", a negative example that teaches the model to output silence.
The enrollment clip itself is unaffected: it was already fetched from the real
`target_speaker` before this swap.

At the end of `__getitem__`, the reference waveform is zeroed out:

```python
# Warp target speech to zeros
if inactive_target_speaker is not None:
    target_speech = torch.zeros_like(target_speech)
```

Same pattern as [`task.ns`](ns.md) (`puresound/task/ns.py`, ~line 441). On
rows where this branch fires:

- `clean_speech` is all-zero, so the waveform-level training target for these
  rows really is silence.
- `consistency_noise` (`noisy_speech - target_speech`) therefore equals the
  full mixture.
- `vad_target` comes out all-zero, built via
  `create_empty_vad_target(target_speech)` -- which only reads
  `target_speech.shape`, never its values.
- `speaker_id` is still the real enrolled speaker's index: the
  `target_speaker` id string is deliberately left untouched, since it is used
  as the `self.spk2idx[target_speaker]` lookup key a few lines later.

> **Fixed in this pass:** the zeroing line used to assign over `target_speaker`
> (the speaker-id string) instead of `target_speech`, despite its own comment.
> That left `clean_speech` un-zeroed *and* made every row drawing this branch
> raise `KeyError` at the `spk2idx` lookup, crashing the worker -- which is
> presumably why every recipe config in this repo ships
> `add_inactive_target: {used: False}`.

## Class: `TargetSpeakerExtractCollateFunc`

Pads the four waveform keys and renames two scalars, the same convention as
[`NoiseSuppressionCollateFunc`](ns.md):

| `__getitem__` key | collated batch key |
|---|---|
| `enroll_speech` | `conditional_speech` |
| `speaker_id` | `target` |
| `audio_sr` | `sr` |
| `audio_length` | `length` |

`noisy_speech`, `clean_speech`, `consistency_noise` keep their names.
`added_noise` is not collated (the same gap as `task.ns`). `vad_target` is
padded and included only if at least one item in the batch has it.

## Recipe wiring

The reference recipe is `egs/target_speaker_extraction/main.py` +
`egs/target_speaker_extraction/config/default_config.yaml`, pairing this
dataset with `EncDecCondMaskBase` ([system.miso](../system/miso.md)) -- the
enrollment clip is a second audio stream that needs its own encoder, exactly
the MISO case [system.siso](../system/siso.md)'s `EncDecMaskBase` docstring
distinguishes itself from.

```python
from puresound.task.tse import TargetSpeakerExtractDataset, TargetSpeakerExtractCollateFunc

dataset = TargetSpeakerExtractDataset(
    metafile_path="data/libri_train",
    min_utt_length_in_seconds=4.0,
    min_utts_in_each_speaker=5,
    target_sr=16000,
    training_sample_length_in_seconds=6.0,
    audio_gain_normalized_to=-28,
    enroll_speech_args={
        "enroll_length_seconds": 6,
        "gain_normalized_to": -28,
        "add_inactive_target": {"used": False, "prob": 0.1},
        "add_noise": {
            "used": True, "prob": 0.5, "noise_folder": "musan_noise/",
            "snr_range": [10, 30], "prob_white_noise": 0.1,
            "white_noise_snr_range": [10, 30],
        },
        "add_reverb": {"used": False, "prob": 0.5, "rir_folder": "rirs/",
                       "target_rir_type": "full"},
        "add_volume": {"used": False, "prob": 0.5,
                       "perturbed_range": [0.2, 1.2], "clipping_prob": 0.1,
                       "clipping_range": {"min": [0.0, 0.1], "max": [0.9, 1.0]}},
    },
    augmentation_speech_args={"used": True, "is_target": False, "prob": 1.0,
                               "add_n_cases": 2, "snr_range": [-20, 20]},
)
```
