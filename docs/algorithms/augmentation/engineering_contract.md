# Engineering contract

繁體中文版本：[engineering_contract.zh-TW.md](engineering_contract.zh-TW.md)

The rules that make synthesis reproducible, comparable and safe to extend. They
hold across `DeviceChain`, `NoiseStage`, `OverlapGating`, `AudioEffectAugmentor`
and the dataset `__getitem__` paths; the module docstrings state them locally,
this page collects them.

## 1. RNG contracts

Synthesis draws from three global generators: Python `random`, NumPy
`np.random` and the PyTorch global generator.

### 1.1 No draw unless the stage fires

Every probability gate has the same shape:

```python
block is not None and block.used and torch.rand(1) < block.prob
```

The probability draw sits inside the short circuit. Disabling a block, or a task
not having it, leaves every later stage drawing exactly what it drew before, so
**adding a knob does not change the data of any recipe that does not enable it**.
If the draw sat outside (draw first, then check `used`), even a default-off knob
would shift every later draw and change every existing recipe's output. This
contract is what lets knobs accumulate.

### 1.2 Draw order is part of the output

Swapping two stages, or two draws inside a stage, shifts the shared stream, and
the same seed produces different data even though each stage is unchanged. Some
stages consume a variable number of draws: `OverlapGating._draw_overlap_rate`
takes one value in the zero-overlap regime and two in the others, so even the
order of its regime thresholds cannot be changed
([Scene construction](scene_construction.md)).

### 1.3 Retries

- **`apply_linear` escalations draw nothing.** An escalation calls the backend
  again, so the function passed in must be a pure filter or gain; random
  parameters are drawn outside and closed over
  ([Spectral and channel effects](spectral_channel.md)).
- **Bounded sampling retries do draw, by design.** Redrawing an interferer
  utterance whose sample rate does not match (up to 5 times), re-cropping an
  all-silent window in `align_audio_list` (up to 10 times) and reopening an
  empty utterance (up to 5 times, then an error) are part of sampling, not error
  recovery. Their consumption is bounded because their count is.

### 1.4 A seeded item reseeds all three streams

`DynamicBaseDataset.parse_item_key` accepts
`(speaker, sample_rate[, seed[, seconds[, epoch]]])` and applies, in order: the
row length, the epoch's curriculum values, then the seed:

```python
random.seed(seed)
np.random.seed(seed % (2**32))
torch.manual_seed(seed)
```

All three are needed: room geometry is sampled through NumPy, speakers and
utterances through `random`, most gates and range draws through PyTorch.
`% 2**32` is NumPy's seed range. Applying the curriculum first means every
seeded draw already sees the epoch's values; re-composing the stages for a new
epoch (`rebind_augmentation_blocks`) draws nothing. This is what makes the
validation set deterministic across epochs, runs and worker layouts. Utterance
pools are ordered lists, never sets, so a seed picks the same utterance in every
process regardless of `PYTHONHASHSEED`.

## 2. Synthesis order

One row of `NoiseSuppressionDataset.__getitem__` (voice isolation substitutes
its own row types through the hooks named in the table):

| # | Step | Implementation | Page |
|---|---|---|---|
| 1 | load, optional RMS rescale | `choose_an_utterance_by_speaker_name` → `AudioIO.open(target_lvl=...)` | [Level and dynamics](level_dynamics.md) |
| 2 | crop or pad to the row length (bounded anti-silence retry) | `align_audio_list` | [Scene construction](scene_construction.md) |
| 3 | row plan (target-absent; in VI also real-recording and session rows) | `_plan_row` | [Scene construction](scene_construction.md) |
| 4 | foreground channel (source-level RIR: `full` for the mixture, the target window for the target) | `_prepare_foreground` | [Room acoustics](room_acoustics.md) |
| 5 | interferers: sample, media coloring, per-source RIR | `_sample_interferers` | [Scene construction](scene_construction.md), [Spectral and channel effects](spectral_channel.md) |
| 5a | inside the RIR draw, before caching: DRR contrast, then direct smear | `AudioEffectAugmentor.apply_rir` | [Distance cues](distance_cues.md) |
| 6 | overlap gating (Bernoulli or turn taking), unless the plan skips it | `OverlapGating.apply` | [Scene construction](scene_construction.md) |
| 7 | foreground × interferer mix (hard SIR; `mix_mode` in VI) | `_mix_foreground_with_interferers` | [Scene construction](scene_construction.md) |
| 8 | target-absent subtraction | `__getitem__` | [Scene construction](scene_construction.md) |
| 9 | playback echo at an ERLE | `__getitem__` | [Scene construction](scene_construction.md) |
| 10 | paired peak guard | `avoid_audio_clipping` | [Level and dynamics](level_dynamics.md) |
| 11 | speed perturbation on the pair | `sox_speed_perturbed` | [Spectral and channel effects](spectral_channel.md) |
| 12 | whole-mix RIR (rows without source-level reverb) | `AudioEffectAugmentor.apply_rir` | [Room acoustics](room_acoustics.md) |
| 13 | row-initial ambient lead: mask all speech over the first seconds | `__getitem__` | [Scene construction](scene_construction.md) |
| 14 | noise: recorded, then white, then the capture floor | `NoiseStage.apply` | [Scene construction](scene_construction.md) |
| 15 | VAD reference snapshot of the target | `__getitem__` | [Device chain](device_chain.md) |
| 16 | device chain (analogue group, converter, transmission), plus an optional paired view | `apply_chain_views` → `DeviceChain.apply` | [Device chain](device_chain.md) |
| 17 | crop to the row length, VAD labels, sample dict, provenance | `__getitem__` | — |

Ordering constraints:

- **5a before the cache.** The target re-fetches the same RIR by `rir_id` to take
  a different window of it; a change applied after caching would give mixture and
  target different impulses.
- **8 before any rescale.** The subtraction uses a snapshot of the foreground's
  contribution taken before any level change.
- **13 after 11 and 12, before 14 and 15.** Timings are final, the lead is filled
  with the row's own ambience rather than digital silence, and the labels inherit
  the mask.
- **15 before 16.** Labels are computed on the clean target, after speed (so the
  time axis matches the mixture) and before any chain damage.
- **14 before 16.** The noise floor goes through the device response with the
  speech.

## 3. Configuration and code

All augmentation schemas are Pydantic models in `puresound/config/augmentation.py`,
built on `StrictConfig` (`extra="forbid"`).

- **An unknown key is an error.** A misspelled key fails at load time instead of
  leaving a block at its default, and a recipe that still carries the knobs of a
  removed mechanism fails loudly. See
  [Configuration](../../usage/configuration.md).
- **Fields required when enabled are validated at load.** Each block's
  `enabled_contract` (via `require_fields_when_enabled`) rejects `used: true`
  without its parameters, instead of a `None` surfacing mid-training.
- **A schema default is the behavioural default.** Datasets read blocks by
  attribute and use the value as given; there is no second set of defaults.
- **A disabled block reaches the dataset as `None`.**
  `BaseRecipe.augmentation_kwargs` forwards each `augmentation_*` block (and
  `vad_label`) as `<name>_args`, replacing a block with `used: false` by `None`.

### Blocks

| YAML block | Schema | Consumer | Tasks |
|---|---|---|---|
| `augmentation_speech` (`media_voice`, `echo_playback`, `overlap_control`, `mix_mode`) | `SpeechAugmentation` | `ns.py`, `voice_isolation.py`, `OverlapGating` | NS, VI (`mix_mode` VI only) |
| `augmentation_reverb` (`simulator`, `simulator.pregenerated`, `drr_contrast`, `direct_smear`) | `ReverbAugmentation` | `DynamicBaseDataset` initialisation, `AudioEffectAugmentor` | all |
| `augmentation_noise` (`room_coloring`, `absolute_floor`, `noise_sources`, `snr_bands`) | `NoiseAugmentation` | `NoiseStage` | all |
| `augmentation_src`, `augmentation_ir_response`, `augmentation_hpf`, `augmentation_volume`, `augmentation_compressor` | per-stage schemas | `DeviceChain` | NS, VI, TSE |
| `augmentation_codec`, `augmentation_packet_loss` | `CodecAugmentation`, `PacketLossAugmentation` | `DeviceChain` | NS, VI |
| `augmentation_speed` | `ContinuousSpeedAugmentation`; `DiscreteSpeedAugmentation` for speaker embedding | `ns.py`, task datasets | all |
| `augmentation_target_absent` | `TargetAbsentAugmentation` | `_plan_row` | NS, VI |
| `augmentation_row_initial_ambient` | `RowInitialAmbientAugmentation` | `__getitem__` | NS, VI |
| `augmentation_realfar`, `augmentation_realnear` | `RealFarAugmentation`, `RealNearAugmentation` | `voice_isolation.py` | VI |
| `augmentation_session_rows` | `SessionRowsConfig` (switch named `enabled`) | `task/session_rows.py` | VI |
| `vad_label` | `VadLabelConfig` (backend options under `args`) | VAD labeler | all |
| `curriculum` | `CurriculumConfig` (`puresound/config/curriculum.py`) | per-epoch knob schedule, applied in `parse_item_key` | all |

### Ownership

- **A block exists only in a task that consumes it.** The recipe model of each
  task declares its blocks; `augmentation_speech.mix_mode` on a
  noise-suppression recipe is rejected, and `augmentation_realfar` is a field of
  the voice-isolation recipe only. `augmentation_session_rows` and
  `augmentation_row_initial_ambient` are mutually exclusive.
- **Blocks handed to a component constructor forward only the keys the recipe
  wrote.** The room simulator, bank loaders, VAD labelers, DRR contrast and
  direct smear receive `delegated_kwargs(block)` (`exclude_unset=True`), so the
  component's own defaults apply to everything else.

## 4. Distribution changes and comparability

The synthesis output is the training distribution. A change to that output is a
new distribution, and synthetic-domain metrics are not comparable across it: a
difference may come from the model or from the data. "Changes the output"
includes fixing a bug, making an approximation exact, reordering draws and
changing a default.

- **Prefer a default-off knob.** By §1.1 recipes that do not enable it are
  unaffected, so the new distribution is opt-in.
- **Treat an unavoidable change as a break.** Checkpoints trained on either side
  can only be compared on evaluations that do not pass through synthesis (real
  recordings); synthetic metrics need a retrained control. The commit message
  states that the distribution changed and which checkpoints fall on which side.
- Which break happened when is recorded in the commit history, not here.

## 5. Tests

| Test | Guards |
|---|---|
| `test/task/test_synthesis_fingerprint.py` | a fully spelled-out recipe and a minimal one synthesise the same audio (a default lives in one place); a seeded item is a pure function of its seed |
| `tools/rng_fingerprint.py` | before/after comparison of a real recipe for refactors that must not change synthesis; catches changes that shift both recipes of the test above together |
| `test/task/test_device_chain.py` | stage order, which signals each stage touches, no draws when disabled, linearity of the analogue path, provenance keys |
| `test/task/test_noise_stage.py`, `test/task/test_overlap_gating.py` | no draws when a source or gate does nothing, reported SNR and overlap |
| `test/audio/test_dsp.py` | `apply_linear`, resampler linearity, `compressor_gain` properties |
| `test/config/test_config_schema.py` | unknown keys rejected at every nesting level, required-when-enabled fields, blocks owned by the right task, disabled blocks reaching the dataset as `None` |
| `test/audio/`, `test/task/` | individual knobs (DRR contrast, direct smear, noise sources, row-initial ambient, session rows) |
| `test/rir/` | RIR generation, calibration and bank contracts |

When a change makes the fingerprint test fail, the author decides explicitly
whether it is an intended break (update the expected values and record the
break) or an accident (revert).

## 6. Adding a knob

1. **Schema.** Add the field and its enabled contract; nothing else may be
   required when `used: false`.
2. **RNG.** Put the probability draw inside the short circuit, and confirm with
   `tools/rng_fingerprint.py` that a recipe without the knob is unchanged.
3. **Position.** Place it in the order of §2 and check the constraints listed
   there (cache, subtraction, labels).
4. **Provenance.** If evaluation needs to bucket by it, add it to the relevant
   scalar list, and name who reads it before adding it.
5. **Linearity.** A stage that models a linear device and acts on a
   mixture/target pair uses `clamp=False` or `apply_linear`.
6. **Documentation.** Describe the algorithm on the matching page under
   `docs/algorithms/augmentation/` and the dataset behaviour under
   `docs/architecture/task/`, in both languages. Comments and documentation
   follow the rules in [Repository layout](../../repository_layout.md): no
   results, dates or version names; where evidence is needed, point to the test
   that pins the behaviour or to a paper.
