# Engineering contracts

繁體中文版本：[`engineering_contract.zh-TW.md`](engineering_contract.zh-TW.md)

The preceding chapters describe the techniques themselves. This chapter
describes the cross-cutting rules that make them reproducible, comparable, and
safely extensible. These rules are scattered across module docstrings; here
they are collected in one place.

## 1. RNG determinism contracts

The synthesis pipeline shares three global random number generators: Python's
`random`, NumPy's `np.random`, and PyTorch's global generator. There are four
contracts, honoured by `DeviceChain`, `NoiseStage`, `OverlapGating`, and
`AudioEffectAugmentor` alike.

### 1.1 No draw unless the stage fires

Every probability gate has the same shape:

```python
block is not None and block.used and torch.rand(1) < block.prob
```

The probability draw sits **inside** the short circuit.

**Effect**: after disabling a block (or when a dataset simply has no such
block), every later stage draws exactly the random numbers it drew before.
Therefore **adding a knob does not change the data produced by any recipe that
does not enable it** — old recipes regenerate bit-identically.

**Why this contract is worth enforcing so strictly.** If the probability draw
sat outside the short circuit (drawing first, then checking `used`), disabling
one stage would shift the random numbers of every stage after it. The
consequence would be that adding even a default-off knob changes the output of
every existing recipe, and no checkpoint could be compared with anything
trained before it. This contract is the precondition for knobs accumulating
safely.

### 1.2 Draw order is load-bearing

Swapping two stages, or even two draws within one stage, shifts the shared RNG
stream and makes the same seed produce different data. "Each stage's own
implementation is unchanged" is not a reason the order may be adjusted.

One easily overlooked case: `OverlapGating._draw_overlap_rate` **consumes a
variable number of draws depending on the regime**
([ch7](scene_construction.md) §3.2). The zero-overlap regime consumes one random
number; the other two consume two. Consequently even the semantic order of the
regime thresholds cannot be swapped.

### 1.3 Randomness rules for retry paths

There are two classes of retry, with different rules:

* **`apply_linear`'s escalation must not consume randomness.** It re-calls the
  backend, which is why the `fn` passed in must be a pure filter or gain
  ([ch5](spectral_channel.md) §7.4). Where random parameters are needed, draw
  them outside and pass them in via a closure.
* **Bounded sampling retries do consume randomness, and that is intended.**
  Redrawing a speaker whose sample rate does not match, or recropping to avoid
  an all-silent window ([ch7](scene_construction.md) §2.1), are part of the
  sampling logic rather than error recovery. Their counts are bounded, so their
  consumption is bounded.

### 1.4 A seeded item must reseed all three streams

When an item key carries an `item_seed`, `DynamicBaseDataset.parse_item_key`
reseeds all three streams for every task:

```python
random.seed(item_seed)
np.random.seed(item_seed % (2**32))
torch.manual_seed(item_seed)
```

All three must be reseeded, because the pipeline uses all three: room geometry
sampling goes through NumPy, speaker and utterance selection through Python's
`random`, and most probability gates and range draws through PyTorch. Miss any
one and the row cannot regenerate bit-identically.

`% (2**32)` is NumPy's seed range limit.

This is the mechanism behind the deterministic validation set: the same item
produces identical data across epochs, runs, and worker layouts.

## 2. The synthesis order table

The complete station order for one row:

| # | Step | Implementation | Ch |
|---|---|---|---|
| 1 | Load and RMS rescale | `AudioIO.open(target_lvl=...)` | ch4 §1 |
| 2 | Crop/align (with bounded anti-silence retry) | `align_audio_list` | ch7 §2.1 |
| 3 | Row plan (target_absent / realnear / realfar) | `_plan_row` | ch7 §1 |
| 4 | Foreground channel (source-level RIR: full → mix, early → target) | `_prepare_foreground` | ch2 §4 |
| 5 | Interferer sampling → media coloring → per-source RIR | `_sample_interferers` | ch7 §2 / ch5 §6 |
| 5a | (Before the RIR is cached: DRR contrast → direct smear) | `Augmentor.apply_rir` | ch3 §3–4 |
| 6 | Overlap gating (Bernoulli / turn-taking) | `OverlapGating.apply` | ch7 §3 |
| 7 | Foreground × interferer mixing (hard SIR / mix_mode) | `_mix_foreground_with_interferers` | ch7 §4 |
| 8 | Target-absent subtraction | `__getitem__` | ch7 §1.2 |
| 9 | Echo playback (ERLE) | `__getitem__` | ch7 §5 |
| 10 | Paired clipping guard | `avoid_audio_clipping` | ch4 §1 |
| 11 | Speed perturbation (paired) | `sox_speed_perturbed` | ch5 §5 |
| 12 | Whole-mix RIR (non source-level rows only) | `Augmentor.apply_rir` | ch2 §6 |
| 13 | The three noise sources (recorded → white → floor) | `NoiseStage.apply` | ch7 §6 |
| 14 | VAD reference snapshot | `__getitem__` | ch7 §6 / ch6 |
| 15 | Device chain (analogue group → A/D → transmission group) | `DeviceChain.apply` | ch6 |
| 16 | Truncate to `sample_length`, assemble the sample dict | `__getitem__` | — |

Several ordering rationales, collected here:

* **Step 5a must precede the cache**: the target re-fetches the same RIR by
  `rir_id`; see ch2 §6.
* **Step 8 must precede any rescaling**: the subtraction uses a pre-rescale
  snapshot; see ch7 §1.2.
* **Step 14 sits after speed and before noise**: the time axis must match the
  mixture, and the labels must be computed on clean speech.
* **Step 13 precedes step 15**: the noise floor must pass through the device
  response together with the speech; see ch7 §6.3.

## 3. Config-to-code mapping

All schemas are defined in `puresound/config/augmentation.py` using Pydantic in
strict mode. Two validation principles:

**An unknown key is an error, not a default.** Two failure modes motivated this:
a typo (`porb: 0.5`) used to leave the block silently running at probability
zero; and deleting a mechanism does not delete its knobs (historically, ten
knobs outlived their code in 33 config files). Full discussion in
`docs/configuration.md`.

**Fields required when `used: true` are validated by `enabled_contract`.** This
makes "enabled but unparameterised" fail at load time rather than yielding a
`None` mid-training.

### Block mapping

| Config block | Schema | Consumer |
|---|---|---|
| `augmentation_speech` (with `media_voice`, `echo_playback`, `overlap_control`, `mix_mode`) | `SpeechAugmentation` | `ns.py` / `voice_isolation.py` / `OverlapGating` |
| `augmentation_reverb` (with `simulator`, `pregenerated`, `drr_contrast`, `direct_smear`) | `ReverbAugmentation` | `dynamic_base` init plus `Augmentor` |
| `augmentation_noise` (with `room_coloring`, `absolute_floor`) | `NoiseAugmentation` | `NoiseStage` |
| `augmentation_src` / `ir_response` / `hpf` / `volume` / `compressor` / `codec` / `packet_loss` | Individual schemas | `DeviceChain` |
| `augmentation_speed` | `ContinuousSpeedAugmentation` (enhancement tasks) / `DiscreteSpeedAugmentation` (speaker tasks) | `ns.py` / speaker tasks |
| `augmentation_target_absent` | `TargetAbsentAugmentation` | `_plan_row` |
| `augmentation_realfar` / `augmentation_realnear` | `RealFarAugmentation` / `RealNearAugmentation` | `voice_isolation.py` (that task only) |
| `augmentation_vad_label` | `VadLabelConfig` | VAD labeler (backend-specific options go through `args`) |

### Block ownership principles

**A block exists only in the task that actually consumes it.** `mix_mode` raises
on a plain NS dataset; `augmentation_realfar` is a field of `voice_isolation`
alone. The task discriminator decides which model is used.

**Blocks handed wholesale to a constructor forward only the keys the recipe
actually wrote.** The bank loader, room simulator, and VAD labeler are in this
category. This keeps those components' own defaults in effect rather than
having them overridden by the config model's defaults — `delegated_kwargs`
implements the behaviour.

**A schema field default *is* the behavioural default.** Datasets read them by
attribute access, so the default written in the schema is the value the pipeline
actually uses; there is no second set of defaults.

## 4. Distribution versioning and checkpoint comparability

This pipeline's output **is** the training distribution. Any modification that
changes the output produces a new distribution, and **synthetic-domain metrics
are not directly comparable across distributions** — a difference may come from
the model or from the data.

"Changes the output" is broader than intuition suggests. It includes:

* fixing a bug;
* making an imprecise approximation exact;
* adjusting a sampling order;
* changing any stage's default parameters.

### 4.1 Rules

**Treat it as a break.** For a modification that changes synthesis output,
checkpoints trained either side of it can only be compared on **evaluations
that do not pass through the synthesis pipeline** (real-recording gates).
Synthetic-domain metrics require a retrained control to be meaningful. Record
break points by commit.

**Prefer a default-off knob.** The contract in §1.1 guarantees that recipes not
enabling the knob are unaffected, so the new distribution is opt-in and forces
no break. This is the preferred way to extend the pipeline.

**Mark unavoidable breaks explicitly.** The commit message must state that the
distribution changed and which checkpoints fall on which side of the line.

### 4.2 Where experimental records live

Which break happened when, and which checkpoints belong to which distribution,
are experimental records and are not in this handbook. They live in
`egs/voice_isolate/benchmarks/`.

## 5. The test net

| Test | What it guards |
|---|---|
| `test/test_utils/test_synthesis_fingerprint.py` | The synthesis output fingerprint; the first thing to fail when the distribution changes unintentionally |
| `test/test_utils/test_device_chain.py` | Stage order, target following, RNG contracts, provenance scalars |
| The `test/test_rir_*.py` family | RIR generation, calibration, bank contracts (list in [ch2](room_acoustics.md) §9) |
| Per-task dataset tests | Row types, gating behaviour, seeded reproduction |

The fingerprint test is the core of this net: it turns "did the distribution
change" from a human judgement into a CI judgement. When a modification makes
it fail, the author must explicitly decide whether this is an intended break
(update the fingerprint and record it) or an accident (revert).

## 6. Checklist for adding a knob

1. **Schema**: add the field and its `enabled_contract` (nothing else should be
   required when `used: false`).
2. **RNG contract**: put the probability draw inside the short circuit; add a
   test asserting that an old recipe's fingerprint is unchanged when the knob is
   off.
3. **Application point**: confirm its correct position in the order table of §2,
   particularly the three constraints concerning the cache, subtraction, and
   rescaling (end of §2).
4. **Provenance**: if eval needs to bucket by it, add it to the relevant scalar
   list — and first establish who will read it
   ([ch6](device_chain.md)'s `RIR_PROVENANCE_KEYS` is the counterexample).
5. **Linearity check**: if the stage models a linear phenomenon and acts on a
   mixture/target pair, confirm it is wrapped in `apply_linear` or uses the
   backend's clamp switch ([ch5](spectral_channel.md) §7).
6. **Documentation**: add a section to the corresponding chapter here
   (algorithm and engineering); add the signature to the API reference
   (`docs/audio/`, `docs/task/`).

## 7. Editorial rules for this handbook

* The scope is the **techniques themselves**: derivations, model assumptions,
  the physical meaning of parameters, implementation contracts, and known
  pitfalls.
* **Experimental conclusions, verdicts, and measurement numbers are not
  recorded here**; they live in `egs/voice_isolate/benchmarks/`. Where a
  technique's design motivation came from a measurement, cite the source (probe
  README, docstring, benchmarks path) without restating the numbers.
* Code comments likewise carry no experimental traces (repo convention).
  Existing measurement citations in docstrings are the design-rationale record
  for that knob; before moving or deleting one, confirm benchmarks holds the
  same content.
