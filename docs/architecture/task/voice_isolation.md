# puresound.task.voice_isolation

繁體中文版本：[`voice_isolation.zh-TW.md`](voice_isolation.zh-TW.md)

Near-field foreground voice isolation dataset: keep the talker close to the
device, suppress everything further away. It specialises the
[NoiseSuppressionDataset](ns.md) synthesis skeleton through its row-type hooks;
there is one synthesis path, and everything specific to the near/far decision
lives in overrides.

## Class: `VoiceIsolationDataset`

Adds three row types and one mixing block to the noise-suppression set. Each is
behind its own config block, and an absent or disabled block draws nothing from
the RNG stream, so a recipe without them regenerates bit-identically.

| keyword (recipe key) | row type |
| --- | --- |
| `augmentation_realfar_args` (`augmentation_realfar`) | **real-far rows**: the far interferers are finished loudspeaker-to-air-to-mic recordings from a pool manifest, inserted with no RIR applied. A convolved far channel carries only the linear time-invariant part of a capture chain; a real recording also carries its level, spectral tilt, transducer non-linearity and noise floor. `prob` is the share of rows; `lone_far_prob` is the share of those with no near talker (target = silence, the "lone far voice = suppress" example); `turn_taking_prob` is the row-level turn-taking rate. |
| `augmentation_realnear_args` (`augmentation_realnear`) | **real-near keep rows**: the foreground is a genuine close-mic recording and the target is that recording itself. Interferers come from the real-far pool, preferably the same room and never the same speaker, so real captured speech sits on the keep side of the same mixtures whose far side is real and the keep/suppress boundary rests on proximity cues rather than capture-chain identity. |
| `augmentation_session_rows_args` (`augmentation_session_rows`) | **session rows**: one user, one or two bystanders and the row's own floor, arranged as a script of turns, with per-frame and per-turn identity labels (`puresound.task.session_rows`). |
| `augmentation_speech_args.mix_mode` | explicit foreground/interferer level relationships (below). |

### Choosing the row type

`_plan_row` decides in this order, each draw guarded by its block:

1. **Session row** -- only when the block is enabled and the row is at least
   `min_seconds` long, tested before the probability draw so short length
   buckets draw exactly what they would without the block. The row is rendered
   in full here; the plan skips overlap gating and the whole-mix RIR, forces the
   interferers, and carries the speed change onto the background reference.
2. **Real-near row** -- draw with `prob`. The corpus utterance drawn upstream is
   discarded and replaced by a pool recording; no synthetic room is simulated
   and the whole-mix RIR is skipped. When a real-far pool is loaded, the row's
   interferers are forced and come from it; the target is always present.
3. **Real-far row** -- draw with `prob`, then decide lone-far (target-absent)
   with the block's own `lone_far_prob`, independently of the synthetic
   target-absent rate, so the two far-field sources can be tuned separately.
4. Otherwise the base plan (synthetic target-absent draw).

### Real-recording pools

A pool manifest is JSON Lines, one object per recording: `wav_path` (required),
`distance_m`, `room`, `speaker`, `mic`. `egs/voice_isolate/scripts/build_real_recording_pool.py`
builds them. With `stitch_to_length: true`, a take shorter than the row is
extended with further takes from the same `(speaker, room, mic)` -- same talker,
same chain -- instead of being zero-padded, which on a keep row would make half
the target digital silence. Real-far interferers report
`{"source_receiver_distance": distance_m, "origin": "real"}` as their channel
metadata; the interferer count comes from `augmentation_speech.add_n_cases`.

### `mix_mode`

`augmentation_speech.mix_mode.modes` is a list of `MixModeEntry`
(`name`, `prob`, `physical`, `distance_level`, `sir_range`, `jitter_db`). One mode
is drawn per row by `prob` weight (weights need not sum to 1; an empty list
means `physical`).

- **`physical`** sums the post-RIR signals without rescaling. Sources are
  RMS-normalised at load and every RIR is peak-normalised per channel, so the
  ratio lands near 0 dB; the distance cues that survive are DRR, tail shape and
  spectral tilt, not level.
- **`distance_level`** reinstates the level cue: SIR =
  20·log10(d_itf / d_fg) + jitter, from the scene's actual geometry and the
  nearest interferer. When either distance is unknown the row falls back to the
  base single-SIR draw.
- Any other mode draws its SIR from its own `sir_range`.

Real-far rows skip `mix_mode` and use the base SIR draw, because the level
ratio between a simulated near channel and a real far recording is not
physical. Session rows mix with their own SIR distribution and then add a
forced capture floor.

### Emitted labels

`_emit_task_metadata` adds per-row float scalars (`VOICE_ISOLATION_SCALAR_KEYS`,
NaN where undefined), computed from simulation metadata only -- they are
supervision and evaluation labels, never inference inputs:

`foreground_distance`, `foreground_drr`, `nearest_interferer_distance`,
`strongest_interferer_drr`, `drr_gap`, `rt60`, `n_interferers`,
`target_absent`, `target_present`, `has_background_speech`, `near_count`,
`far_count`, `mix_mode` (a code from `MIX_MODE_CODES`: none 0, legacy 1,
physical 2, moderate 3, counter_level 4, distance_level 5, session 6),
`realized_speech_sir`, `noise_snr`, `overlap_fraction`, `turn_taking`.

They feed auxiliary heads (for example
[`DistHeadRegressionLoss`](../../algorithms/losses/dist.md)) and evaluation
bucketing. When the session block is enabled, every row -- session or not --
also carries the session label contract (`SESSION_LABEL_KEYS`: `user_active`,
`bystander_active`, `turn_id`, `turn_role`, `turn_speaker`, `turn_chain`,
`turn_distance`, `row_source_id`; and the `SESSION_SCALAR_KEYS` diagnostics),
which the identity and proximity losses read. Session rows long enough for
`paired_view_min_seconds` are selected for a second capture view with
`paired_view_prob`.

## Class: `VoiceIsolationRowPlan`

`RowPlan` extended with `use_realnear`, `use_realfar`, the real-near pick
(`realnear_room`, `realnear_speaker`, `realnear_fg_metadata`) and `session`
(the rendered `SessionRender`, `None` on every other row type).

## Class: `VoiceIsolationCollateFunc`

`NoiseSuppressionCollateFunc` plus:

- the scalar labels above and the session scalars, one tensor per key;
- the session per-frame and per-turn labels (`collate_session_labels`);
- `far_target`, padded like the other waveforms (zeros for rows without one);
- `background_vad_target` / `background_vad_reference`, zero-filled for rows
  that have none whenever any row in the batch has one.

## Recipe wiring

`task: voice_isolation` is a top-level recipe key; it selects
`VoiceIsolationRecipe`, the only schema with the real-recording and session
blocks.

```yaml
task: voice_isolation      # driver: egs/voice_isolate/main.py
augmentation_realfar:
  used: True
  prob: 0.2
  lone_far_prob: 0.15
  turn_taking_prob: 0.3
  pool_manifest: data/realfar_pool/voices.train.jsonl
  stitch_to_length: True
augmentation_realnear:
  used: True
  prob: 0.15
  turn_taking_prob: 0.5
  pool_manifest: data/realfar_pool/voices.near.train.jsonl
  stitch_to_length: True
augmentation_session_rows:
  enabled: True            # this block uses `enabled`, not `used`
  prob: 0.5
  min_seconds: 12.0
```

The recipe refuses `augmentation_session_rows` together with
`augmentation_speech.is_target: true` (the bystanders would become part of the
target) or with `augmentation_row_initial_ambient` (the lead would erase speech
the turn script claims). The released recipes in `egs/voice_isolate/config/`
show the full wiring.
