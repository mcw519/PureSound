# v20 R1a, data side: session rows -- what was built and what it measures

2026-09-04. Deliverable: the **data half** of stage R1a in
`../v20_self_enrolling_foreground_design.md` (§4.1 is the mandate; review items
5 and 6 are the constraints). The heads and the two new loss terms are the other
half; nothing in this directory implements them, but §8 checks the batch keys
against what they consume and §11 pre-flights the recipe that switches them on.

Nothing in this directory is trained. Every number below is measured on rows
drawn through the real recipe dataloader.

---

## 0. Why a row type, and what the old rows look like

`../v19c_diagnostics/training_data_audit/tables.md`, 1206 rows of the shipped
v16 recipe: the target's onset is inside 0.5 s on **90.7 %** of rows, an
interferer holds the floor for >= 1 s before it on **6.0 %**, the longest
target-free gap is **0.53 s** at the median and >= 5 s on **0.3 %**, re-entry
after >= 5 s on **0.3 %**, **no row renders the same talker twice**, and the
realised SIR is **+0.5 dB** at the median with p5 -8.1. Under that distribution
"whoever is already talking, keyed on recent near-like sound" is a low-cost
solution -- which is the behaviour the record says the model learned (H1/H2 of
the design). To change the model, change what is cheap.

A **session row** is the same synthesis machinery arranged as a conversation:
one user U (near draw) taking several turns and sometimes changing seat, one or
two bystanders B (far draw, or -- on a share of rows -- in U's own distance
class), the row's own floor in the gaps, and per-frame / per-turn labels on the
`vad_target` grid so a loss can pool a turn and know whose it was.

---

## 1. Files

| file | what it is |
|---|---|
| `puresound/task/session_rows.py` | **new.** The row type: script builder, renderer, label emitter, collate helper. Holds the contracts (one chain draw per row; a gap is floor, never digital zeros; a long user gap is never target-absent). |
| `puresound/config/augmentation.py` | **+2 models.** `SessionRowsConfig`, `SessionTurnShapeConfig`. Validated, every knob defaulted, `enabled: False` by default. |
| `puresound/config/recipe.py` | **+1 field, +1 validator.** `VoiceIsolationRecipe.augmentation_session_rows`, and a refusal to load a session recipe together with `augmentation_speech.is_target` or `augmentation_row_initial_ambient`. |
| `puresound/task/ns.py` | **hooks only.** Three `RowPlan` fields (`skip_overlap_gating`, `speed_perturb_companions`, `speed_factor`), a guard on the overlap-gating call, the speed factor recorded plus the companion perturbation, and a no-op `_emit_row_labels` hook. |
| `puresound/task/voice_isolation.py` | **wiring.** Registry entry, plan field, and the five hook overrides; `MIX_MODE_CODES["session"]`; the collate calls. |
| `test/test_task/test_session_rows.py` | **new.** 31 tests (§7). |
| `config/exp/train_dpcrn_v20_r1a.yaml` | **new.** v16 verbatim + the block, and (§11) the two heads and two loss terms the other half of R1a adds. |
| `benchmarks/probes/v20_session_rows/sample_rows_v20.py` | the audit's method plus the session and label columns. |
| `benchmarks/probes/v20_session_rows/aggregate_v20.py` | the tables below. |
| `benchmarks/probes/v20_session_rows/batch_identity.py` | the default-off byte-identity check (§6). |
| `benchmarks/probes/v20_session_rows/memory_smoke.py` | the peak-memory smoke test (§5). |
| `benchmarks/probes/v20_session_rows/loss_reach.py` | calls the two new loss terms on real batches: how often is each actually non-zero (§8.1). |

---

## 2. The knobs, as set in `config/exp/train_dpcrn_v20_r1a.yaml`

```yaml
augmentation_session_rows:
  enabled: True
  prob: 0.5                # of ELIGIBLE rows
  min_seconds: 12.0        # eligibility: the 12 s and 30 s buckets only
  max_seconds: 60.0        # a no-op today; the cap review item 5 asked for
  n_bystanders: [1, 2]
  user_distance_range: [0.3, 1.0]
  bystander_distance_range: [1.5, 4.0]
  shape_probs: {user_first: 0.25, bystander_first: 0.30, user_gap: 0.30, overlap: 0.15}
  user_gap_seconds: [5.0, 20.0]
  bystander_open_seconds: [2.0, 8.0]
  rir_move_prob: 0.3
  distance_matched_bystander_prob: 0.2
  sir_range: [-5.0, 10.0]
  sir_low_tail_prob: 0.30
  sir_low_tail_range: [-10.0, -5.0]
  floor_dbfs_range: [-60.0, -45.0]
  pair_prob: 0.25
  pair_pool_size: 256
```

`min_seconds` is the load-bearing one. Eligibility is tested **before** the
probability draw, so the 3 s and 6 s buckets of this recipe are bit-identical to
v16's (§6) -- the short-context behaviour v16 exists to protect is trained on
exactly the rows that trained it. `trainer.length_schedule` is untouched, which
is why §5's diversity budget is unchanged and why v15's budget-shortfall
confound is not re-imported.

---

## 3. Temporal shape: v16 vs R1a

Method: `sample_rows_v20.py`, 652 rows each (59 consecutive train batches),
**seed 7**, 8 dataloader workers, real corpus / real RIR bank / real noise
folder. Detector = the pipeline's own labels (`vad_target`,
`background_vad_target`) on the 100 fps grid. `[p10,p90]` after each median.
The two samples are independent draws from the two distributions, not paired
rows: once a session row is drawn, that worker's RNG stream diverges.

| column | v16 | R1a | audit reference (v16, n=1206) |
|---|---|---|---|
| n | 652 | 652 | 1206 |
| target onset, s | 0.00 [0.00,0.41] | 0.00 [0.00,0.83] | 0.00 [0.00,0.44] |
| onset <= 0.5 s | 0.901 | **0.874** | 0.907 |
| onset >= 1 s | 0.064 | **0.094** | 0.066 |
| interferer before onset, s | 0.00 [0.00,0.24] | 0.00 [0.00,0.51] | 0.00 [0.00,0.33] |
| **interferer >= 1 s before onset** | 0.056 | **0.078** | 0.062 |
| longest target-free gap, s | 0.53 [0.00,1.87] | 0.58 [0.04,3.25] | 0.53 [0.01,1.81] |
| longest *interior* gap, s | 0.36 [0.00,1.21] | 0.39 [0.00,1.51] | 0.41 [0.00,1.22] |
| gap >= 5 s | 0.003 | **0.051** | 0.003 |
| interior gap >= 5 s | 0.002 | **0.027** | -- |
| **re-entry after >= 5 s** | 0.002 | **0.027** | 0.003 |
| target active fraction | 0.81 [0.54,1.00] | 0.79 [0.35,0.99] | 0.81 [0.51,1.00] |
| target-absent share | 0.0215 | 0.0383 | 0.0340 |

Per length bucket (the block can only fire on 12 s and 30 s):

| bucket | n | onset>=1s v16 -> R1a | itf>=1s pre v16 -> R1a | gap>=5s v16 -> R1a | reentry>=5s v16 -> R1a |
|---|---|---|---|---|---|
| 3 s | 320 / 320 | 0.054 -> 0.066 | 0.045 -> 0.047 | 0.000 -> 0.000 | 0.000 -> 0.000 |
| 6 s | 240 / 240 | 0.064 -> 0.077 | 0.060 -> 0.068 | 0.000 -> 0.004 | 0.000 -> 0.000 |
| 12 s | 72 / 72 | 0.127 -> 0.250 | 0.113 -> 0.222 | 0.014 -> 0.347 | 0.000 -> 0.194 |
| 30 s | 20 / 20 | 0.000 -> 0.150 | 0.000 -> 0.150 | 0.050 -> 0.300 | 0.050 -> 0.150 |

And the session rows themselves, inside the R1a sample (n = 47):

| column | session rows only |
|---|---|
| row length, s | 12.00 [12.00,30.00] |
| onset <= 0.5 s | 0.638 |
| onset >= 1 s | 0.362 |
| interferer >= 1 s before onset | 0.319 |
| longest target-free gap, s | 5.65 [2.64,9.67] |
| gap >= 5 s | 0.638 |
| re-entry after >= 5 s | 0.340 |
| target active fraction | 0.33 [0.19,0.57] |
| target-absent | 0.000 |

Read the whole-distribution column as "9 % of rows moved, so the aggregate moves
by ~9 % of the way", and the bucket rows as what actually happened where the
block fires. The three shapes the design asked for are all present at
non-trivial rates on the rows that carry them: bystander-first at 0.32 of
session rows (against a 0.06 base rate over all v16 rows), a >= 5 s user gap at
0.34, and re-entry after >= 5 s at 0.34 (against 0.003).

### 3.1 The target-absent share

The design requires this one to be *unchanged*: raising the target-absent rate
cost Dawn WER 0.392 -> 0.626 and deletion 0.286. Structurally it is untouched --
no session row is ever target-absent (measured: **0 of 297** session rows across
all three passes), and the `augmentation_realfar.lone_far_prob` draw that
produces lone-far rows is not on the session path. Measured:

| sample | n | target-absent share |
|---|---|---|
| v16, seed 7, this probe | 652 | 0.0215 (14) |
| R1a, seed 7, this probe | 652 | 0.0383 (25) |
| v16, seed 7, the audit | 1206 | 0.0340 (41) |
| **v16, seed 21** | 1610 | **0.0391 (63)** |
| **R1a, seed 21** | 1610 | **0.0366 (59)** |

The seed-21 pair is the pre-registered decision sample (1600 rows requested,
1610 delivered -- the probe stops on a completed batch). It settles what the
seed-7 pair could not: at n = 652 the two shares were 0.0215 and 0.0383, which
is a difference of 0.017 with a 95 % interval of +-0.017, i.e. no measurement at
all. At n = 1610 the difference is **-0.0025 +- 0.0132** (95 %, two independent
binomials) -- R1a's share is if anything the *lower* of the two, and both sit on
the audit's 0.0340. The invariant holds.

It holds for a structural reason, visible per bucket:

| bucket | v16 n | absent | share | R1a n | absent | share | R1a session rows |
|---|---|---|---|---|---|---|---|
| 3 s | 820 | 36 | 0.0439 | 820 | 41 | 0.0500 | 0 |
| 6 s | 576 | 20 | 0.0347 | 576 | 18 | 0.0312 | 0 |
| **12 s** | 162 | 5 | 0.0309 | 162 | **0** | **0.0000** | 89 |
| **30 s** | 52 | 2 | 0.0385 | 52 | **0** | **0.0000** | 29 |

On the two buckets the block fires on, R1a's target-absent share is exactly
zero: a session row *replaces* a row that could have been target-absent, and a
session row never is. The 3 s and 6 s buckets move only as sampling noise
(0.0439 -> 0.0500 on n = 820 is 0.006 +- 0.019); those rows are bit-identical
to v16's row for row until a session row earlier in the same worker moves the
stream on (§6 pins the identity where the stream cannot have moved). So the
direction of the only real change is *fewer* target-absent rows, not more --
which is the safe side of the failure this invariant exists to prevent.

The seed is verified rather than assumed: a fresh seed-21 run of
`train_dpcrn_v16_lengthmix.yaml` reproduces the first 44 rows of
`absent_v16_lengthmix.jsonl` field for field (bucket, `spkid`,
`realized_speech_sir`, `noise_snr`, `foreground_distance`, `foreground_drr`,
`rt60`, `nearest_interferer_distance`, `target_absent`).

---

## 4. Session rows characterised

Second pass, `--only-bucket 30 --only-bucket-n-spk 2`: the natural mix gives
only ~47 session rows in 652, which is too few to read a shape share off. This
pass pins every batch to the 30 s bucket -- where a 5-20 s gap actually fits --
and yields **132 session rows of 260**. Numbers are that pass unless the column
says otherwise; the 652-row natural-mix numbers are in brackets where they
differ meaningfully.

| quantity | value | knob |
|---|---|---|
| session share of eligible rows | 0.508 [0.072 of all rows] | prob 0.5 |
| shape user_first | 0.288 | 0.25 |
| shape bystander_first | 0.235 | 0.30 |
| shape user_gap | 0.348 | 0.30 |
| shape overlap | 0.129 | 0.15 |
| turns per row | 9.0 [6.0,14.0] | max_turns 16 |
| user turns per row | 5.0 [3.0,7.0] | -- |
| **RIR move share** | 0.273 | 0.3 |
| ...of those, landed on 2 distinct channels | 0.917 (n=36) | -- |
| ...seat change, m | 0.15 [0.01,0.43] | -- |
| **distance-matched bystander share** | 0.167 | 0.2 |
| nearest interferer distance, matched rows, m | 0.56 [0.40,0.91] | user range 0.3-1.0 |
| nearest interferer distance, other rows, m | 2.44 [2.04,3.39] | bystander range 1.5-4.0 |
| user distance, m | 0.69 [0.47,0.92] | 0.3-1.0 |
| scripted user gap, s (rows with one) | 12.97 [5.90,17.29] | 5-20 |
| rows with a scripted gap >= 5 s | 0.348 | -- |
| drawn SIR, dB | -0.57 [-8.70,7.45] | -- |
| **P(drawn SIR <= -5 dB)** | **0.295** | 0.30 requested, >= 0.15 required |
| level ratio (U while it talks vs B while it talks), dB | -3.31 [-19.46,7.95] | -- |
| P(level ratio <= -5 dB) | 0.462 | -- |
| SIR over double-talk frames, dB | -2.80 [-17.83,8.63] | -- |
| P(double-talk SIR <= -5 dB) | 0.400 | -- |
| forced floor drawn, dBFS | -52.1 [-57.9,-46.6] | -60..-45 |
| **quiet-frame level in the mixture, dBFS** | -38.7 [-50.9,-29.4] | -- |
| ...minimum over quiet frames, dBFS | -45.3 [-57.4,-37.3] | -- |
| **rows whose quiet frames are digital silence** | **0.000** | the contract |
| `user_active` equals `vad_target` frame for frame | **1.000** | the contract |
| rows with a non-contiguous `turn_id` run | **0.000** | the contract |
| `turn_id` frame coverage | 0.71 [0.39,0.84] | -- |
| distinct chain ids per row | 1.00 | one chain draw per row |
| distinct user speaker ids per row | 1.00 | a move is not a new talker |

Two of these deserve a sentence.

**The move is a channel change, not much of a distance change.** 91.7 % of moves
land on a genuinely different near channel of the same room; the *distance*
between the two seats is only 0.15 m at the median, because the pre-generated
bank gives each room exactly two near channels and their spacing is what it is
(measured on 310 bank rooms: |d2 - d1| median 0.194 m, p90 0.424 m, max
0.553 m). So the row delivers "same voice, same chain, different transfer
function and a small distance change" -- which is the mechanism the design wants
(identity must survive it) -- and *not* "the user walked across the room". A
larger seat change needs a bank with more near channels per room, not a knob.

**"SIR" needs saying which frames.** The drawn SIR is applied by `add_bg_noise`
against the whole row's RMS, and on a turn-taking row neither talker occupies
the whole row. Over the frames the *user* is active the achieved SIR is high by
construction (the bystander is mostly quiet then); the two numbers that mean
something are the level ratio (each talker measured while it talks: -3.3 dB
median, 46 % at or below -5 dB) and the double-talk SIR (-2.8 dB median, 40 % at
or below -5 dB). The requested heavier low tail is delivered:
P(drawn SIR <= -5 dB) = 0.295 against 0.05 realised in v16.

### 4.1 Counterfactual identity material

The user and the bystanders are drawn from **one shared speaker pool** -- the
same list the recipe's foreground comes from -- so "user" is a property of the
role a row gave a voice, never of the voice. In the 132-session pass: 125
distinct speakers appeared as U, 193 as B, and **14** appeared as both. That is
what a shared pool of 1850 speakers predicts at this sample size (E[both]
= 1850 * (1-(1-1/1850)^132) * (1-(1-1/1850)^264) ~ 17), i.e. the roles are
independent of identity; over a 20-epoch run (~19 k session rows) every speaker
that appears at all appears in both roles.

The other half of the counterfactual is the distance-matched bystander (0.167 of
rows, landing at 0.56 m against the user's 0.69 m), which is what stops a
near/far scalar from satisfying an identity objective (review item 4).

### 4.2 Cross-chain pairs: implemented, and rare within a batch

`row_source_id` works as specified -- a paired row renders the material its slot
determines and hands the RNG streams back before the noise and the device chain,
so two rows with the same id are the same source through two independent chain
draws (`test_paired_rows_share_their_material_and_not_their_chain` asserts
exactly that: same script, same talkers, same SIR, different `turn_chain`).

**But at the R1a knobs no two such rows land in the same batch.** Measured:
0 within-batch pairs in 58 natural-mix batches and 0 in 130 thirty-second
batches, with 32 % of session rows paired. The arithmetic says why: a 12 s batch
holds 6 rows, of which ~3 are sessions and ~0.8 are paired, and two paired rows
must additionally draw the same slot out of 256; a 30 s batch holds 2 rows. This
is not a bug in the emission -- it is a property of the batch sizes the length
schedule fixes:

* Collisions need `pair_pool_size` of order the number of paired rows in a
  batch. `pair_prob: 1.0, pair_pool_size: 8` would give ~0.4 pairs per 12 s
  batch -- at the price of only 8 distinct paired sessions in the whole run
  (~120 repeats each per epoch). That trade is a knob, not a default.
* A guaranteed within-batch pair needs the sampler to put the same speaker in a
  batch twice (`trainer.n_utt_per_speaker: 2` with the bucket sizes halved),
  which is a `trainer.length_schedule` change and therefore out of this round
  (v15 confound), or a `pair_seed` slot on the sampler tuple, which needs
  `puresound/system/runner.py` -- outside this deliverable's write scope.
* **What R1a should do instead:** the design already says pairs may come from
  "other rows of the batch **/ an embedding queue**" (§4.2). `row_source_id` is
  stable across batches by construction (it encodes the length bucket, so the
  same slot in a 12 s and a 30 s batch is deliberately a *different* id), so a
  queue keyed on it collects pairs at ~32 % of session rows -- about 300 paired
  rows per epoch, ~150 pairs. The proximity loss's cross-chain term should read
  a queue, not a within-batch match, or it will be a graph-carrying zero.

---

## 5. Diversity and memory budget (review item 5)

Sessions are trained as **long rows inside the existing length schedule**. The
design's preferred form -- state-carrying truncated segments, so several 10 s
segments of one session share recurrent state within a row -- does not exist in
this trainer (no state is carried across rows or across truncated windows), and
building it is an architecture change, not a data change. `max_seconds` is added
as the knob for it; at 60.0 it is a no-op on today's longest bucket.

The consequence is that the budget item 5 asked to pre-register is **unchanged**,
because a session row *replaces* a long row rather than adding one:

| per training batch | v16 | R1a |
|---|---|---|
| rows (mean over 58 batches) | 11.24 | 11.24 |
| audio-seconds (mean) | 66.19 | 66.21 |
| distinct foreground speakers (mean) | 11.24 | 11.22 |
| talker draws, 1 + far_count (mean) | 24.79 | 25.31 |
| session rows (mean) | 0.00 | 0.81 |

(11.24 rather than the recipe's nominal 10.5 because the sample stopped after 59
batches; the schedule's expectation is 10.5. The 0.02-speaker difference is a
session row's `spkid` occasionally coinciding with another row's.)

### Peak memory, <= 30 steps, one L4 (21.95 GiB)

`memory_smoke.py`, `--gpu 0`, one device, no DDP, no validation, no
checkpointing, same module / seven losses / `accumulate_grad_batches: 2` /
`bf16-mixed` as the recipe, 30 batches (= 15 optimiser steps at accumulate 2).
The v16 header's reference numbers were measured under 2-GPU DDP, so read these
as the same measurement minus DDP's gradient buckets -- which for an 801 K
parameter model is ~3 MB, not a rounding difference that matters here.

| bucket | session rows | batches | peak allocated GiB | peak reserved GiB | batch time med / mean s | first batch s | v16 header |
|---|---|---|---|---|---|---|---|
| 30 s x 2 | on | 30 | **16.92** | 19.50 | 0.47 / 0.63 | 24.9 | 16.9 |
| 30 s x 2 | off | 30 | **16.92** | 19.50 | 0.50 / 1.39 | 25.8 | 16.9 |
| 12 s x 6 | on | 30 | **19.00** | 21.32 | 0.69 / 0.80 | 33.0 | 19.0 |
| 60 s x 1 | on | 30 | **16.26** | 20.58 | 0.50 / 1.47 | 25.2 | -- |

Peak *allocated* is the number to read: it is shape-driven and reproduces
exactly. The same four configurations measured on the other card of the same
box give 16.920 / 16.920 / 19.001 / 16.255 GiB -- identical to three decimals --
while peak *reserved* moved by up to 1.5 GiB between the two runs (allocator
fragmentation, not usage). The 12 s x 6 row reproduces the v16 header's 19.0
GiB, which is the check that this harness measures the same thing the header
did.

**Session rows cost no memory.** 30 s x 2 with the block on and off is the same
16.92 GiB to three decimals, because a session row replaces a long row of the
same shape: same samples in, same seven losses, same second forward for the
channel-consistency term. The block changes what is *in* the tensors, not how
many there are. Batch time is the same to within the dataloader's own noise
(median 0.47 vs 0.50 s; the means differ because the mean is dominated by
whichever run happened to stall on a worker, and the 30-batch sample is too
short to separate a session row's render cost from that).

**Is a 60 s bucket feasible?** **On memory, yes -- with more headroom than the
recipe's own peak bucket.** 60 s x 1 peaks at 16.26 GiB allocated / 20.58
reserved against the card's 21.95, *below* the 12 s x 6 bucket the shipped
recipe already runs (19.00 / 21.32): halving the rows per step more than pays
for doubling their length, because activation memory scales with total
audio-seconds per batch (60 s here against 72 s there) and not with row length.
A 30 s x 2 -> 60 s x 1 swap is therefore memory-neutral-to-cheaper.

Throughput is the real cost, and it is the reason R1a does not take the bucket:
per wall second the three configurations deliver 95 / 90 / 41 audio-seconds
(mean batch time against seconds per batch), so a 60 s x 1 bucket buys its
length at ~2.3x the wall time per audio-second -- a one-row batch cannot fill
the GPU, and the worker's per-row render cost stops overlapping. Adding the
bucket is also a `trainer.length_schedule` change, which is exactly the edit
this round refuses (v15's budget-shortfall confound). `max_seconds: 60.0` is
therefore left as a no-op knob with its cost now on the record, and R1a's
sessions stay long rows inside the existing schedule.

---

## 6. Default-off byte identity

`batch_identity.py`, the shipped v16 recipe, 20 batches / 234 rows, seed 7, 8
workers, comparing 24 tensor keys per batch (waveforms, both VAD targets, every
scalar the task and the device chain emit). Re-run on the final code, after
every edit in this deliverable:

```
reference: 20 batches, 234 rows, seed 7
  [determinism (same config twice)] 20 batches, 0 mismatching tensors
  [enabled, prob 0] 20 batches, 0 mismatching tensors
  [enabled, prob 1, min_seconds 60] 20 batches, 0 mismatching tensors
RESULT: IDENTICAL
```

The third variant is the strongest: the block is fully on (`prob: 1.0`,
`rir_move_prob: 1.0`, `pair_prob: 1.0`) and no row is eligible, so the
eligibility test is proved to precede every draw. `test/test_task` covers the
same property per row for an absent / disabled / prob-0 block over 24 seeded
items, and for a 6 s row under a fully-enabled block.

The variants are built by overriding the *v16* recipe, not by loading the R1a
one, so this check is unaffected by the heads and loss terms §11 switches on --
it is a statement about the data path only, which is the thing every existing
recipe depends on.

---

## 7. Tests

`uv run pytest test/test_task -q` -> **31 passed**;
`uv run pytest test/test_task test/test_losses test/test_system -q` ->
**171 passed** (the batch contract spans all three). What they pin, beyond the
byte identity above: `user_active` equals the row's own `vad_target`; `turn_id`
is one contiguous run per turn and 0 on double talk; every turn vector agrees
with the ids and carries exactly one chain id; the labels survive speed
perturbation (>= 95 % of user-active frames inside a user turn span, gate taper
allowed); a >= 5 s user gap keeps `target_present = 1` with a non-silent target
somewhere; the gap's quiet frames sit within 3 dB of the drawn floor and are
never digital zeros while the target there is silent; the device chain fires
exactly once per session row and `noisy - target` is the bystander bus plus the
floor to 15 %; the collate pads a mixed batch with the documented pad values and
is untouched when no row carries labels; a moved user keeps one speaker id and
two distinct seats; the script always gives both roles a turn, never lets two
bystanders talk at once, reaches all four shapes, and demotes rather than
truncates a shape the row cannot hold; the source scope restores all three RNG
streams; paired rows share material and differ in chain; and the pair id
separates the length buckets.

### 7.1 Four thresholds, measured rather than guessed

Four assertions had been written with slack instead of with a measurement --
the gate taper was the stated reason, and the taper is real, but the slack was
one to two orders of magnitude wider than the taper needs. Each was re-derived
from its own distribution and tightened; all four pass, and the first was
additionally checked for *power* by injecting the bug it exists to catch. Each
row's distribution comes from re-running that test's own dataset fixture over
16-24 seeds with the assertion replaced by a print, so the numbers below and
the numbers in the assertions' docstrings are the same numbers.

| assertion | was | measured | now |
|---|---|---|---|
| speed-perturbed labels: share of user-active frames inside a user turn (dilated 3, double talk allowed) | `>= 0.95` | 0.998 at the worst of 24 rows, 1.000 median. Same measurement with the speed factor dropped from the mapping: 0.774 worst, 0.968 median | `>= 0.99` |
| gap level against the drawn floor | `+-3.0 dB` | \|measured - drawn\| <= 0.037 dB over 16 rows, 0.002 median (without the 4-frame erosion: up to 1.31 dB, which is what the erosion is for) | `+-0.25 dB` |
| `noisy - clean - far_target` is the floor | `rel=0.15` | \|ratio - 1\| <= 0.0044 over 16 rows -- the residual is exact, the only slack is a finite Gaussian draw's sample RMS | `rel=0.02` |
| `turn_id` is 0 on double talk | `(ids == 0).sum() > 0` | of the energy VAD's both-active frames, 0.939 carry no id at the worst of 20 rows, 0.979 median | `>= 0.90`, **plus** a new exact test |

The last one deserves the detail, because loosening it was the one case where
the fade genuinely confuses the measurement and where "just tighten it" would
have been wrong. The contract is about the *script's* double talk; the row-level
test can only see the *energy VAD's* both-active frames, and those are a strict
superset (a raised-cosine turn edge and a reverb tail keep the leaving talker
audible for a few frames into the arriving talker's turn, and those frames do
legitimately carry the arriving turn's id). So the row-level assertion is now a
share with a measured floor, and the exact contract moved to a new test that can
see the script -- `test_turn_id_is_zero_on_exactly_the_scripted_double_talk`,
which over 1200 scripts x 3 row lengths asserts **zero** scripted double-talk
frames carry an id, every id occupies exactly one contiguous run, and every id
lies inside its own turn's span (that last one matters because
`pool_turn_means` indexes `turn_role` / `turn_speaker` *by* the id: an id
outside its span would pool one turn's frames under another turn's label).

Two allowances in that table are the contract and were left alone. Allowing
double-talk frames in the speed test is mandatory: `turn_id` is 0 there by
design, and without the allowance the same measurement reads 0.81 at the median
**on correct code** -- it would be asserting the contract's negation. The
3-frame dilation and the 4-frame erosion are the gate: `fade_samples` is 400
samples (2.5 frames at hop 160) and the labeler's analysis window is another
400, so audible energy can precede a turn's first labelled frame by about five
frames in total.

---

## 8. The batch contract the loss agent codes against

Per row, on the `vad_target` grid (frame 400 / hop 160 at 16 kHz, 100 fps);
collated with padding. Emitted on **every** row of a session-enabled recipe --
a key present on some rows only would collate into a tensor shorter than the
batch.

| key | dtype / shape | meaning | pad |
|---|---|---|---|
| `user_active` | float [B, T] | energy VAD of the dry user signal | 0.0 |
| `bystander_active` | float [B, T] | energy VAD of the summed dry bystanders | 0.0 |
| `turn_id` | long [B, T] | 1..K per contiguous single-talker turn; 0 = no turn or double talk | 0 |
| `turn_role` | long [B, K_max] | 1 user, 2 bystander | 0 |
| `turn_speaker` | long [B, K_max] | global speaker index (`dataset.spk2idx`) | -1 |
| `turn_chain` | long [B, K_max] | this row's chain-draw id, one value per row | 0 |
| `row_source_id` | long [B] | paired-source identity; -1 = unpaired | -- |

Plus the diagnostics `session_row`, `session_shape`, `session_n_turns`,
`session_n_user_turns`, `session_rir_move`, `session_move_distance_delta`,
`session_move_channels`, `session_matched_bystander`, `session_gap_seconds`,
`session_sir_db`, `session_floor_dbfs` (float scalars, one per row), and the
unchanged `foreground_distance` / `foreground_drr` /
`nearest_interferer_distance` / `n_interferers`.

Notes the losses depend on:

* A **non-session row** carries the true `user_active` / `bystander_active` for
  its own signals, `turn_id` all zero, and **zero-length** turn vectors. A batch
  with no session row therefore collates to `[B, 0]` turn tensors -- the honest
  shape for "no turns here", and the case a loss must read as a
  zero-contribution row.
* `turn_id == 0` already excludes the script's double talk. A frame where the
  *energy VAD* says both talk is not necessarily one of those (the gate taper
  and reverb tails spread a few frames), so excluding
  `user_active AND bystander_active` on top is right.
* A turn with no single-talker frame at all never appears in `turn_id`. That is
  by construction (§the longest-run rule) and such turns are dropped from the
  script before rendering, so audio and labels always agree.
* `turn_chain` is one value per row, drawn from [1, 2^31); 0 is only ever pad.
* The separation `target` stays U's own signal exactly as today. A long user gap
  is silent *in the target* because the user is not talking, and the row is
  still `target_present = 1`.

### 8.1 Checked against the two losses, not against the design document

The consumers are `puresound/nnet/loss/identity.py` and `proximity.py`. Their
module docstrings state the contract independently; every line of it was checked
against what the generator emits, and **no mismatch was found, so no data-side
change was needed**. What was checked, and how:

* **Keys.** The seven above and nothing else; `SESSION_LABEL_KEYS` is that list.
  Both losses also read the pre-existing `foreground_distance`,
  `foreground_drr`, `nearest_interferer_distance`, `n_interferers`,
  `consistency_noise`, `vad_target`, `background_vad_target`, all unchanged.
* **Dtypes and shapes**, read off a real collated batch: `user_active` and
  `bystander_active` `float32 [B, T]`; `turn_id` `int64 [B, T]`; `turn_role`,
  `turn_speaker`, `turn_chain` `int64 [B, K_max]`; `row_source_id` `int64 [B]`.
* **`K_max` shared.** The three per-turn tensors are padded from per-row vectors
  that `_turn_vectors` builds in one pass, so their lengths are equal row by row
  and `pad_sequence` gives all three the same `K_max`. This is load-bearing:
  `pool_turn_means` takes `n_turns` from `turn_speaker.shape[1]` in the identity
  loss and from `turn_role.shape[1]` in the proximity loss, and raises if
  `turn_id.max() > n_turns`. Measured on 250 session rows across two passes: 0
  rows where `turn_id`'s maximum exceeds the turn-vector width.
* **`turn_speaker >= 0` on every real turn.** The losses drop a turn with a
  negative speaker id (that is the pad). Measured: 0 of 250 session rows carry a
  negative id on a user or bystander turn -- the pool is `dataset.total_spks`,
  which is exactly the set `dataset.spk2idx` is built from.
* **The frame grid.** The labels are on `frame_count(L, 400, 160)`; the heads
  are on the encoder's STFT grid (`win_length: 512`, `hop_length: 160`).
  Measured by running the R1a backbone at every bucket:

  | row | samples | label T | head T (`identity_emb`, `proximity`) | delta |
  |---|---|---|---|---|
  | 3 s | 48000 | 298 | 297 | 1 |
  | 6 s | 96000 | 598 | 597 | 1 |
  | 12 s | 192000 | 1198 | 1197 | 1 |
  | 30 s | 480000 | 2998 | 2997 | 1 |
  | 60 s | 960000 | 5998 | 5997 | 1 |

  The offset is a constant 1 at every length, not a drift -- which is the actual
  proof that the two grids share hop *and* origin (frame `i` starts at sample
  `i * 160` on both sides; only the analysis window differs, 400 against 512).
  `align_frames` truncates to the common prefix, i.e. it drops the label grid's
  last frame and nothing else.
* **Non-zero reach, measured by calling the losses.** The identity loss needs
  two eligible turns *and* a speaker holding two of them -- and at exactly two
  turns of one speaker its numerator equals its denominator, so in practice it
  needs **>= 3 eligible turns with >= 1 non-matching speaker**. Over 80 real
  batches of this recipe (seed 21, natural length mix), with the two heads
  randomly initialised over a bottleneck of the verified grid:

  | quantity | share of batches |
  |---|---|
  | carries a session row | 0.3375 (27/80) |
  | >= 3 eligible turns | 0.3375 (27/80) |
  | some speaker holds >= 2 eligible turns | 0.3375 (27/80) |
  | **`IdentityContrastiveLoss` != 0** | **0.325 (26/80)** |
  | **`RelativeProximityLoss` != 0** | **0.325 (26/80)** |
  | **its cross-chain consistency term != 0** | **0.000 (0/80)** |

  Eligible turns per session batch run 3 to 20 (53 of 80 batches have zero --
  the 3 s and 6 s buckets -- and both losses are graph-carrying zeros there,
  which is what keeps DDP from seeing an unused head). The one session batch of
  27 that produced a zero is the informative case: its session row had no
  eligible bystander turn, so every eligible turn shared one speaker, and
  *both* terms are then structurally zero -- the identity loss because its
  positives are its whole candidate set, the proximity loss because its
  ordering term needs a user turn and a bystander turn in the same row. Both
  gradients w.r.t. the bottleneck are exactly 0 there and non-zero elsewhere
  (checked by `backward`).
* **The consistency term never fires** (§4.2, §10). That is a rate, not a
  contract mismatch: the key is emitted, typed and paired correctly, and two
  rows carrying the same `row_source_id` do differ in `turn_chain` by
  construction -- they just never land in one batch at `pair_prob: 0.25` /
  `pair_pool_size: 256` with 2-to-6-row batches.

**Deviation from the brief:** the recipe key is `augmentation_session_rows`, not
`session_rows`. Every pipeline block a dataset takes as `<name>_args` carries
that prefix, `BaseRecipe.augmentation_kwargs` forwards exactly those, and
`test/test_utils/test_config_schema.py` asserts the two sets match -- a
prefix-less block fails two of its contract tests, and that file is outside this
deliverable's write scope. The knob *inside* the block is `enabled`, as
pre-registered. No batch key changed.

---

## 9. Reproducing

```bash
cd egs/voice_isolate
S=/tmp/v20_session_rows

# temporal shape, both configs
uv run python benchmarks/probes/v20_session_rows/sample_rows_v20.py \
    --config config/exp/train_dpcrn_v16_lengthmix.yaml \
    --n-rows 652 --num-workers 8 --seed 7 --out $S/v16_rows.jsonl
uv run python benchmarks/probes/v20_session_rows/sample_rows_v20.py \
    --config config/exp/train_dpcrn_v20_r1a.yaml \
    --n-rows 652 --num-workers 8 --seed 7 --out $S/r1a_rows.jsonl
# the target-absent decision sample (§3.1): the same two configs, seed 21, 1600 rows
uv run python benchmarks/probes/v20_session_rows/sample_rows_v20.py \
    --config config/exp/train_dpcrn_v16_lengthmix.yaml \
    --n-rows 1600 --num-workers 8 --seed 21 --out $S/absent_v16_lengthmix.jsonl
uv run python benchmarks/probes/v20_session_rows/sample_rows_v20.py \
    --config config/exp/train_dpcrn_v20_r1a.yaml \
    --n-rows 1600 --num-workers 8 --seed 21 --out $S/absent_v20_r1a.jsonl
# session rows at their natural length
uv run python benchmarks/probes/v20_session_rows/sample_rows_v20.py \
    --config config/exp/train_dpcrn_v20_r1a.yaml --only-bucket 30 \
    --only-bucket-n-spk 2 --n-rows 260 --num-workers 8 --seed 7 \
    --out $S/r1a_session30_rows.jsonl
uv run python benchmarks/probes/v20_session_rows/aggregate_v20.py \
    --label v16 --label R1a --label "R1a 30s-only" \
    $S/v16_rows.jsonl $S/r1a_rows.jsonl $S/r1a_session30_rows.jsonl

# byte identity and memory
uv run python benchmarks/probes/v20_session_rows/batch_identity.py \
    --batches 20 --num-workers 8 --seed 7
for spec in "30 2" "12 6" "60 1"; do set -- $spec
  uv run python benchmarks/probes/v20_session_rows/memory_smoke.py \
      --seconds $1 --n-spk $2 --steps 30 --gpu 0
done
uv run python benchmarks/probes/v20_session_rows/memory_smoke.py \
    --seconds 30 --n-spk 2 --steps 30 --gpu 0 --disable-session-rows

# how often each new loss term is actually non-zero (§8.1)
uv run python benchmarks/probes/v20_session_rows/loss_reach.py \
    --batches 80 --num-workers 8 --seed 21

# the checkpoint pre-flight (§11)
uv run python scripts/preflight_ckpt_recipe.py \
    --ckpt /work/any_exp_link/puresound_exp/dpcrn_v16_lengthmix/lightning_logs/\
version_0/checkpoints/epoch=19-step=10000.ckpt \
    config/exp/train_dpcrn_v20_r1a.yaml config/exp/train_dpcrn_v16_lengthmix.yaml
```

Training (warm start from v16 ep19; never from scratch -- the eight-stage ladder
is load-bearing):

```bash
cd egs/voice_isolate
uv run python main.py config/exp/train_dpcrn_v20_r1a.yaml --training \
    --pretrained_ckpt_path /work/any_exp_link/puresound_exp/dpcrn_v16_lengthmix/\
lightning_logs/version_0/checkpoints/epoch=19-step=10000.ckpt
```

---

## 10. What is NOT measured here, and what would break

* **No model has seen these rows.** Every number above is a property of the
  data, or of a loss called on the data with an untrained head. Whether a
  session row teaches anything is R1a's training question, and the design's own
  warning applies: v10's T2 answered "more bystander-first rows" by suppressing
  more (turn-taking keep violations 6 -> 10, moderate WER +0.024). Session rows
  *without* the identity/proximity objectives would be that experiment again,
  which is why the recipe enables both (§11) rather than shipping the data axis
  alone -- and why the data-axis-alone variant is documented as an ablation to
  run deliberately, not as the default.
* **The within-batch cross-chain pair rate is zero at these knobs** (§4.2). The
  proximity consistency term must read a queue keyed on `row_source_id`.
* **The user's seat change is 0.15 m**, not a walk across the room (§4).
* **A paired twin does not share its speed perturbation.** The source scope ends
  before the noise stage, and the speed draw sits after it, so twins differ in
  tempo as well as in chain and noise. Turn-pooled scalars tolerate that; a
  frame-aligned comparison between twins would not.
* **`echo_playback` is off in this recipe.** If it were enabled it would add an
  unlabelled talker to a session row -- the labels describe U and the scripted
  bystanders only.
* **The Silero VAD backend is untested here.** `user_active` is always the
  energy labeler on the same grid; with `vad_label.backend: silero` the row's
  `vad_target` would be Silero's and the two would no longer be equal.
* **`augmentation_row_initial_ambient` is refused** together with this block
  (the recipe validator): it masks speech out of the row's opening after the
  script is written. `augmentation_speech.is_target: True` is refused for the
  same class of reason.

---

## 11. The recipe now switches R1a's other half on -- and its pre-flight

`config/exp/train_dpcrn_v20_r1a.yaml` no longer carries commented placeholders.
The heads and the two loss terms are enabled, with the argument names the loss
modules actually take (the placeholders had guessed `exclude_overlap`,
`teacher_momentum` and `cross_chain_weight`; the real signatures are
`temperature` / `momentum` / `min_turn_frames` and `margin` /
`consistency_weight` / `min_turn_frames`, and `extra="forbid"` on the head
configs would have rejected the guessed key names too):

```yaml
loss_func:
  # ... v16's seven terms, unchanged ...
  - type: IdentityContrastiveLoss
    weighted: 0.1
    args: {temperature: 0.1, momentum: 0.99, min_turn_frames: 1}
  - type: RelativeProximityLoss
    weighted: 0.1
    args: {margin: 1.0, consistency_weight: 1.0, min_turn_frames: 1}

model.backbone.backbone_args:
  identity_head:  {enabled: True, dim: 64, kernel_t: 5}
  proximity_head: {enabled: True, hidden: 64}
  expose_bottleneck: True
```

`scripts/preflight_ckpt_recipe.py` against the warm-start checkpoint
(`dpcrn_v16_lengthmix/.../epoch=19-step=10000.ckpt`), verbatim (exit 1):

```
  train_dpcrn_v20_r1a.yaml                   MISSING 14: ['backbone.identity_head.out.bias', 'backbone.identity_head.out.weight', 'backbone.identity_head.trunk.dwconv.bias']
  train_dpcrn_v16_lengthmix.yaml             ok

preflight FAILED for 1 recipe(s): the benchmark would score a model that is not the checkpoint.
```

The 14 are **exactly and only the two new heads**, which is the expected result
of adding a head to a warm-start recipe -- not a defect in the file. The script
lists three; all 14, with shapes:

| key | shape |
|---|---|
| `backbone.identity_head.trunk.dwconv.weight` | (128, 1, 5) |
| `backbone.identity_head.trunk.dwconv.bias` | (128,) |
| `backbone.identity_head.trunk.norm.weight` | (128,) |
| `backbone.identity_head.trunk.norm.bias` | (128,) |
| `backbone.identity_head.out.weight` | (64, 128) |
| `backbone.identity_head.out.bias` | (64,) |
| `backbone.proximity_head.trunk.dwconv.weight` | (128, 1, 5) |
| `backbone.proximity_head.trunk.dwconv.bias` | (128,) |
| `backbone.proximity_head.trunk.norm.weight` | (128,) |
| `backbone.proximity_head.trunk.norm.bias` | (128,) |
| `backbone.proximity_head.net.0.weight` | (64, 128) |
| `backbone.proximity_head.net.0.bias` | (64,) |
| `backbone.proximity_head.net.2.weight` | (1, 64) |
| `backbone.proximity_head.net.2.bias` | (1,) |

The other direction is the one that would be a real defect, and it is clean:
**0 dropped trained backbone weights** -- no weight v16 trained goes unused, so
the warm start really is v16's model plus two fresh heads. `main.py
--pretrained_ckpt_path` loads non-strictly, so those 14 initialise from their
constructors, which is the intent. The pre-flight's non-zero exit is therefore
expected for this recipe and must be read alongside the MISSING list: an exit 1
whose list contains anything *other* than `identity_head.*` / `proximity_head.*`
is the failure the script exists to catch.

There are also **4 unexpected keys** in the checkpoint --
`loss_func_list.1.stft_losses.{0,1,2}.window` and
`loss_func_list.2.hann_window`. They are pre-existing and unrelated: the v16
recipe itself reports the same four (the pre-flight ignores loss-function
buffers by design, which is why it prints `ok` for v16), and they are FFT
windows a loss reconstructs at build time.

**To run the data axis alone**, comment the two `loss_func` entries and the
three `backbone_args` lines *together* -- a loss without its head raises at the
first step, by design (`IdentityContrastiveLoss` names the missing
`backbone_args.identity_head` in its error). That variant reports 0 missing / 0
unexpected, i.e. it is v16's model exactly.
