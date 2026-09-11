# `dpcrn_curriculum_v1` — conversational rows, a second capture view, and two per-frame heads

2026-09-11. Recipe `config/exp/train_dpcrn_curriculum_v1.yaml`, warm-started from
`dpcrn_curriculum_v0.ckpt`, 40 epochs, 2×L4, ~34 h across two interruptions. Shipped
checkpoint: **ep39**. Records: `../full_gate/` (`v1_ep19`, `v1_ep39` nine-gate logs),
`../field_test_vector/records/set_v3/block_curriculum_v1_*`, `curriculum_v1_probes.md`.

## What was asked

The previous step reached about 70% of the released lineage and its remaining distance
was concentrated in one place: far-field suppression on real device recordings, which
neither more budget nor a different room mixture nor a different row length would move.
The pattern across those three negative forks was consistent — every change that made
the model gentler made suppression shallower, and gentleness was not what was missing.

This step adds what the schedule had never carried: rows that are **conversations**
rather than mixtures (a user and bystanders taking turns, with gaps, re-entry and a
seat change), a **second capture view** of the same row for a consistency term, and two
**per-frame heads** — relative proximity, supervised by each rendered turn's distance,
and presence. Each arrives on its own ramp; the file's constants are the end values, so
validation measures the final distribution rather than the starting one.

## Result: the far-field wall moved, on the chain that matters

Field numbers are 5-checkpoint blocks (`eval_field_block.py`), paired per clip. The
device chain is the internal field recordings; the QVF clips are the cross-chain
reference material and are reported separately, because this round separates them.

### The internal device chain

| block metric | **v1 ep39** | v1 ep19 | v0 ep99 | released ladder |
|---|---|---|---|---|
| cold-start far, no context | **−7.07** | −5.02 | −0.30 | −0.96 |
| cold-start far, room-tone context | **−28.27** | −27.71 | −3.58 | −14.51 |
| keep, near clips (n=24, median) | −0.16 | −0.14 | −0.16 | −0.19 |
| keep, double-talk (n=8, median) | −1.34 | −1.27 | −1.26 | −1.52 |
| keep violations | 1 | 1 | 1 | 1 |

Cold-start far-field suppression is the one this project has chased longest: a lone far
voice with no near anchor, which every earlier version passed through almost untouched.
It moves from −0.96 dB to −7.07 dB without context and from −14.51 dB to −28.27 dB with
room tone (p = 0.000 and p = 0.031 against the ladder, paired). **Near-field keep on the
same chain is unchanged** — 24 clips, median −0.16 dB against the ladder's −0.19, and
the single violation is the sentinel recording whose noise floor sits 25 dB high and
which every version violates.

### The rest of the nine gates

| gate | **v1 ep39** | v1 ep19 | v0 ep99 | released ladder |
|---|---|---|---|---|
| in-domain SI-SDRi median | +7.32 | +7.22 | +6.87 | +8.02 ⚠ |
| Dawn WER (unprocessed 0.184) | **0.172** | 0.180 | 0.178 | 0.180 |
| Dawn deletion | 0.088 | 0.089 | 0.089 | 0.094 |
| moderate-reverb WER delta | −0.156 | −0.149 | −0.108 | −0.171 |
| turn-taking SUPPRESS | −17.10 | −17.51 | −15.12 | −16.14 |
| turn-taking ok/fail | **94/6** | 94/6 | 84/16 | 87/13 |
| extreme-reverb monitor | +0.012 | +0.026 | — | — |
| synthetic far-only (expand / high / boundary) | −34.5 / −35.0 / −38.0 | −40.4 / −37.6 / −43.5 | — | — |

⚠ in-domain crosses the synthesis-chain boundary against the ladder and is comparable
only within this lineage.

Turn-taking suppression passes the released ladder on both depth and ok-rate. The
primary WER gate reaches 91% of the ladder's reduction, up from 63% one step ago, and
Dawn is the best of any version measured.

## The cost, and what it is

Keep on the **cross-chain reference clips** regresses. Worst case, `keep_in_touch_near1`
goes to −32.8 dB — the user, attenuated to inaudibility. Three probes say what that is:

1. **Harmonic peak-to-valley** (`audit_span_metric.py` method, 64 ms window): the
   harmonic contrast of that clip is *unchanged* under the model (27.1 → 27.7 dB) while
   its level drops 19.7 dB. The voice is not shredded; it is turned down. The model
   decided that talker is far and attenuated uniformly.
2. **The proximity head reads it as far.** Per-frame mean readout, where higher is
   nearer: healthy near clips +1.35 / +1.41, true far clips −3.00 / −2.46, and the two
   collapsed keep clips **−0.59 / −1.16** — between the two classes, on the far side.
   The keep damage is monotone in this readout across every clip measured.
3. **The keep-side noise guard does not tilt.** Adding noise to every keep clip down to
   8 dB speech-to-floor gives medians −0.57 / −0.60 / −0.75 / −0.84 / −1.26 against the
   ladder's −0.59 / −0.65 / −0.55 / −0.70 / −1.29; every level pairs at p > 0.05. The
   aggressiveness is not a general fragility, and the worst clip is one the ladder also
   collapses (−27.2 dB at 8 dB speech-to-floor).

So the regression is a **per-recording calibration failure on a chain outside the
deployment one**, not damaged speech and not a broader loss of keep. That matches what
this repository already knows about cross-chain behaviour: each recording sits at its
own offset, and absolute thresholds die across chains while self-calibrated ones live.

## ep19 versus ep39

They are a trade, not a ranking. ep19 keeps more: the worst cross-chain keep clip is
−12.6 dB instead of −32.8, and the synthetic far-only probes are 2.6–5.9 dB deeper.
ep39 recognises more: Dawn 0.172 against 0.180, the primary WER gate −0.156 against
−0.149, the extreme-reverb monitor halved, and cold-start far on the device chain 2 dB
deeper. Both blocks leave device-chain keep untouched. **ep39 is the shipped
checkpoint**; ep19 stays in the run directory as the gentler operating point, and the
two blocks are archived side by side so the trade can be re-read.

## What this direction achieved

* **A wall moved.** Cold-start far-field suppression — no near anchor, the bot-idle
  case — went from "passes the far voice through" to −7 dB without context and −28 dB
  with it, on the internal recordings, with keep untouched there. Three previous rounds
  aimed at this axis and none of them moved it.
* **Aggressiveness without deletion, on the trained chain.** The previous step's verdict
  said the missing quantity was aggressiveness and that every gentler variant traded it
  away. This step bought it: turn-taking suppression now passes the released ladder at a
  *better* ok-rate, and the near-field keep median improved at the same time.
* **The proximity head is real and readable.** Supervised only by relative ordering of
  rendered turn distances, it reaches 98% pairwise ordering accuracy in training and its
  readout separates near from far on unseen real recordings — and, where the model fails,
  it says so. It is a per-frame exportable state, so it is available to a runtime guard.
* **The curriculum mechanism carried three new axes at once** without a staged chain of
  runs, and the recipe is one file.

## What it did not achieve

* The cross-chain keep failure is now **worse, not better** — and the same head that
  explains it is the natural place to fix it. An absolute threshold on the readout will
  not survive the chain change (the margin between a mis-read near clip and a true far
  clip is 1.3–1.9 units); a self-calibrated, per-recording readout is what the evidence
  supports.
* The primary WER gate is still short of the released ladder (−0.156 against −0.171).
* Whether the gain comes from the session rows, the paired view, or the proximity head
  is unattributed. All three arrived together, and separating them costs one fork each.

## Reproducing

```bash
cd egs/voice_isolate
uv run python main.py config/exp/train_dpcrn_curriculum_v1.yaml --training \
    --pretrained_ckpt_path pretrained_ckpt/dpcrn_curriculum_v0.ckpt
```

Resume an interruption with `--ckpt_path` on this run's own latest checkpoint, never
with a fresh warm start: the curriculum reads the epoch from the trainer.
