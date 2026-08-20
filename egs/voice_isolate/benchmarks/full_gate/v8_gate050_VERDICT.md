> ⚠️ **Measured on field set v2 (27 clips, 90D/270D only).** The field benchmark was
> rebuilt as v3 on 2026-08-20 (64 clips, four orientations + seven QVF clips, new
> hand labelling) — see `benchmarks/field_test_vector/SET_V3.md`. **Numbers below do
> not compare against v3 runs.** Synthetic and WER stages are unaffected.

# v8 + presence gate at b_hi 0.50 -- NOT a deployment candidate

Run 2026-08-19, chain `3cde9d1`. `v8_baseline.txt` and `v8_gate050.txt` are the
same checkpoint, same day, same code, same chain, on two GPUs in parallel -- the
only difference is `--presence-gate benchmarks/probes/presence_readout_v8.npz`.
Both at `dry_blend 0.9`.

`b_hi 0.50` was chosen because `b_traj_sweep.py` found it violation-free on the
27-clip field set. It is. The full gate says that measurement had no resolution.

## Verdict: fails 4 of 9 stages, one of them catastrophically

| stage | baseline | + gate | |
|---|---|---|---|
| 1 field, keep clips | 0/15 violations, worst -2.26 | **0/15, worst -2.26** | identical |
| 1 field, cold-start far | 9 FAIL / 1 PARTIAL | 8 FAIL / 2 PARTIAL | barely moved |
| 2-5 synthetic F-only | -31.1 / -29.4 / -32.3 / -26.7 | **-33.9 / -33.2 / -36.0 / -31.8** | +2.8 to +5.1 |
| 6 Dawn WER | 0.180 (**beats** raw 0.184) | 0.186 (**loses** to raw) | **FAIL** |
| 7a moderate WER (PRIMARY) | 0.406, delta -0.176 | 0.425, delta -0.157 | **FAIL** |
| 7b BUT-OFFICE | 0.539, -0.024 vs mix | 0.584, **+0.021** vs mix | **FAIL** |
| 8 BUT extreme reverb | 0.677, +0.019 vs mix | 0.707, **+0.049** | **FAIL** |
| 9 turn-taking KEEP | -0.13 med, **94 ok / 6 viol** | **-4.49 med, 35 ok / 65 viol** | **CATASTROPHIC** |
| 9 turn-taking SUPPRESS | -16.14, 87 ok / 13 fail | -25.66, 98 ok / 2 fail | +9.5 dB |
| 9 SI-SDR | +5.42 | **-0.23** | **-5.65 dB** |

Stage 9 is the whole story in one row: far-solo suppression improves by 9.5 dB and
**65 of 100 near spans lose the user**, taking SI-SDR from +5.42 to below zero.

## The mechanism: the readout is confounded with reverberation

Deletion rate, the four WER sets ordered by how reverberant they are:

| set | reverb | del base | del gate | delta |
|---|---|---|---|---|
| Dawn Chorus | real noisy, mild | 0.094 | 0.109 | +0.015 |
| moderate | RT60 0.20-0.65 | 0.187 | 0.213 | +0.026 |
| BUT-OFFICE | RT30 0.56-0.69 | 0.234 | 0.382 | **+0.148** |
| BUT reverb | RT30 1.15-1.84 | 0.278 | 0.493 | **+0.215** |

**Monotone in reverberation, and it explodes past RT ~0.5 s.** The readout was
fitted on one room's field recordings at one RT60. Reverberation moves the
bottleneck in the direction it reads as "absent" -- which is physically exactly
what you would expect, because **reverberation is the far-field cue**. A
reverberant near talker and a distant talker look the same to this readout, so it
gates out the user.

That is why the field set showed zero cost and every other real set showed
deletion: the field set is the room the readout was fitted in.

## What survives

* **The detector's separation is real** -- held-out balanced accuracy 0.958 on the
  270D recording, and the synthetic F-only probes all improve 2.8-5.1 dB. The
  readout is not noise.
* **The actuator works.** Suppression moves where the blend could not: turn-taking
  far-solo +9.5 dB, synthetic probes +3-5 dB. `b_traj_README.md`'s conclusion that
  a gain can move what `dry_blend` cannot is confirmed at scale.
* **The dead zone does what it claims.** Every field keep clip is bit-identical.

## What this kills

`b_hi 0.50` with a readout fitted on **one room**. It does not kill the gate
architecture, and it does not kill the presence quantity. It kills this readout,
and it identifies the axis that has to be fixed first: **the presence estimate has
to be invariant to reverberation before its output can drive a gain.**

## The methodological lesson, which is the more valuable half

The field set said zero violations. The turn-taking set said 65. Both are "real
recordings" scorecards; they disagree by an order of magnitude because the field
set is 27 clips from a single room and cannot see a reverberation-conditional
failure at all.

**A gate that passes the field scorecard has not been tested.** Any future
operating point has to clear stage 9 and the WER sets before it is worth
discussing, and the b-trajectory sweep should never again be read as an operating
curve on its own.

## Measurement gap in this run

`eval_wer.py` names its per-item transcript dump from the checkpoint stem only, so
both runs wrote the same file and the second overwrote the first. Aggregate WERs
are unaffected (they are computed in-process and logged) but **no paired test
between the two systems is possible from these logs.** Given the effect sizes on
7b/8/9 it does not change the verdict; it would have mattered had the only signal
been 7a's +0.019. The tag needs the gate in it.
