# Field benchmark — block protocol (supersedes single-checkpoint readings)

2026-09-02. Tools: `scripts/eval_field_block.py` (whole set v3), `scripts/eval_coldstart_v2.py`
(cold-start subset, the pilot), `scripts/eval_s2f_keep_probe.py` (keep-side guard).
`run_full_benchmark.sh` runs stage 1b automatically when `BLOCK_CKPTS` is set.

## Why

Every field number this recipe compared before 2026-09-01 was one checkpoint on one cut.
Measured on v16's own run (ep8–19, no knob change): cold-far median −0.35..−3.39 dB, session
mean sd 1.95 dB, KEEP violation count 2..5, single clips swinging up to 19.9 dB between
adjacent epochs. Most recorded single-point deltas were inside that band.

## Protocol

* **Block** = the run's last five checkpoints, fixed before any field score is read.
  Choosing the epoch by field score is checkpoint selection and is forbidden.
* **Every clip** in set v3: sessions, lone-near, lone-far, double-talk, both chains.
* **Context conditions** on cold-start clips: `none` (zero context = stream-start worst case)
  and `ambient` (3 s of the recording's own room tone, 3 seeded draws = deployment steady
  state). Sessions carry their own context and get `none` only.
* **Metrics**: keep preservation, suppress reduction, absolute residual over floor, sir_out;
  reported as block mean per clip with the per-checkpoint spread (±) beside every median.
* **Verdicts** (ok / FAIL / violation) are computed on block means with the standard bars
  (KEEP < −3 dB, FAIL > −6 dB, PARTIAL residual > 12 dB).
* **Comparisons** only via `compare`: per-clip paired Wilcoxon per group, plus a severity
  table of every keep clip that moved ≥ 1 dB, tagged by chain — a count guard cannot see
  an already-violating clip deepening, nor chain-selective damage.
* **Keep guard**: the s2f sweep (native/20/15/12/8 dB added noise). Baseline curve, v16 ep19:
  −0.25 → −0.31 → −0.38 → −0.55 → −1.02 (median), cliff on qvf_keep_in_touch_near1 at s2f 8.
  Judge against the curve. The level-swept variant is retired: InstantLN makes the network
  scale-equivariant, so it is flat by construction and has no power.

## Standings

(filled in below from the 2026-09-02 block run)

### 2026-09-02 — four blocks, dry_blend 1.0 (± = per-checkpoint sd, median over clips)

| | v8 block (ep15–19) | v16 ep19 block | v16 ep39 block | v17 block |
|---|---|---|---|---|
| sessions supp median | **−11.78** ±1.30 | −9.73 ±1.41 | −11.64 ±1.73 | −9.26 ±1.21 |
| cold-far device `none` | −0.96 ±0.64 | **−1.61** ±0.63 | −1.26 | −1.31 |
| cold-far device `ambient` | **−14.51** ±1.57 | −12.65 ±2.50 | −14.43 | −10.03 |
| cold-far qvf `none` | −2.58 | −2.44 | **−4.86** | −2.57 |
| keep near violations / worst | 4 / −7.97 | **3 / −8.34** | 4 / −15.68 | 4 / −11.61 |
| keep double-talk median / viol | −1.53 / 1 | **−1.11 / 1** | −1.65 / 3 | −1.86 / 3 |
| sessions keep median | −0.73 | −0.77 | −1.04 | −0.85 |

Paired verdicts (Wilcoxon on block means):
* **v16 ep19 vs v8**: sessions **v8 deeper** (delta +2.18 dB, p=0.062, n=6); ambient cold-far
  v8 deeper (+2.91, p=0.062); cold-far `none` v16 better by 0.34 dB (p=0.049); double-talk
  keep v16 better by 0.27 dB (p=0.007). **The single-checkpoint reading "v16 ep19 beats v8
  by 1.1 dB on sessions" does not survive** — the released v8 checkpoint (ep19) happens to
  be its run's weakest of the last five on sessions (−9.83 vs block −11.78). On the field
  set v16 ep19 is a lateral move: slightly safer keep, slightly shallower suppression.
* **v16 ep39 vs ep19**: sessions deeper (−0.90, p=0.062) bought with keep damage
  (keep_in_touch near −6.7→−15.7, dt −8.9→−15.8; sessions keep p=0.031).
* **v17 vs v16 ep19**: cold-far `none` **worse** (+0.35, p=0.020) and five QVF keep clips
  degrade 1.7–6.9 dB. The v17 verdict hardens from "null" to "negative".

Consequence: the deployment-default question (v16 ep19 vs v8) is NOT settled by the field
set. It now hinges on the block-level Dawn deletion and the reverb-WER gates, neither of
which has been run as a block.

### Keep-side s2f guard (block of 5, all 38 keep clips) — no version tilts the curve

| added-noise s2f | v8 block | v16 ep19 block | v17 block |
|---|---|---|---|
| native | −0.59 | −0.47 | −0.44 |
| 20 dB | −0.65 | −0.60 | −0.60 |
| 15 dB | −0.55 | −0.56 | −0.61 |
| 12 dB | −0.70 | −0.65 | −0.73 |
| 8 dB | −1.29 | −1.11 | −1.22 |

Paired: every level p > 0.1 for both v16-vs-v8 and v17-vs-v16 — the s2f keep tilt is a
property of the model family, not of any round. The cliff clip is always
`qvf_keep_in_touch_near1` (v8 −27.2 / v16 −19.3 / v17 −29.9 dB at s2f 8).

### Window sweep, block means (session suppression dB)

| clip | block | full | 30 s | 10 s | 5 s | 3 s |
|---|---|---|---|---|---|---|
| 90d | v8 | **−13.32** | −6.92 | −4.74 | −2.89 | −0.84 |
| | v16 ep19 | −11.10 | **−7.17** | **−5.16** | **−3.57** | **−1.21** |
| 180d | v8 | **−12.59** | **−5.55** | −3.12 | −1.97 | −1.69 |
| | v16 ep19 | −10.45 | −4.94 | **−3.60** | **−2.06** | **−1.78** |
| 270d | v8 | **−13.28** | −4.51 | −4.20 | −2.40 | −2.19 |
| | v16 ep19 | −10.35 | **−5.15** | **−4.93** | **−3.18** | **−2.50** |

The single-checkpoint "v16 wins 15/15" was wrong at the long end: on blocks v8 is 2–3 dB
deeper at full length on every session, while v16 is 0.2–0.8 dB deeper at ≤10 s. **v16's
length-mix training did what it was designed to do — flatten the context curve — and paid
for it in ceiling.** Neither dominates; the choice is a deployment question (how much
context the product has when it must decide).

### 2026-09-03 — v18 (loss-floor flags) block

| | v18 block |
|---|---|
| sessions supp median | −8.62 ±0.56 |
| cold-far device `none` / `ambient` | −0.84 / −10.76 |
| keep near violations / worst | 4 / −12.11 |
| keep dt median | −1.79 |

vs v16 ep19: cold-far `none` worse +0.53 dB (p=0.017); sessions ns; keep/s2f unchanged.
Verdict and mechanism: `benchmarks/probes/v18_VERDICT.md`.
