# Cold-start protocol v2 — the resolving instrument, and the standings under it

2026-09-01. Tool: `scripts/eval_coldstart_v2.py`. Why v1 readings were unreadable and where
the 3 s room-tone effect comes from: `benchmarks/probes/v17_round_design.md` §2.

## Protocol

A cold-start reading is NEVER one checkpoint on one zero-context cut. It is:

* a **block of checkpoints** — the run's last five, fixed before looking at any field
  score (picking the epoch by this metric is checkpoint selection; the same-run null band
  spans ~3 dB);
* × **context conditions** — `none` (worst-case stream start) and `ambient` (3 s of the
  recording's own room tone, 3 seeded draws, from outside every annotated span; this is
  the deployment steady state — a live stream has always heard the room);
* × **both sides** — lone-far suppress AND lone-near keep, absolute residual / sir_out,
  whole-span and steady-state (t>=3 s) views;
* winners are called ONLY by `compare` mode: per-(clip, condition) block means, paired
  Wilcoxon signed-rank.

Known limitation: ambient pads exist for 7/20 far and 8/27 keep clips (the session files'
un-annotated gaps are scarce); comparisons pair on the covered intersection.

## Standings (2026-09-01, dry_blend 1.0)

| | v8 block (ep15-19) | v16 ep19 block | v16 ep39 block |
|---|---|---|---|
| far `none` block-median | -1.06 | -1.71 | -1.52 |
| far `none` <= -6 dB | 4/20 | 4/20 | 6/20 |
| far `ambient` block-median | -12.57 | -13.19 | -14.47 |
| far `ambient` <= -6 dB | 7/7 | 7/7 | 6/7 |
| keep violations / worst | 4 / -7.97 | 3 / -8.34 | 4 / **-15.68** |

Paired verdicts:
* **v16 ep19 vs v8, far none: delta median -0.03 dB, p=0.26 — NO DIFFERENCE.** The
  single-point reading "-1.21 vs -0.33" that credited v16 with a cold-start gain was
  noise. Zero-context cold start has never been moved by any training round, v1..v16.
* v16 ep39 vs ep19: all conditions p>0.6 — the second cycle bought nothing here.
* Ambient context is worth **+11..13 dB to every model family member** — an order of
  magnitude more than any training intervention on record for this axis.

## What consumes this

* Any round claiming a cold-start effect reports BOTH columns of this instrument and the
  paired p-values, against the standing blocks above.
* The v17 Part-B target, stated in these terms: move the `none` column toward the
  `ambient` column (calibrate faster from less), guards unchanged elsewhere.
* Deployment note: the `ambient` column is the product's steady state — keep streaming
  state alive across silence, prime new streams with buffered room audio.

## ERRATUM (2026-09-03) — the `ambient` column is speech, not room tone

The `ambient` pads are drawn from the only ≥ 3 s un-annotated gap in set v3 (90D 98.4–107.0 s),
and that gap contains an unlabelled utterance at 101.4–103.9 s (−47 dBFS). True floor from the
same recording (or 180D's −79 dBFS gap) moves cold-far suppression by ≤ 0.1 dB on both v8 and v16
blocks; the +11..13 dB was the utterance. Read every `ambient` number here as "3 s pad that
sometimes contained a talker". The corrected instrument and the anchor-length / decay curves are
in `../probes/reference_matrix_README.md` (anchor needs ≥ 1 s of speech, is gone after 10 s of
floor, and costs keep violations on the next talker). The deployment note "prime new streams with
buffered room audio" is withdrawn with it.
