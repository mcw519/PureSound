# `dpcrn_curriculum_v0` — the eight-rung lineage as one scheduled run

2026-09-09. Recipe `config/exp/train_dpcrn_curriculum.yaml`, cold start, 120 epochs,
2×L4, ~42 h + 21 h. Records: `../full_gate/curriculum_v0_ep99.txt`,
`../field_test_vector/records/set_v3/block_curriculum_v0_*`.

## What was asked

The released model's behaviour was reached through eight warm-started runs, 220 epochs
in total, and the last rung's recipe alone from scratch reaches almost none of it. Two
things could explain that: the ORDER the distributions arrived in, or simply the
budget. They had never been separated, and the recipes had never been written down as
one artefact.

`curriculum` (`puresound/config/curriculum.py`) makes the second question askable: the
knobs the rungs changed between runs become schedules inside one run. This is the first
recipe written that way — ten tracks, four phases, everything the ladder moved.

## Result: about 70% of the lineage, from one run

Judged at cosine troughs, `dry_blend 0.9`; field numbers are 5-checkpoint blocks
(`eval_field_block.py`), which is the only protocol that resolves them.

| gate | ep19 | ep39 | ep59 | ep79 | **ep99** | ep119 | released ladder |
|---|---|---|---|---|---|---|---|
| in-domain SI-SDRi median | +4.16 | +5.83 | +6.20 | +6.46 | **+6.87** | +7.08 | +8.02 ⚠ |
| moderate-reverb WER delta | −0.018 | −0.065 | −0.084 | −0.090 | **−0.108** | −0.122 | −0.171 |
| turn-taking SUPPRESS | −4.63 | −8.23 | −11.74 | −13.16 | **−15.12** | −14.39 | −16.14 |
| turn-taking ok/fail | 34/66 | 69/31 | 80/20 | 80/20 | **84/16** | 85/15 | 87/13 |
| turn-taking KEEP ok/viol | 94/6 | 94/6 | 91/9 | 90/10 | **92/8** | 93/7 | 94/6 |
| Dawn WER (raw 0.184) | 0.183 | 0.189 | 0.184 | 0.176 | **0.178** | 0.181 | 0.180 |
| field sessions (block) | — | — | — | −7.93 | **−8.28** | −7.44 | −11.78 |

⚠ In-domain synthesises its audio at eval time, so it crosses the `9c56e02` chain
boundary and cannot be ranked against a pre-boundary record; the column is here for the
trough-to-trough trend only. Every other row reads fixed audio off disk.

**Two gates match or beat the ladder**: Dawn WER 0.178 with deletion 0.089 against
0.180 / 0.094, and keep preservation across the 27 near clips is *better* (median delta
+0.04 dB, p=0.044, paired). **The cross-chain far clips moved for the first time** —
`qvf_price_far1` −17.2 against −3.6, `qvf_price_far2` −30.1 against −18.5,
`qvf_gym_session` −14.0 against −8.9 as block means. That wall had not moved for any
version before this one. Caveat: n=6 in that group and the aggregate median delta is not
significant, so it needs an independent replication before it counts as an axis.

**What is still short** is far suppression on the device chain: field sessions −8.28
against −11.78, cold-far with an ambient lead-in −3.58 against −14.51.

## Why the remaining gap is not budget, room mix, or row length

Three candidate explanations, three experiments, all negative:

**Budget.** Epochs 60–119 introduce nothing — every track's last point is at 60 — so
they are 60 epochs of pure budget at the final values. Sessions went −7.93 → −8.28 →
−7.44 across three blocks in that stretch (paired p=0.688 between the first and last).
The ASR gates kept improving over the same epochs, so budget still buys something; it
does not buy this.

**Room mixture.** `train_dpcrn_curriculum_hardbank.yaml` retires the two easy room
views at epoch 60 instead of leaving them a third of the draws, so the end-state pool
matches the ladder's bank composition. Forked from the parent's epoch 60, 40 epochs.
Sessions **−5.95** against the parent's −8.28 at the same epoch: retiring the easy views
made it *shallower*, not deeper. The easy views are not diluting the hard end; they
appear to hold a reference class that keeps the near/far contrast legible.

**Row length.** `train_dpcrn_curriculum_fixedlen.yaml` drops the mixed row lengths and
trains every row at 6 s, which is what the ladder trained at. Same fork point, same 40
epochs. Sessions **−2.40**: the largest negative of the three. The prediction came from
the measured cost of *adding* the length axis to the ladder (2.05 dB shallower
sessions) and the sign did not transfer. The mechanism is that the last length
distribution wins: the parent's mixed-length phases install long-context anchoring, and
40 epochs of 6 s-only training take it back out. Adding the axis is cheap, removing it
is not.

Both forks moved the same way — every suppression number shallower, every keep number
better (fixedlen's double-talk keep −1.19 and KEEP ok-rate 96/4 are the best in the
table, better than the ladder's 94/6, and six QVF keep clips improve 1.2–3.9 dB). They
are not worse models; they slid down the aggressiveness trade-off, and aggressiveness is
what this recipe is short of.

## What that leaves

The one axis nobody has tested is the ladder's own mechanism: each rung trained to
convergence on a *fixed* distribution before the next rung's weights started from it.
The scheduled run reproduces the sequence of knob values with continuous ramps; it does
not reproduce converging on each distribution in turn. A staged variant — three or four
segments, each fixed for its whole cycle — is the direct test, and it is a different
question from this one.

## Deliverable

`pretrained_ckpt/dpcrn_curriculum_v0.ckpt` (this run's ep99) is **not** a replacement
for the deployment default: it gives up 3.5 dB of field-session suppression. It is the
reproducible single-run baseline — one command, one config, no warm-start chain, ~70% of
a lineage that took eight — plus the first movement on the cross-chain wall, and the
scheduling mechanism the next recipe line can write its phases with.

## Costs and caveats

* `dry_blend 1.0` is not an option for this checkpoint: Dawn WER goes to 0.234 with
  deletion 0.144 (0.176 / 0.088 at 0.9). It leans on that knob harder than the ladder.
* BUT-OFFICE reads +0.009 and the extreme-reverb monitor +0.024, both inside their noise
  bands on a set that cannot resolve them; the primary WER gate is the moderate set.
* This run's validation set uses the recipe's file constants, which are each track's
  *epoch-0* value — so its validation rows carry no real recordings, no media voice and
  no mic high-pass, and its validation loss is measured on an easier distribution than
  the ladder's runs were. Validation loss is therefore comparable within this run only.
  A recipe written after this one should put the schedule's *final* values in the file
  and let the curriculum write the early ones, so validation measures the end state.
