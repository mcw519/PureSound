# v11b (compression invariance) -- judged at ep19 and ep31

v11b = v11a + `augmentation_compressor` and NOTHING else (prob 0.35, ratio
1.5-6, threshold -34..-22 dBFS; the same gain curve on mixture and target, so
the target stays the near component of the compressed mixture). Warm-started
from v8, judged with `presence_head_judgment.py` (official -60 dBFS protocol,
reproduces the v11a table to +-0.005). ep19 is the cosine trough and the fair
comparison point against v11a's ep19; **ep31 is mid-cycle** -- training was
stopped there by hand, so ep31 is what exists, but by the standing lesson
(cosine mid-cycle numbers are not verdict-grade) directions are readable and
magnitudes are provisional. The ep39 trough was never reached.

## Heads: the question v11b was built to answer

Cold-start per-group AUC, present-vs-absent, no fitting:

| group | v11a ep19 | v11b ep19 | v11b ep31 |
|---|---|---|---|
| **qvf_scenario3** | 0.252 | 0.365 | **0.411** |
| 180d (held-out) | 0.841 | 0.876 | **0.889** |
| 90d | 0.805 | 0.863 | 0.883 |
| 270d | 0.895 | 0.921 | 0.923 |
| qvf_plumbing | 0.841 | 0.865 | 0.805 |
| qvf_gym | 0.763 | 0.732 | 0.691 |
| qvf_price | 0.712 | 0.682 | 0.720 |

Sessions (keep vs suppress within one recording):

| session | v11a | v11b ep19 | v11b ep31 |
|---|---|---|---|
| qvf_scenario3_session | 0.596 | 0.616 | **0.726** |
| 180d_session | 0.800 | 0.890 | 0.867 |
| 270d / 90d | 0.878 / 0.894 | 0.915 / 0.934 | 0.924 / 0.917 |
| qvf_gym_session | 0.651 | 0.660 | 0.725 |

The pathology clip (`qvf_scenario3_far1`, the bystander every version passes
through): median logit **+8.10 -> +7.28 -> +6.09** across the three columns,
while the true users sit at +5.3..+5.9. Monotone toward correct, **still
inverted** -- the bystander still reads more present than the user.

**Read:** compression is a CONFIRMED partial cause of the QVF inversion, and
its effect accumulates with exposure (ep31 > ep19). It closed roughly half the
gap to 0.5 and did not finish the job; the remaining chain factors
(codec / EQ / noise gate) are untested, and were predicted to exist when the
offline compression probe explained only ~1/3 of the phantom-distance shift.
Bonus not asked for: the rig chain improved everywhere including the held-out
orientation -- real device chains carry mild compression, so the augmentation
is general robustness, not a tax. gym/plumbing dips are single-clip groups
inside the noise band; watch, don't conclude.

The strategic number is **scenario3_session 0.596 -> 0.726**: within-recording
ordering on the worst chain crossed into the range the self-calibrated gate
needs (`presence_selfcal_README.md`), and that combination produced the first
nonzero actuation on scenario3 (-1.9 dB, zero keep damage).

## Separator: what the multi-task + compression pressure did to the mask

Field-set v3 scorecard, ep31 vs the v8 baseline
(`../field_test_vector/records/set_v3/scorecard_v11b_ep31.tsv`, same protocol
as `scorecard_v8.tsv`, dry_blend 0.9):

| axis | v8 | v11b ep31 |
|---|---|---|
| KEEP-VIOLATIONS | 4 | **1** (only the 0d sentinel, -3.97 vs -4.05 -- unmoved) |
| healed | -- | qvf_keep_in_touch_near1 -3.32, **_dt1 -11.79**, qvf_price_near1 -6.29 |
| anchored sessions (90d/270d/180d) | -12.35 / -11.29 / -9.25 | **-6.64 / -5.95 / -6.02** |
| device cold-start far (n=14) | 0 ok / 2 PARTIAL / 12 FAIL | 0 / 0 / 14 |
| qvf_scenario3_session | -0.80 FAIL | -0.91 FAIL |

The compression training healed exactly the QVF-chain keep deletions --
including the worst deletion in the set, a double-talk user cut at -11.79 dB --
at the price of roughly **half the anchored far suppression** (270d and gym now
sit under the -6 bar). Cold start and scenario3 unchanged, as expected: those
were never the separator's to win.

## Verdict

* **Mechanism confirmed, recipe not finished.** Compression invariance belongs
  in every future recipe; it is not sufficient for the QVF chain alone.
* **Not a v8 replacement** at ep31: the anchored-suppression cost is above any
  threshold we would ship. Whether that cost is a mid-cycle artifact is
  unknowable without resuming to the ep39 trough (~8 GPU-h) -- recorded as an
  open option, not an obligation.
* **Best head carrier so far** -- and at ep31 the factory threshold is dead
  (logit drift), so this head is only usable behind self-calibration.
* Next axes ranked in `../v12_candidate_axes.md`.
