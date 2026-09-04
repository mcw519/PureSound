# Is the wrong-anchor deletion graded by how much NEARER the anchor was?

Condition 'other3' (= diffspk3) on the utt arm. d = anchor utterance's near_dist minus
the user's near_dist (negative = the anchor talker was CLOSER to the mic than the user
who follows). Metric = Delta_on1: the first 1 s of the user's speech, paired vs 'none'.


## v8  (n=200)   Spearman rho(d, Delta_on1) = +0.149, p = 0.035

| stratum | n | median d (m) | median Delta_on1 (dB) | p (vs 0) |
|---|---|---|---|---|
| anchor nearer than user | 67 | -0.24 | -1.72 | 2.5e-10 |
| similar distance | 66 | +0.00 | -1.49 | 4.3e-08 |
| anchor farther than user | 67 | +0.24 | -1.02 | 7e-10 |

## v16ep19  (n=200)   Spearman rho(d, Delta_on1) = +0.139, p = 0.05

| stratum | n | median d (m) | median Delta_on1 (dB) | p (vs 0) |
|---|---|---|---|---|
| anchor nearer than user | 67 | -0.24 | -1.77 | 4.2e-10 |
| similar distance | 66 | +0.00 | -1.28 | 2.7e-07 |
| anchor farther than user | 67 | +0.24 | -0.78 | 2.1e-10 |

Reading: the effect is graded in the predicted direction (a nearer anchor deletes the
next talker more) but weakly -- and it is highly significant in EVERY stratum, including
when the anchor was FARTHER than the user. So the rule is not 'keep whatever was nearest';
a talker-change component sits on top of the relative-distance component.
