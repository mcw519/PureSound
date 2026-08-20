> ⚠️ **Measured on field set v2 (27 clips, 90D/270D only).** The field benchmark was
> rebuilt as v3 on 2026-08-20 (64 clips, four orientations + seven QVF clips, new
> hand labelling) — see `benchmarks/field_test_vector/SET_V3.md`. **Numbers below do
> not compare against v3 runs.** Synthetic and WER stages are unaffected.

# What the model believes about distance, on the clips it fails to suppress

`../../scripts/probe_distance_head.py`, run 2026-08-18 on `dpcrn_v8/v9/v10` over the 27
`field_cases/test_vector_cases` clips at `dry_blend 0.9`. Raw scorer output in
`dist_head_field_v{8,9,10}.tsv`.

The DistHead is training-only -- inference never reads it -- which is what makes it a clean
probe: it was never tuned to make the field set look good.

**Read both halves.** The first answers the question the probe was written for. The second
came out of the control groups, was not what the probe was aiming at, and matters more.

---

# Part 1 -- the distance cue survives our capture chain

## The question it was written to settle

* **A.** the cue survives to the bottleneck but the mask ignores it -> couple mask depth to
  the estimate.
* **B.** the cue is gone on this capture chain -> only data from that chain helps; no loss
  or dosage change will.

## Answer: A, and not marginally

Near and far estimates do not overlap on any version (near clips read from the foreground
slot, far clips from the interferer slot):

| | near est. max | far est. min | gap | ratio |
|---|---|---|---|---|
| v8 | 0.61 m | 1.71 m | +1.10 m | 2.80x |
| v9 | 0.76 m | 1.45 m | +0.69 m | 1.91x |
| v10 | 0.74 m | 1.50 m | +0.76 m | 2.03x |

23 clips, zero overlap. The bottleneck separates a 30-50 cm user from a 2-3 m bystander on
exactly the chain where suppression fails.

## The model's own estimate predicts its behaviour better than the truth does

Spearman rho against reduction, over the 10 labelled cold-start far clips:

| | rho(true distance) | rho(**model's estimate**) |
|---|---|---|
| v8 | +0.57 (p=0.086) | **+0.77 (p=0.009)** |
| v9 | +0.71 (p=0.021) | **+0.81 (p=0.005)** |
| v10 | +0.71 (p=0.021) | **+0.82 (p=0.004)** |

This settles the talker confound `../field_test_vector/RESULTS.md` point 4 flagged. The
estimate knows nothing about labels or identity and predicts suppression *better* than the
label does, so the split is distance, not who is speaking.

## And there is a hard threshold in that estimate

v10, sorted by what the model believes:

```
est 1.50 m  ->  -11.95 dB   270d_far1 (labelled 200 cm)
est 1.57 m  ->   -9.46 dB   270d_far3 (labelled 300 cm)
est 1.65 m  ->   -6.25 dB   270d_far2 (labelled 200 cm)
est 1.75 m  ->   -3.53 dB   90d_far6  (labelled 300 cm)
est 1.77 m  ->   -7.82 dB   90d_far1  (labelled 200 cm)
est 1.85 m  ->  -10.86 dB   90d_far2  (labelled 200 cm)
----------------------------- ~1.9 m ------------------------------
est 2.04 m  ->   -0.59 dB   90d_far3  (labelled 300 cm)
est 2.10 m  ->   -0.51 dB   270d_far4 (labelled 300 cm)
est 2.25 m  ->   -0.16 dB   90d_far5  (labelled 300 cm)
est 2.27 m  ->   -0.51 dB   90d_far4  (labelled 300 cm)
```

Perfectly separating, no exceptions. The two clips *labelled* 300 cm that the model places
under 1.8 m are both suppressed -- its estimate explains its behaviour where the label does
not.

## Coverage is not the reason the estimate is compressed

True 0.30-0.50 m reads 0.53-0.76 (over); true 2.00-3.00 m reads 1.45-2.27 (under). The head
never predicts past ~2.3 m even for a 3 m talker. The obvious suspect is too few far
examples, and it is ruled out -- the real 22.4% of the training bank
(`real_rir_16k_train_view`, 54,615 far channels) reaches well past 3 m:

```
far channels  p50 2.52 m   p90 5.19 m   p99 11.82 m   max 15.51 m
              68.0% beyond 2 m    38.4% beyond 3 m    21.4% beyond 4 m
```

Compression toward the training far median with a tail to 15.5 m under a SmoothL1 on
log-distance is the likelier mechanism, but this probe does not establish it: the head and
the mask read the same bottleneck, so which of them saturates first is not separated here.

---

# Part 2 -- distance is not what cold start needs

## Both slots report a number even when that source is not there

The head is a regression with no "absent" output, so it always answers. Reading the
controls (v10):

| what is actually in the clip | foreground slot | interferer slot |
|---|---|---|
| user only, **no bystander** | 0.60-0.74 m (correct) | **1.95-2.51 m** -- nobody is there |
| bystander only, **no user** | **0.80-0.91 m** -- nobody is there | 1.50-2.27 m (correct) |
| both talking | 0.67-0.74 m (correct) | 1.77-1.89 m (correct) |

**This is the cold-start failure.** On a lone-bystander clip the foreground slot reports a
user at 0.8-0.9 m. The model believes someone is there, so of course it does not suppress.
The decision cold start actually needs is not *how far* but *is anyone there*.

## The presence signal exists, is directionally consistent, and is 20-100x weaker

It is not absent. All three independently-trained checkpoints separate present from absent,
in the same direction -- with no near user, the foreground estimate drifts *outward*:

| | user present (near + dt, n=15) | user absent (lone far, n=10) | margin |
|---|---|---|---|
| v8 | 0.53-0.74 m | 0.80-1.10 m | **+0.06 m** |
| v9 | 0.64-0.87 m | 0.88-1.24 m | **+0.01 m** |
| v10 | 0.60-0.74 m | 0.80-0.91 m | **+0.06 m** |

Against the same head's margin on *distance*, both sources present, v10: 0.74 m vs 1.77 m,
**+1.03 m / 2.4x**.

So: distance carries a metre of headroom, presence carries one to six centimetres. Three
trainings agreeing on the sign says the signal is real rather than noise; a 1 cm margin on
25 clips says it cannot be gated on as it stands. Something to amplify, not to wire up.

The interferer slot carries no presence signal at all -- lone-user clips report a bystander
at 1.95-2.51 m, lone-bystander clips at 1.50-2.27 m, completely overlapping.

## This explains two earlier results

* **Near-anchor dependence** (isolated far -1.2 dB vs anchored -19.4 dB): with an anchor the
  model never has to judge presence, the user is audibly talking. Without one it must, it
  cannot, and it defaults to "someone is there".
* **The gate-only experiment failing on real recordings** (2026-07-10, near and far both read
  ~0.9): a gate *is* a presence detector, and presence is the weak axis.

Two independent attempts, one finding: the distance information is there, the presence
information is not.

## Double-talk is the constraint, not the risk

The model places both sources correctly at once (v10): foreground 0.67-0.74 m against a true
0.30-0.50, interferer 1.77-1.89 m against a true 2.00-3.00, 2.49-2.79x apart on all four
clips. And the field scorecard has every double-talk near voice surviving in all four
versions, worst keep -2.35 dB. So it is the healthy case.

What it does constrain is the shape of any fix. **This head is utterance-level** --
`DistHead.forward` is `self.net(x.mean(dim=(2, 3)))`, pooling over frequency *and time* to
one number per clip. In double-talk both sources are live, so a single global "the
interferer is at 1.8 m" used to set mask depth would attenuate the user along with the
bystander. Anything conditioned on distance has to be per-frame; the current head cannot be
wired up as it is.

## What follows

Not "couple mask depth to the distance estimate", which was the reading from Part 1 alone.
Distance would improve double-talk, where nothing is currently wrong, and leave cold start
untouched, because cold start is bounded by presence.

The target is **making "is there a near user" a decision with headroom**. What is known:
the signal exists, its sign is consistent across three trainings, and it currently has
1-6 cm of margin against 103 cm for distance. Whether to amplify it by supervision (an
explicit near-presence output, using the target-absent labels the dataset already emits) or
by representation (per-frame rather than utterance-pooled) is the open question.

Double-talk is the guardrail on that work: it is the only case that can show the user is
not being suppressed along with the bystander, and all four versions currently hold it.
