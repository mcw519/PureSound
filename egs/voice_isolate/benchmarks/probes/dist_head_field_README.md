# What the model believes about distance, on the clips it fails to suppress

`../../scripts/probe_distance_head.py`, run 2026-08-18 on `dpcrn_v8/v9/v10` over the 27
`field_cases/test_vector_cases` clips at `dry_blend 0.9`. Raw scorer output in
`dist_head_field_v{8,9,10}.tsv`.

The DistHead is training-only -- inference never reads it -- which is what makes it a clean
probe: it was never tuned to make the field set look good.

## The question it was written to settle

* **A.** the cue survives to the bottleneck but the mask ignores it -> couple mask depth to
  the estimate.
* **B.** the cue is gone on this capture chain -> only data from that chain helps; no loss
  or dosage change will.

## Answer: A, and not marginally

The near/far estimates do not overlap on any version:

| | near est. max | far est. min | gap | ratio |
|---|---|---|---|---|
| v8 | 0.61 m | 1.71 m | +1.10 m | 2.80x |
| v9 | 0.76 m | 1.45 m | +0.69 m | 1.91x |
| v10 | 0.74 m | 1.50 m | +0.76 m | 2.03x |

23 clips, zero overlap. The bottleneck separates a 30-50 cm user from a 2-3 m bystander on
the chain where suppression fails.

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

Perfectly separating, no exceptions. Note the two clips *labelled* 300 cm that the model
places under 1.8 m are both suppressed -- the estimate explains the behaviour where the
label does not.

## The estimate is compressed, and it is not for lack of far labels

True 0.30-0.50 m reads 0.53-0.76 (over); true 2.00-3.00 m reads 1.45-2.27 (under). The head
never predicts past ~2.3 m even for a 3 m talker.

Coverage was the obvious suspect and it is not the answer. The real 22.4% of the training
bank (`real_rir_16k_train_view`, 54,615 far channels) reaches well past 3 m:

```
far channels  p50 2.52 m   p90 5.19 m   p99 11.82 m   max 15.51 m
              68.0% beyond 2 m    38.4% beyond 3 m    21.4% beyond 4 m
```

So the head has ample supervision out there. Compression toward the training far median
(2.52 m) with a tail to 15.5 m under a SmoothL1 on log-distance is the more likely
mechanism, but this probe does not establish it: the head and the mask both read the same
bottleneck, so which of them saturates first is not separated here.

## What follows

The wall is not "the cue does not survive our capture chain". It is that the mask stops
acting on a cue it still has, past roughly 2 m of what it believes. That is branch A --
couple mask depth to the estimate, rather than collecting more far-field data.
