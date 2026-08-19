# Is "is there a near user" decodable from the frozen bottleneck?

Run 2026-08-18 on `dpcrn_v8` and `dpcrn_v10`. Scripts beside this note
(`presence_probe_extract.py`, `presence_probe_fit.py`); the features are derived,
so they are not committed.

## Why this was run

`dist_head_field_README.md` Part 2 found that presence -- not distance -- is what
cold start is short of, and that the DistHead's presence margin is 1-6 cm against
103 cm for distance. The obvious next move is an explicit near-presence head, and
the obvious objection is that **this has already failed**: 2026-07-10 froze the
separator, trained a VAD gate head, reached balanced accuracy 0.90 on synthetic
RIR, and produced a gate that read ~0.9 for near *and* far on real recordings,
never closing.

That experiment tested transfer. It never tested decodability. Those are
different questions, and only the second says whether the information is there.

## Method

Freeze the model, hook the tensor entering `dist_head` (the bottleneck,
`[N, 128, F, T]`), pool over frequency and time exactly as `DistHead` does, and
fit an L2 logistic regression (C=0.01) for "a near user is talking in this clip".

* **Real**: the 25 non-session `test_vector_cases` clips -- 15 present (lone-near
  plus double-talk), 10 absent (lone bystander).
* **Synthetic**: the shipped recipe with `realfar.lone_far_prob` forced to 0.0 and
  to 1.0 in separate passes, since absent rows are only 3% of it otherwise. 500
  present / 426 absent. Labels read off the row (`clean_speech` all-zero), not off
  the knob.

**n=25 with d=128 separates random labels**, so every real-side number is reported
against a 200-draw label-permutation null. Without that this is a false-positive
generator.

## Results

| | (i) synthetic 5-fold | (ii) transfer, own threshold | (ii) transfer, best threshold | (iii) real-fit, leave-one-out | permutation p |
|---|---|---|---|---|---|
| **v8** | 0.966 | **0.500** | 0.883 | **1.000** | 0.005 |
| **v10** | 0.969 | **0.500** | 0.850 | **0.900** | 0.005 |

Balanced accuracy throughout. v10's (iii) AUC is 1.000 against a null of mean
0.379 / max 0.780, also p=0.005.

## What each column says

**(i) 0.97 in-domain** -- reproduces 2026-07-10's synthetic number, slightly better.

**(ii) 0.500 transferred** -- reproduces 2026-07-10's real-recording failure
exactly, and the mechanism is visible: mean p(present) is **1.000 on clips with a
user and 0.999 on clips with nobody**. The classifier does not disagree with the
truth, it has no opinion -- it asserts "someone is there" with total confidence on
every real clip.

**(iii) 1.000 / 0.900 real-fit** -- the new result. Fit on the real clips
themselves, presence separates, at p=0.005 against permutation. **The information
is in the bottleneck on the real chain.** 2026-07-10's conclusion -- that the
representation cannot tell near from far on real recordings -- does not hold. What
it measured was that a synthetic-fitted boundary does not transfer.

**(ii) best-threshold 0.883** -- most of the achievable separation returns from
moving the threshold alone, leaving the learned direction untouched. So the
dominant failure is calibration, not representation and not direction.

## Ruled out: clip duration

The feature pools over time; real clips run to 23.7 s (median 8.9) against
synthetic items at 6.0 s, so the collapse could have been a pooling-length
artifact. Re-extracting the real features truncated to 6 s:

| real features from | own threshold | best threshold | AUC |
|---|---|---|---|
| 20 s (as measured) | 0.500 | 0.850 | 0.760 |
| 6 s (matched) | 0.500 | 0.817 | 0.753 |

Same picture. Not duration.

## Caveats

* **n=25, from two recordings.** The permutation null is what makes (iii)
  credible; it is not a substitute for more material.
* **"Best threshold" is picked on the same 25 points**, so 0.883 is an optimistic
  upper bound on what threshold tuning buys, not an estimate of it.
* **Utterance-level.** A deployed gate needs per-frame, which this does not measure.
* v8's 1.000 against v10's 0.900 is two or three clips at this n; not worth reading
  as v8 being better on this axis.

## What follows

An explicit near-presence head will not simply repeat 2026-07-10, provided it
solves calibration rather than only fitting synthetic. The information is there.

Threshold self-calibration is also a cheap lever on its own. Item 5 in
`../../NEXT_STEPS.md` already names "session-start self-calibrated threshold", and
lists it as conditional on other axes producing signal. This is that signal: on v8
the direction learned from synthetic already ranks the real clips well enough for
0.883, and the only thing between that and 0.500 is where the boundary sits.

---

# Per-frame (2026-08-18, follow-up)

The section above asks "is there a near user in this clip". A gate needs "is there
one right now". Same frozen bottleneck, frequency pooling only, no time pooling:
`presence_probe_frames.py` / `presence_probe_frames_fit.py`, v8.

## Material

The two STREAM recordings, which alternate near and far in labelled spans --
21,754 labelled frames at 99.8 fps over **25 spans**. Frames in gaps between spans
are dropped; the label there is genuinely unknown.

Subsampled to every 10th frame (2,176) for the fit: at 100 fps consecutive frames
are nearly identical, and 25 folds x 51 fits on 21k rows buys nothing.

## The split has to be by span

Frames inside one span are almost the same vector, so a random frame split puts
most of a test frame's own span into training. Both ways, same data:

| | balanced accuracy |
|---|---|
| split by frame (wrong) | 0.924 |
| split by span (right) | 0.912 |

Worth reporting that the gap is small. It was expected to be large, and the fact
that it is not means this result is not leakage-driven.

## Result

| | balanced accuracy | AUC |
|---|---|---|
| leave-one-span-out | **0.902** | 0.963 |
| shuffled span labels (50 draws) | mean 0.467, p95 0.599, **max 0.822** | |

p = 0.020. Note the null reaches 0.822 -- with 25 spans, shuffling can get lucky,
so this is solid rather than overwhelming, and weaker than the utterance-level
p=0.005.

So per-frame presence is readable from the frozen bottleneck on real recordings,
at about 0.90.

## Short spans are where it fails

| span length | spans | mean correct | worst |
|---|---|---|---|
| < 2.5 s | 7 | 77.7% | 44.4% |
| 2.5-8 s | 3 | 83.2% | 65.5% |
| > 8 s | 15 | 90.8% | 74.1% |

Length against accuracy is rho +0.39, **p=0.055** over 25 spans -- directional but
not established. Superseded by the time course below, which asks the same question
with hundreds of frames per point instead of 25 spans.

## How long the decision takes to settle

Aligning every span to its own start rather than correlating whole-span accuracy
with length (`presence_probe_timecourse.py`, stride 2 = 20 ms resolution,
predictions still leave-one-span-out so a frame's own span is never in training):

| time since the turn changed | all | near spans | far spans |
|---|---|---|---|
| 0-0.25 s | 51.7% | 64.6% | **32.3%** |
| 0.25-0.5 s | 62.3% | 59.4% | 66.7% |
| 0.5-1 s | 75.5% | 78.9% | 70.4% |
| 1-2 s | 90.9% | 91.3% | 90.4% |
| 2-3 s | 94.3% | 94.4% | 94.3% |
| 3-5 s | 94.8% | 94.9% | 94.8% |
| 5-8 s | 97.3% | 98.2% | 96.3% |
| 8-12 s | 97.5% | 97.7% | 97.0% |
| >12 s | 92.0% | 92.5% | 90.6% |

**Chance for the first quarter second, then usable from about 2 s.** It crosses
90% in the 1-2 s bin and plateaus at 95-97%.

**The first 0.25 s of a far turn is worse than chance -- 32.3%.** When a bystander
starts talking the reading still says "near", actively wrong, for about a quarter
of a second. Near turns read 64.6% over the same window, so the asymmetry is
consistent: **quicker to open than to close.** That is the safe direction for a
gate -- it protects the user rather than cutting them off -- and it costs the
first ~250 ms of every bystander utterance.

The >12 s row is 1,403 frames but from only 8 of the 25 spans, so it reflects
which spans happen to be long, not a decay. Do not read it as degradation.

This is the number a session-start calibration window has to be sized against:
under a second of audio is not enough to decide anything, and ~2 s is where the
decision becomes worth acting on.

## The limitation that matters most here

These are STREAM recordings: the near user talks somewhere in the session, so
context carries across the far stretches. That is the *anchored* condition, and
`near-anchor-dependence` measured anchored far suppression at -19.4 dB against
-1.2 dB isolated. **So 0.90 per-frame here does not transfer to cold start.** The
cold-start question is the one the utterance-level section answers, on separate
clips each played from t=0.

Read the two halves as: during a call, presence tracks frame by frame (0.90). At
the start of a call, presence is decodable but the boundary is in the wrong place
(0.500 as calibrated, 0.883 with the boundary moved).
