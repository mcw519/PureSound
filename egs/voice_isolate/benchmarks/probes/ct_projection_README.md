# CT->device projection -- real-chain targets are manufacturable, at 6.25 dB

2026-08-23. Scripts beside this note: `ct_projection_poc.py` (first pass),
`ct_projection_render.py`, `ct_projection_joint.py` (overlap handling),
`ct_projection_holdout.py` (the correction), `ct_projection_sweep.py` /
`ct_projection_taps.py` (the ceiling). Material: NOTSOFAR-1 MTG_31060,
4 speakers, 6 minutes, CC BY 4.0.

Question: can worn close-talk (CT) mics be projected onto a distant device
channel to make TRAINING TARGETS that live entirely in the real capture chain
-- no RIR convolution anywhere in the supervision path? That is the data the
"convolution vs end-to-end recording" wall says we need, and the prerequisite
for a meeting-mode task and for layer-2 contrastive pairs.

## Verdict

**Yes, at 6.25 dB (about 75% of speech-bin energy), and that is a hard
ceiling for the linear family.** Usable, with a real and quantified cost.

## RETRACTED: the first 9.4-12.2 dB headline

The first pass fitted per-utterance filters and scored them on the SAME
utterance. `ct_projection_holdout.py` shows that was ~6 dB of overfitting:

| scoring | 13 taps |
|---|---|
| in-sample (fit and score on one utterance) | 11.88 dB |
| fit on first half, score on second (~1 s later) | 5.47 dB |
| **block-interleaved (128 ms blocks, ~0 s gap, no shared frames)** | **5.87 dB** |

Interleaved scoring, which removes shared data at near-zero time gap, already
drops 6 dB. Physical drift adds only 0.4 more. So the gap is overfitting, not
movement. Two consequences, both corrections to what this note said before:

* the headline number becomes ~6 dB, not 9.4-12.2;
* "the residual fell BELOW the room noise floor" was read as success. It is
  the opposite -- a projection that only captures speech cannot cancel noise.
  It was the overfitting signature, visible and misread.

## The ceiling is real: three knobs, one number

All block-interleaved (honest). `ct_projection_sweep.py`, `ct_projection_taps.py`:

| fitting window | best over ridge |
|---|---|
| the utterance alone | 6.05 |
| **+/-5 s / +/-15 s of the same speaker's solo frames** | **6.25** |
| +/-60 s | 3.65 |
| all solo frames (one filter per speaker) | 0.55 |

| filter length (past taps) | at +/-15 s |
|---|---|
| 208 ms | 6.25 |
| **400 ms** | **6.27** |
| 640 ms | 6.18 |
| 960 ms | 6.22 |

Ridge 3e-3 / 3e-2 are within noise of each other; 3e-1 costs ~1 dB.
**Nothing moves the number.** The "our filter is shorter than the room's
reverb tail" hypothesis is refuted: 208 ms and 960 ms score the same.

What DOES survive from the first pass, with corrected magnitude: **time
locality matters**. One filter per speaker for the whole meeting scores 0.55
dB honestly; a +/-15 s window scores 6.25. Filters stay usable for ~15 s and
are worthless meeting-wide.

Since window, length and regularisation are all exhausted, the residual 25%
is most likely where the mapping stops being linear-time-invariant at all --
head/torso directivity changing within an utterance, and whatever the worn
mic does to its own signal. Breaking that needs a different estimator family
(nonlinear/neural, or the CTRnet route), not tuning.

## Overlapped speech: works where it can be estimated, which is rarely

Overlap cannot be fitted one speaker at a time (least squares would explain
the other voices with this speaker's CT). The general form is to cut the
meeting at every activity change and solve all active speakers JOINTLY on
each interval; solo is the n=1 case. On a demo excerpt (16 s, 23% overlap,
`ct_projection_joint.py`), overlap regions go from 5% explained (solo-only,
which outputs nothing there) to **60%**.

But four speakers' worth of taps need ~2 s of continuous overlap, and natural
overlaps are interjections: only **2 intervals in the whole meeting qualify,
8% of overlap frames**. Transplanting a neighbouring solo filter was tested as
the fix and FAILED -- the staleness curve is brutal (own 10.1 / 1-5 s 4.5 /
5-20 s 1.6 / 20-60 s **-2.3**, i.e. worse than not projecting), so forcing
100% coverage that way collapses overlap quality to 0.52 dB. Factory v1
therefore covers solo intervals plus the rare long overlap, and leaves the
rest empty.

## Why 75% is still usable

The projection is by construction a linear filter of that speaker's own CT
signal. It therefore **cannot** contain room noise, **cannot** contain another
speaker (beyond CT bleed, measured at -25..-45 dB), and **cannot** drop a
syllable. The 25% shortfall shows up as room-colouring error, not as missing
speech -- so this target cannot teach the deletion failure that every other
axis of this project has been fighting. That is the trade being made:
synthetic targets are exact but do not transfer; these are approximate in
colouring but live in the real chain.

## What this unlocks

1. **Meeting-mode keep-target factory v1**: solo intervals + long overlaps,
   +/-15 s fitting window, 25 taps, ridge 3e-3. Meeting selection is part of
   the pipeline -- a 6-speaker high-overlap meeting (MTG_30984) yields almost
   nothing, while MTG_31060 (everyone >= 16 s solo) is good material.
2. **Layer-2 contrastive pairs**: the same projections give one utterance's
   image at several devices on one chain.

## Caveats

* One meeting fully measured; the cross-meeting scan is cheap and still owed.
* All numbers are speech-dominant T-F bins (device energy >= floor + 10 dB).
* The listening page uses in-sample projections -- what the factory actually
  ships -- so it sounds better than 6.25 dB implies.
