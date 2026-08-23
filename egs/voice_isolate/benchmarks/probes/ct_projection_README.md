# CT->device projection POC -- real-chain keep targets are manufacturable

2026-08-23, `ct_projection_poc.py` beside this note. Question: can NOTSOFAR's
worn close-talk (CT) mics be projected onto a distant device channel to make
TRAINING TARGETS that live entirely in the real capture chain -- the data the
"RIR convolution vs end-to-end recording" wall says we need, and the
prerequisite for both a meeting-mode task and the layer-2 pair factory?
Method: per-frequency FIR (13 past + 1 future STFT taps, FCP-style, least
squares with diagonal loading), following arXiv:2606.13109 / 2605.19695.

## The four runs, and what each settled

1. **MTG_30984 (6 speakers)** -- wrong exam: four of six speakers had zero
   solo utterances, so the sum test was invalid. Lesson for the factory:
   meeting SELECTION is part of the pipeline; high-overlap meetings yield
   almost nothing.
2. **MTG_31060 (4 speakers, everyone >= 16 s solo), one filter per speaker**:
   projection beats the naive lag+scale baseline by +14..+31 dB (the filter
   decisively learns the room), bleed -25..-45 dB (one -14.8 case), lags
   0-20 ms -- but residual drop on solos only 1.2-4.6 dB.
3. **Speech-dominant-bin drop (>= floor + 10 dB): 1.3-4.3 dB** -- killed the
   "it's just noise the target correctly excludes" excuse. The static filter
   genuinely misses speech.
4. **Per-utterance filters (the NTT recipe): 9.4-12.2 dB** on the same bins,
   all four speakers. The static filter's miss was TIME-VARIANCE -- head
   movement plus device clock drift over a 6-minute meeting -- and
   per-utterance estimation absorbs it.

| speaker | naive | global filter | per-utterance |
|---|---|---|---|
| Bert | -20.2 | 1.4 | **9.4** |
| Jim | -28.2 | 4.3 | **9.7** |
| Serena | -26.5 | 3.4 | **12.2** |
| Sofia | -16.4 | 1.3 | **10.0** |

(first column: naive-baseline SI-SDR; others: speech-bin residual drop, dB)

## Verdict

**PASS, with the per-utterance recipe.** Projected targets explain ~88-94% of
speech-bin energy on solo utterances, aligned to the mixture, zero RIR
convolution anywhere in the supervision chain. Good enough for a first
training recipe; not perfect -- residual speech ~-10 dB under the target will
soften supervision slightly, and the upgrade path (CTRnet-style unsupervised
cleaning, needed anyway for OVERLAPPED regions where no solo filter exists)
is known and deferred.

## What this unlocks, in order

1. **Keep-target factory v1**: solo utterances only, per-utterance projection,
   meeting selection by per-speaker solo yield (MTG_31060-grade meetings).
   Feeds a meeting-mode task (keep everyone in-room, suppress out-of-zone +
   noise + reverb) whose input/target pairs are both real-chain.
2. **Layer-2 pair factory**: the same projections give same-utterance
   images at multiple devices/distances on one chain -- contrastive material.
3. The listening page for the global-filter version (undersells the
   per-utterance result) was published for ear-verification of run 2.

## Caveats

* One meeting fully measured; the generalization scan across meetings is
  cheap and still owed.
* Per-utterance filters exist only where solos exist; overlap regions are
  out of scope for factory v1.
* Utterances shorter than ~1 s (< 40 frames) are skipped -- too few frames
  for 15 taps.
