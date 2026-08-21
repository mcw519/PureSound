# What the near/far cue is and is not made of

Run 2026-08-20/21 on `dpcrn_v8` (the deployed default) and the v11a heads.
Scripts beside this note: `level_matched_cases.py`, `compression_probe.py`,
`ema_contrast_by_rt60.py`, `anechoic_level_dependence.py`.

Prompted by a physical analysis of what should break in extreme rooms under a
mono, near-realtime constraint. Four of its predictions were testable against
this pipeline. **Two confirmed, two refuted**, and the refutations are the more
useful half because they close off repair directions that looked obvious.

## Confirmed: high reverberation collapses the cue by ABSOLUTE value

The near talker's own direct sound loses to the tail as RT60 grows, and its
absolute DRR walks into the range a distant talker occupies in a dry room:

| rt60 band | near DRR median | far DRR median | gap |
|---|---|---|---|
| 0.0-0.3 | **+5.03** | -3.54 | 8.6 dB |
| 0.3-0.5 | +2.21 | -6.93 | 9.1 dB |
| 0.5-0.7 | +0.37 | -9.22 | 9.6 dB |
| 0.7-0.9 | **-0.91** (p5 -4.71) | -10.71 | 9.8 dB |

**The gap never narrows; the absolute value stops identifying the class.** A
model that learns a DRR threshold must fail as reverberation grows, and the
deletion cost measured in `full_gate/v8_gate050_VERDICT.md` does exactly that:
+0.015 (Dawn) -> +0.026 (RT60 0.20-0.65) -> +0.148 (RT30 0.56-0.69) -> +0.215
(RT30 1.15-1.84).

The sharpest framing of it is not ours: **at any one frame the model cannot tell
whether the energy is a near talker's direct sound or a far talker's tail from a
second ago.** That is why the cue is only meaningful by comparison, and it makes
one fact out of four that were filed separately:

| observation | |
|---|---|
| anchored far suppression -19.4 dB vs isolated -1.2 dB | a reference exists / does not |
| double-talk healthy in all four versions | both sources live = mutual reference |
| deletion rises monotonically with reverberation | absolute threshold decays |
| cold start is THE defect | single source removes the reference |

## Refuted: the EMA bank does not flatten with reverberation

The prediction was that a long tail keeps the slow averages elevated and
collapses the fast/slow contrast, costing the head its onset transient. Measured
on the v11a head's actual bank (50 ms against 4 s), on the live training
distribution:

| rt60 band | fast-vs-slow contrast |
|---|---|
| 0.0-0.3 | 0.306 |
| 0.3-0.5 | 0.311 |
| 0.5-0.7 | **0.326** |
| 0.7-0.9 | 0.319 |

Flat, if anything rising. The reason is an implementation detail worth writing
down: **the EMA runs on bottleneck features, not on energy.** Those 128 channels
are network activations; they do not accumulate monotonically because a room
rings. The prediction is right about energy and does not transfer to this
feature space.

**Caveat that matters:** RT60 <= 0.85 only, because no bank above it exists. The
prediction may still hold past 1.5 s and we cannot say.

## Refuted: the model does not fall back on level when the room cue thins

Two independent tests, and the second was built specifically to give the
hypothesis its best chance.

**On real recordings.** `level_matched_cases.py` renormalises each field
recording so its own user voice sits at the device median -- -14 to -28 dB on the
QVF publication clips. The phantom foreground estimate on lone-bystander clips
moved **0.70 -> 0.71 m**; the -28 dB clip moved 0.00. (This also withdraws a
`rho = -0.49` correlation reported earlier as if it were the mechanism: it was
confounded, and the causal claim built on it is retracted.)

**In a near-anechoic room.** A purpose-built bank (`exp/hybrid_rir_16k_anechoic`,
240 RIRs, small rooms at the Sabine floor) reaches RT60 0.070-0.120 s, well below
the 0.17 training floor. It genuinely thins the cue: near DRR median +13.67 dB
against the trained distribution's -0.9..+5.0, and the **near-vs-far gap narrows
from ~9 dB to 5.44 dB**. Then a 24 dB level swing:

| | estimate shift across -18 / 0 / +6 dB |
|---|---|
| trained (rt60 0.17-0.85) | **0.000 m** |
| near-anechoic (rt60 0.07-0.12) | **0.000 m** |

**Level invariance survives the cue being removed.** "Quiet reads as far, loud
reads as near" does not happen, in either condition. The repair this implies --
level augmentation, level normalisation at inference -- would buy nothing.

## The question this opened, which is better than the one it closed

Between the two banks the cue thinned by 3.6 dB of near/far separation and the
estimate moved **0.670 -> 0.665 m**. Insensitive to level, and insensitive to how
much reflected sound there is.

So either the estimate rides on something else -- spectral tilt and the near-field
proximity boost are candidates (the QVF clip analysis measured centroid
351 -> 620 Hz), or the very-early structure inside the 2.5 ms DRR window, which an
anechoic room makes *cleaner* rather than removing -- **or it is not measuring at
all out of distribution and is reporting a prior**. 0.67 m sits close to the
training near median, and a reading that stable looks more like a constant than
like a measurement.

That is the next thing to settle, and it is decidable: feed anechoic material at
several true distances and see whether the estimate tracks. If it does not, the
distance readout does not function outside the training range, which would change
how every out-of-distribution number on record should be read.

**Settled (2026-08-21): see `dist_cue_anatomy_README.md`.** It tracks
(rho +0.599 on the anechoic bank) at ~1/26 of the true scale -- a measurement,
not a prior -- and the cue is the direct window's fine timing, read jointly
with the spectrum as a comparison, not summed with it.

## Premises corrected along the way

Three statements about this system that the analysis assumed and that are not
true of it:

* **It is not strictly causal.** `delay: [1, 1, 1]` is 3 frames / 30 ms of
  look-ahead, recorded in every export manifest as `streaming_delay_frames: 3`.
  A genuinely causal config exists (`train_dpcrn_wide_causal.yaml`, delay 0) and
  has never been trained.
* **The heads not seeing the mask is a design choice, not a causality limit.**
  They read the bottleneck after `dprnn_block2`, upstream of the mask; in
  streaming the mask for frame t is available at frame t. Wiring one to the other
  is possible and untried.
* **The shared bottleneck is not saturated.** A *linear* readout on the frozen
  bottleneck separates presence at 0.953 across RT60, and the same bottleneck
  simultaneously supports the mask and DistHead's distance regression (near/far
  non-overlapping, 2.8x). What fails is the background head at 0.707 in-domain
  and 0.475 on real recordings -- with identical architecture on identical
  features, which points at supervision rather than capacity.

## Caveats

* The anechoic bank is near-anechoic, not anechoic: the Sabine floor for a
  2.5-4.0 m room is 0.065-0.082 s, so a true free field is not reachable with
  this generator.
* The level sweep is 2 batches per condition on CPU (the GPUs were training) --
  enough for a median that does not move at all, not enough for a small effect.
* `dpcrn_v8`, not v11a, for the level tests: v11b was training and v8 is what
  ships. The EMA measurement is on the v11a head, since that is the component the
  prediction was about.
