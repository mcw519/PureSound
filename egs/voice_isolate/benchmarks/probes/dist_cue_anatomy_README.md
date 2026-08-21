# Where the distance cue lives -- timing, read as a comparison

Run 2026-08-21 on `dpcrn_v8` (the deployed default). Scripts beside this note,
in the order the investigation ran:

| script | question |
|---|---|
| `dist_cue_is_it_a_prior.py` | is the readout measuring, or reporting a prior? |
| `dist_cue_which_window.py` | which time window of the RIR does it read? |
| `dist_cue_smear_check.py` | does the committed smear knob remove that window's cue? |
| `dist_cue_decompose.py` | exact magnitude/phase split (the first-principles tool) |
| `dist_cue_when_or_what.py` | inside the window: WHEN (timing) or WHAT (spectrum)? |
| `dist_cue_ood_check.py` | are the damage numbers information loss, or a wrecked model? |

Picks up the question `extreme_conditions_README.md` ended on: between the
trained and near-anechoic banks the cue thinned by 3.6 dB and the estimate moved
0.670 -> 0.665 m, so either it rides on something neither manipulation touched,
or it is a prior. All numbers are medians over 10 rooms of
`hybrid_rir_16k_realfar` (unless the bank is the variable), one 2 s speech
excerpt, CPU.

## 1. It measures -- but at ~1/26 scale, and only across groups

Same speech rendered through RIRs at known true distances, estimate read off the
foreground slot:

| bank | true 0.3-0.7 m | 0.7-1.1 m | 1.1-2.0 m | 2.0-3.0 m | Spearman rho |
|---|---|---|---|---|---|
| anechoic (rt60 0.07-0.12) | 0.529 | 0.586 | -- | 0.633 | **+0.599, p < 1e-4** |
| trained (rt60 0.17-0.85) | 0.530 | 0.673 | 0.525 | 0.735 | +0.227, **p = 0.15** |

Not a prior: on the anechoic bank the estimate tracks true distance. But 2.7 m
of true range maps onto 0.104 m of estimate range -- **the readout preserves
order at about 1/26 of the true scale**, which is why it can separate the
near group from the far group (0.61 vs 1.71 on the field set, 2.8x) while
ranking *within* a group is not significant even in the trained bank (the
non-monotonic middle bins are room-to-room variance swamping the signal).

This narrows `dist_head_field_README.md`: "distance information is intact"
holds for two-group separation only. Continuous distance is not available.

## 2. The window is the direct arrival

Replace one window of every RIR with noise at its own RMS -- energy kept,
structure gone -- and read how much of the near/far estimate gap survives:

| damaged window | gap kept |
|---|---|
| none (baseline, gap 0.145) | 100% |
| **direct, 0-2.5 ms** | **45%** |
| early, 2.5-50 ms | 62% |
| late, 50 ms+ | 60% |

## 3. The smear lever removes it almost entirely

`smear_direct_arrival` (committed dark as `augmentation_reverb.direct_smear`)
convolves the post-peak window with a random unit-energy causal kernel, peak
index and window energy preserved:

| smear | gap kept |
|---|---|
| 0 ms | 100% |
| 2 ms | 24% |
| **5 ms** | **1%** |

So the knob verifiably attacks the dominant cue. Whether it should ever be
turned on is a different question -- see the reading below.

## 4. Inside the window: WHEN, not WHAT

A real signal is completely described by magnitude (what frequencies) and phase
(when they happen). `dist_cue_decompose.py` splits them exactly, and its
self-check prints the guarantee: phase-randomised keeps |X| to 3.7e-08 relative,
magnitude-flattened keeps phase to 3.1e-08 rad, both keep energy x1.000000.
Unlike the smear, neither forces the peak index, so these numbers are not
directly comparable to part 3's.

Applied to the first 5 ms of every RIR:

| condition | gap kept |
|---|---|
| none | 100% |
| **timing destroyed** (phase randomised) | **35%** |
| spectrum destroyed (magnitude flattened) | 81% |
| both destroyed | **54%** |

Timing carries the cue. And the last row is the anomaly that forced part 5:
destroying BOTH kept *more* separation than destroying timing alone, which is
impossible for independent information channels -- more damage cannot restore
information.

## 5. The anomaly is not out-of-distribution contamination

Suspicion: phase-randomised RIRs are "weird inputs" the network has never seen,
so part of the 35% drop is pathology, not information loss. Two whole-model
readouts on the same manipulated inputs, neither of which is the distance head:

| condition | SI-SDR (enh vs early target) | bottleneck mean abs |
|---|---|---|
| none | 11.91 dB | 1.0292 |
| timing destroyed | 12.91 (**+1.00**) | +1.2% |
| spectrum destroyed | 10.72 (-1.19) | -0.1% |
| both destroyed | 11.53 (-0.38) | +0.9% |

Separation quality does not collapse -- the timing-destroyed condition actually
*improves* it -- and the bottleneck norms move within 1.2%. The manipulated
impulses are ordinary audio to the network. **The gap drops are specific to the
distance readout, so part 4's numbers stand as information loss.**

## Reading: a comparison, not a sum

With OOD ruled out, the surviving explanation for 54% > 35% is that timing and
spectrum are not summed, they are *compared*: scramble one and the comparison
misfires hard; scramble both and two scrambled readings partially cancel. The
percentages are therefore not an additive budget ("timing 65%, spectrum 19%")
-- they are probes of a joint readout.

That makes this the same fact `extreme_conditions_README.md` recorded from the
outside, now visible inside the direct window: **nothing in this system is an
absolute cue.** Anchored-vs-isolated needed a reference talker, DRR-by-RT60
needed a reference room, and timing-vs-spectrum is a reference pair.

## Consequences

* **The v11c "break the cue" recipe is deprioritised.** Its premise was that
  smearing timing forces the model onto spectral tilt as an independent backup.
  Part 4/5 say tilt is not independent -- destroying it barely moves the gap
  (81%) yet its interaction with timing is strong. A training run may find a
  joint solution anyway, but the cheap inference from these numbers is that
  there is no absolute fallback waiting to be found. The knob stays dark.
* **The lever aimed at the root is a reference, not another cue:** the model
  demonstrably ranks (rho +0.599) and compares, and every failure mode is a
  missing comparison point. Session/room self-calibration is the standing
  candidate (`../../NEXT_STEPS.md` item 5, `presence_probe_README.md`).

## Caveats

* 10 rooms, one speech excerpt, one model (v8), medians only. These are
  mechanism probes, not benchmarks; treat the percentages as coarse.
* Part 5's SI-SDR target is the early-window render of the *same* manipulated
  impulse -- it scores whether the model still behaves, not absolute quality.
* The 1/26 scale is measured on the anechoic bank; the trained bank cannot
  measure it (rho not significant there).
* Part 1 bins differ between banks because each bank places sources where its
  rooms allow; the missing 1.1-2.0 m anechoic bin is geometry, not censoring.
