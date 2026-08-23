# Close/far pair ordering on new chain families -- and the v12a autopsy

2026-08-22. `corpus_pair_probe.py` beside this note. Corpora (downloaded to
`exp/real_e2e_corpora/`, all commercially usable): DiPCo (CDLA-P-1.0, dinner
party, arrays 1-4 m across the room), AMI (CC BY 4.0, meetings, table arrays
0.5-1.5 m), NOTSOFAR-1 (CC BY 4.0, 30 conference rooms, table arrays). Every
corpus records the same moment on a worn close mic AND a distant device, so
the probe asks the threshold-free question: on solo-speech windows, does the
close version read more near-present than the far version? Labels come from
mic identity -- no diarization model, no calibration, within one chain.

## Pair ordering (near-head), higher is better

| checkpoint | DiPCo | AMI | NOTSOFAR | field rig cold-start AUC |
|---|---|---|---|---|
| v11a ep19 | **1.000** | 0.912 | 0.875 | 0.841-0.895 |
| v11b ep31 | **1.000** | 0.896 | 0.913 | 0.883-0.923 |
| v12a ep19 | **1.000** | 0.784 | **0.490** | **0.423-0.713** |

Absolute far-side medians tell the mechanism. Under v11a the far device reads:
DiPCo -1.58 (8% called present at threshold 0), AMI +2.12, NOTSOFAR +2.71,
rig far +0.8..+1.7. Under v12a EVERY real-chain far reading moves up -- DiPCo
+0.38, AMI +3.15, NOTSOFAR +5.07 (indistinguishable from its close mics at
+5.10), rig far +5.6..+6.0 (indistinguishable from rig near) -- while
in-domain head loss is the best of the three checkpoints (0.3411) and the
synthetic-far discrimination survives. One mechanism covers all of it:

**The v12a coverage bank (real member at 0.40 sampling weight) taught the head
that a measured-room reverberation signature means a user is present.** The
keep class became dominated by real-RIR rooms; every real-chain far recording
carries that signature; the more ordinary the room (conference rooms, the rig
apartment), the harder the false-present pull. The mirror image of the
2026-08-19 reverb confound ("reverberant near = far"); this time real-reverb
= near. Chain identity leaked into the presence label through the data mix,
exactly what `augmentation_realnear`'s design note warns about.

Two readings that survive v12a: DiPCo ordering stays perfect (the cue
gradient is intact when distances are unambiguous), and qvf_gym improves to
its best value on record (0.776) -- production-processed chains carry little
natural room signature, so they dodge the confound.

Side observation, consistent across all three corpora: v11b ep31's DistHead
transfer is degraded vs v11a ep19 (pair-ordered fg_dist 0.53-0.92 vs
0.97-0.99); v12a partially restores it (0.64-0.95). Training-only head,
recorded as a multi-task-drift datapoint.

## Verdicts

* **Coverage bank v1 weights FAIL the head-transfer gate.** Not the coverage
  thesis itself -- the real-member dominance. The 1-2 m boundary question is
  still unanswered; the clean next experiment is v11b's bank + ONLY the
  boundary member (single variable, no real-weight change).
* **These corpora are now standing diagnostic instruments**: three chain
  families, threshold-free, hours to run, and they dissected a checkpoint the
  field set alone could not have scoped. Keep them eval-only until their
  train/eval split is deliberately decided.
* Fourth confirmation of the relative-judgment thesis where distances are
  unambiguous (DiPCo 1.000 for every checkpoint), and the sharpest
  demonstration yet that absolute calibration is a per-chain lottery.

## The distance ladder is shifted one notch against our spec (ear-verified)

Confirmed by listening (2026-08-23) and consistent with the numbers: the
corpora's "close" is a worn mic at ~5 cm -- one notch NEARER than our user
(30-50 cm at a device mic), a different capture style entirely (proximity
effect, no room). The meeting corpora's "far" table arrays sit at 0.5-1.5 m --
which is OUR KEEP/boundary distance, not our suppress class. Only DiPCo's
across-the-room arrays map onto our far class. Neither corpus publishes
speaker-device geometry, so these are physics/ear calibrations, not labels.

Consequences: (1) pair ordering here measures RANKING, never keep/suppress
classification -- part of NOTSOFAR's "array reads present" is correct product
behaviour, and the v12a verdict rests on the rig far clips (true 2-3 m), not
on this. (2) Ingestion roles flip: headsets are reference sources for pair
manufacturing, NOT keep-class inputs (teaching "5 cm proximity = user" is the
v12a confound in another coat); the table arrays are the real-chain KEEP
material we actually lack; DiPCo's distant arrays are the suppress material.
(3) DiPCo's five dispersed arrays support a three-rung ladder probe
(headset / near array / far array) that maps onto the product spec better
than any binary pair.

## Caveats

* AMI windows are energy-dominance picks (annotation-free v0); AMI/NOTSOFAR
  "far" devices sit at 0.5-1.5 m -- inside our keep/boundary band -- so their
  ordering ceiling is legitimately below 1.0 and their absolute "present"
  rate is not by itself an error. The rig far clips (labeled 2-3 m) are what
  make the v12a regression unambiguous.
* Solo windows only; n = 75-125 pairs per corpus; single far device per
  corpus (U01.CH1 / Array1-01 / first mc ch0).
