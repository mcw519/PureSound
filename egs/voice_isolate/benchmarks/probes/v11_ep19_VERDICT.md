# v11 ep19 presence heads -- the reverb confound is broken; the wall moved to cross-device

Judged 2026-08-20 at ep19 (a genuine CosineAnnealingWarmRestarts trough, T_0=20).
Heads are training-only; this judges the heads themselves, not deployment. The
separator nine-gate check (did multi-task pressure move the mask) is separate and
still owed. Probes on CPU (both GPUs held by training): stratified in-domain via
`presence_stratified_probe.py --head`, real transfer via
`presence_head_real_transfer.py` (no fitting of anything).

## Criterion 1 -- PASS. The reverberation confound is broken.

In-domain near-presence AUC by RT60 band, against v8's fitted readout:

| rt60 band | v8 readout (before) | v11 head | v11 present-median |
|---|---|---|---|
| 0.0-0.3 | 0.571 | 0.977 | 6.61 |
| 0.3-0.5 | 0.523 | 0.969 | 6.11 |
| 0.5-0.7 | 0.533 | 0.933 | 5.51 |
| **0.7-0.9** | **0.472 (inverted)** | **0.966** | 5.97 |
| 0.9+ | 0.558 | 0.984 | 3.02 |
| ALL | 0.529 | 0.959 | -- |

The number that matters is not the AUC (the head trains on this distribution) but
that **the present-frame score does not decay with RT60** (6.61 -> 5.51 -> 5.97,
flat) where v8's DRR-shaped readout fell -1.02 -> -3.04 and inverted at 0.7-0.9.
The head is not DRR-shaped; the margin holds exactly where
`bank_drr_overlap_README.md` shows DRR cannot answer. This is what the EMA
evidence window + explicit supervision were for, and it worked.

## Criterion 2 -- the real wall, and it is now cross-device, not reverberation

v11 heads on the real field clips, no refit, present (keep/near) vs absent
(suppress/far):

| | frame AUC | clip present med | clip absent med | min pres / max abs |
|---|---|---|---|---|
| **RIG chain** (0d/90d/180d/270d) | **0.780** | +4.36 | +0.41 | +1.89 / +3.93 |
| **QVF chain** (ai-coustics) | **0.474** | +1.10 | +0.16 | -2.07 / +8.05 |
| sessions 90d / 270d (keep vs supp) | 0.824 / 0.819 | +4.25 / +4.64 | -0.80 / -0.78 | -- |

**Presence transferred to real recordings for the first time in this
investigation** -- but only across the rig chain. Every prior attempt sat at
chance on real (07-10 gate 0.500 synthetic->real; v8 readout 0.529 real->synthetic).
The rig sessions separate at 0.82, cold-start rig clips at +4.36 vs +0.41.

On the QVF chain (ai-coustics' own phone/web captures, a capture family the
synthetic training never saw) it is at chance -- 0.474, present and absent fully
overlapped, with `qvf_scenario2_far1` and `qvf_scenario3_far1` (bystanders) read
as more present (+7.51, +8.05) than most true near clips, and
`qvf_keep_in_touch_near1` (a user) read absent (-2.07).

## What this means

v11 did what it set out to do: it broke the reverberation confound that killed
the inference-side gate, and made near-presence an explicit signal that survives
the chain boundary for at least one real capture chain. The wall did not
disappear -- it **relocated from reverberation to cross-device capture
signature**. More epochs will not close it (ep39 is the next trough, but the head
loss plateaued by ep13 and this is a data-coverage gap, not convergence): the
training distribution contains no QVF-family chain, so the head has nothing to
learn that generalization from.

## Not deployable, and not the question yet

Even on the rig chain a few far clips leak (the `_far4` clips read +2.7 to +3.9,
above the weakest near clip at +1.89), so a gate on this head would need a
threshold with errors both ways -- and the heads are training-only, so nothing
here changes the shipped output. This judges the representation, which was the
point of v11. Wiring a gate is v12, and only after the cross-device gap is
addressed on the data axis.

## Still owed

* **Separator nine-gate check.** Heads are training-only so inference == the v8
  recipe forward, but multi-task pressure can still have moved the mask. Needs a
  GPU (both held by training) and the standard `run_full_benchmark.sh`.
* n is small and single-clip-per-condition on the real side; the rig-vs-qvf split
  is 24 rig clips vs 14 qvf clips.
