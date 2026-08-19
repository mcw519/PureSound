# Presence margin by RT60 band -- the "before" v11 is judged against

Run 2026-08-19, `presence_stratified_probe.py` beside this note. This is the
measurement `full_gate/v8_gate050_VERDICT.md` demanded: the gate's deletion cost
rose monotonically with reverberation and no single-room scorecard could see it,
so any future presence signal is judged **per RT60 band**, never pooled.

Frames come from the live training distribution (`config/train_dpcrn.yaml` valid
loader, 30 batches, per-row rt60 from the bank metadata). Truth per frame from
the energy labeler on the row's own stems: present = target speech active,
absent = target silent AND far active, neither = dropped. Score = the offline
linear readout (`presence_readout_v8.npz`, fitted on the 90D field recording)
on the frozen v8 bottleneck.

## The "before": chance overall, and the confound in one column

| rt60 band | n present | n absent | med(present) | med(absent) | AUC |
|---|---|---|---|---|---|
| 0.0-0.3 | 23,642 | 6,344 | **-1.02** | -1.54 | 0.571 |
| 0.3-0.5 | 32,346 | 8,412 | **-2.08** | -2.28 | 0.523 |
| 0.5-0.7 | 30,774 | 8,541 | **-2.49** | -2.62 | 0.533 |
| 0.7-0.9 | 22,256 | 7,013 | **-3.04** | -2.67 | **0.472** |
| 0.9+ (real rooms) | 706 | 472 | -5.50 | -6.67 | 0.558 |
| ALL | 109,724 | 30,782 | -2.25 | -2.36 | **0.529** |

Two findings:

* **The present-frame score falls monotonically with RT60** (-1.02 -> -3.04):
  the reverberation confound, measured directly. At 0.7-0.9 the order inverts --
  AUC 0.472, a talking user scores *more absent* than an absent one. This is the
  exact mechanism that deleted the user in the full gate, now visible in a table.
* **The boundary cuts both ways.** 2026-07-10 fitted on synthetic and read 0.500
  on real; this readout fitted on real reads 0.529 on synthetic. Neither
  direction transfers across the chain. The utterance-level probe's 0.958
  held-out number is real-to-real; it says nothing about this distribution.

## What v11 must do to pass

The v11 heads train on exactly the rows this table is computed from, so in-band
AUC will be high almost by construction. The pass criteria are the two things
training cannot give for free:

1. **No inversion, and margin held, in the 0.5-0.9 bands** -- where
   `bank_drr_overlap_README.md` shows DRR cannot answer (near p5 -4.71 below the
   dry-room far median -3.54). A head that learned DRR will sag exactly there.
2. **The same table on the real field recordings** (the b-trajectory extraction
   grid) without refitting anything: the chain boundary is the wall every
   readout has died on, in both directions.

Reference point, same probe in `--head` mode on the 2026-07-10 gate head
(`dpcrn_v6_gate`, no EMA, pre-realfar bank, 4 batches): AUC 0.751 overall on its
own training distribution -- and it never transferred to real. Beating 0.75
in-domain means nothing; the two criteria above are the gate.
