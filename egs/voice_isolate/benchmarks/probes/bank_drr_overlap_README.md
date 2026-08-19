# Can DRR alone separate presence in the training bank?

Run 2026-08-19, `bank_drr_overlap.py` beside this note. Motivation and the
pre-registered decision rule are in the v11 planning discussion: the presence
heads must NOT be learnable as a DRR threshold, because DRR falls with distance
*and* with RT60, so a DRR-shaped head gates out reverberant near users -- the
exact failure `full_gate/v8_gate050_VERDICT.md` measured.

DRR from the RIR wavs with the repo's own `compute_drr_db` at the recipe's 2.5 ms
window; 6,000 items sampled uniformly from the live bank (origins at their
natural ~22% measured share), all five channels each.

**Pre-registered rule** (stated before measuring): best single-DRR-threshold
balanced accuracy on near-vs-far channels
`<= 0.90` -> config-only v11; `>= 0.97` -> mix in the high bank.

## Verdict: 0.874 -> CONFIG-ONLY. And the fallback is refuted.

### Live bank `hybrid_rir_16k_realfar` (28,070 channels)

| | p5 | median | p95 |
|---|---|---|---|
| near DRR | -4.73 | +0.98 | +7.14 dB |
| far DRR | -12.97 | -7.87 | -0.38 dB |

Best single threshold -3.65 dB -> **balanced accuracy 0.874**. The interleave
band [-4.7, -0.4] dB holds **23.0% of all channels**: a head that only learns DRR
is wrong on nearly a quarter of the material, so the gradient pressure to find a
non-DRR cue exists in the live recipe as it stands.

### The ambiguity is almost entirely the measured rooms

| origin | near med | far med | best-thr acc |
|---|---|---|---|
| sim (77.6%) | +1.43 | -8.18 | 0.917 |
| **real (22.4%)** | **-1.69** | **-4.85** | **0.660** |

In the measured rooms the two classes sit 3 dB apart and a DRR threshold barely
beats chance. The realfar merge (2026-08) is what makes this bank DRR-ambiguous;
the simulated side alone would be close to the 0.97 shortcut regime.

And the near-side DRR falls with RT60 exactly as the confound requires
(near median +5.03 at RT60<0.3 down to -0.91 at 0.7-0.9; near p5 reaches -4.71,
*below* the dry-room far median of -3.54): near-wet vs far-dry crossover mass
exists in-bank.

## The high bank would SHARPEN the shortcut, not break it

The planned fallback -- "if DRR is sufficient, mix in `hybrid_rir_16k_high`
(RT60 0.85-1.5)" -- is measured and refuted:

| | near p5 | far p95 | gap | best-thr acc |
|---|---|---|---|---|
| high bank alone | -2.91 | -5.54 | **2.6 dB clean** | **0.989** |
| live + high at 10% | | | | 0.885 (up from 0.874) |

Its far channels are uniformly deep (far-wet: median -9.48) while its **near
channels keep positive DRR even at RT60 1.2-2.0** (median +0.72) -- the hybrid
generator's near side does not lose DRR with reverberation the way measured rooms
do. This is the same generator pathology the M6 audit found (DRR-distance slope
too steep on the synthetic side): in this bank, wetter rooms make DRR *more*
separating. Mixing it in would have reinforced exactly the shortcut v11 must
break. **Do not add it for the presence heads.**

## Caveats

* This classifies *channels* by DRR; the head sees rendered speech frames.
  Rendering adds source-dependent variance on top, which can only blur a DRR
  threshold further -- the 0.874 is an upper bound on how far a pure-DRR head
  gets, which is the conservative direction for this decision.
* Balanced accuracy weights the two classes equally; the training row mix does
  not (interferers outnumber foregrounds 3:2 per room, row types vary). The
  interleave-band mass (23%) is the sampling-independent statement.
* rt60 above 0.9 in the live bank is real-rooms-only (n=200 in the sample) --
  thin, but those rows are also the most DRR-ambiguous ones.

## What this sets up

v11 trains the two presence heads on the live bank unchanged -- one variable.
The rt60-stratified table above is the "before" reference the stratified presence
probe has to be scored against: the heads earn their keep exactly where near p5
crosses below dry-room far medians (RT60 >= 0.5), because there DRR cannot answer
and something else must.
