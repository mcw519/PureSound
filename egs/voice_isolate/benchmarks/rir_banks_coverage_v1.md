# The coverage bank (v12a) -- four RIR pools served as one

2026-08-22. Consumed by `config/exp/train_dpcrn_v12a_coveragebank.yaml` as a
recipe-level `banks:` union -- nothing is physically merged, every member keeps
its own layout and provenance, and `weight` is a sampling probability. This
note is the composition record and the coverage measurement.

## Why

The bank axis's measured defect is **coverage, not fidelity** (2026-08-10):
both earlier generators had ~zero far-channel mass in the 1-2 m decision band,
and the hybrid generator's synthetic DRR slope is 3.8 dB/decade too steep while
M6's matches measured rooms (-10.83 vs -10.53 dB/decade). Meanwhile RIR
fidelity was quantified as NOT the wall (LTI fit <= 0.8 dB, fidelity gap
<= 0.6 dB) and M6 banks measured FLAT on every separator wall when swapped in
alone (m6bank runs, 2026-08-04..10). So this bank is scoped to what coverage
can plausibly buy: the **presence/distance heads**, whose whole job is the
near/far boundary. Expectation for the separator: none.

## Members, measured from their manifests

Far-channel distance mix (% of far channels) and rt60 mix, train+qc-pass only:

| member | items | 1-1.5 m | 1.5-2 | 2-3 | 3-5 | 5+ | rt60>=1.5 s |
|---|---|---|---|---|---|---|---|
| `m6` path-events-m4 release | 39,560 | 0 | 0 | 59 | 39 | 2 | 15% |
| `m6_bnd` boundary release | 8,180 | 38 | 53 | 9 | 0 | 0 | 13% |
| `real` measured view | 27,448 | 13 | 10 | 47 | 30 | 1 | 0% |
| `hybrid` prod pool (v8/v11) | 122,792 | 1 | 1 | 63 | 35 | 1 | 0% |

Notes that matter: the M6 main release has NOTHING under 2.05 m -- every
synthetic 1-2 m sample comes from the boundary release; the boundary release's
rt60 extends past 1.5 s, which puts synthetic mass exactly in the
boundary x high-reverb corner where the reverberation confound lived; 5 m+ is
scarce in every pool (~1-2%), so no target can honestly ask for more.

## Weights: 0.30 / 0.20 / 0.40 / 0.10 (m6 / m6_bnd / real / hybrid)

Hand-set against four requirements, in priority order: fill 1-2 m to roughly a
quarter of far draws; keep measured rooms the anchor (0.40); make M6 the
dominant synthetic (correct gradient) while keeping the hybrid generator
present for diversity and lineage continuity (0.10); leave the rt60>=1.5 s tail
present but small. An L1-optimal fit to a uniform target was rejected: it
pushed `real` to 0.75 because the unreachable 5 m+ band dominated the loss --
a reminder that optimising to an arbitrary target is worse than stating
requirements.

Expected mix at these weights (from the member table):
**1-2 m ~27% of far channels (production pool today: ~5%)**, 2-3 m ~45%,
3-5 m ~27%, 5 m+ ~1%; ~7% of scenes at rt60 >= 1.5 s; origins
M6 0.50 / measured 0.40 / hybrid 0.10.

## Sanity run (the training loader, not a reimplementation)

`AudioEffectAugmentor.init_room_bank` on the recipe's own block
(`usage_role: train` injected exactly as `DynamicBaseDataset` does), 600
`sample_room_scene()` draws: 1000 `sample_room_scene()` draws (union init 107 s -- the release manifests are
read once per process):

* member shares **0.10 / 0.30 / 0.21 / 0.40** (hybrid / m6 / m6_bnd / real) --
  on design to within sampling noise;
* far-channel mix **1-1.5 m 14% / 1.5-2 m 18% / 2-3 m 38% / 3-5 m 25% /
  5 m+ 5%** -- the 1-2 m band lands at **32%** of far draws against the
  production pool's ~5%, and the 5 m+ tail comes out richer than the
  manifest-sample estimate (the served scenes carry `distance` on every far
  channel, including real-bank channels my manifest pass had to drop for a
  missing `distance_m`);
* scenes at rt60 >= 1.5 s: **8%**.

The loader path exercised is the training one: release split=train + qc=pass
filtering, `usage_role` injection as `DynamicBaseDataset` does it, per-member
wav caches, union routing by sampling weight.

## Bench semantics change, recorded on purpose

For models trained on this pool, full-gate stage 4 (high-reverb bank, rt60
0.86-1.49, "UNSEEN reverb") and stage 5 (boundary held-out, "UNSEEN boundary
distances") are no longer out-of-distribution probes -- training now covers
both bands (different rooms, same ranges). Read them as held-out-room
generalisation checks for this lineage. The only OOD frontier left is real
end-to-end recordings, which is where every wall has been anyway.

## What is deliberately NOT in the pool

* `hybrid_rir_16k_boundary_heldout` and `real_rir_16k_heldout*` -- eval-only.
* `hybrid_rir_16k_high` -- stage 4's probe bank stays out even though its rt60
  band is now covered by M6 draws; the ROOMS stay unseen.
* `hybrid_rir_16k_anechoic` -- 240 probe rooms (rt60 0.07-0.12), kept as the
  out-of-range probe for the distance-readout work.
* The flat `hybrid_rir_16k_m6_realfar_260812` view -- superseded by this
  union; its rewritten JSONs carry no release lineage (see the release README).
