# v12 candidate axes (2026-08-22)

Where things stand: `dpcrn_v8` ships. `v11b` ep31 (training stopped mid-cycle)
is the safest keep model on record (field-set keep violations 4 -> 1) and the
best presence-head carrier, but pays ~half the anchored far suppression and
does not replace v8 (`probes/v11b_VERDICT.md`). Two mechanisms are now
independently validated: **compression invariance** (heals the QVF keep
deletions, closes ~half the scenario3 inversion, cumulative with exposure) and
**relative judgment** (within-recording ordering survives both the chain
boundary and checkpoint drift; absolute thresholds died on both --
`probes/presence_selfcal_README.md`).

Discipline, unchanged: one axis per run, warm-start v8 unless stated, judge at
a cosine trough with `probes/presence_head_judgment.py` + the set-v3 scorecard,
180d stays held out of every fit.

| # | axis | what, concretely | evidence FOR | main risk / cost | judge by |
|---|---|---|---|---|---|
| 1 | **Chain factors round 2** | offline probes first: codec, EQ, noise gate through the same phantom-distance methodology that convicted compression; only the factor that moves the readout earns a recipe knob | compression explained ~1/3 offline and ~1/2 in training; the remainder is unclaimed; scenario3 still inverted (0.411) | probes are ~0 GPU; a dead end costs days not weeks | phantom-distance shift offline; then scenario3 AUC in training |
| 2 | **Resume v11b to ep39 trough** | ~8 GPU-h; settles whether the anchored-suppression cost (-12 -> -6.6 dB) is a mid-cycle artifact | every ep19-vs-ep31 head trend is still improving; the standing lesson says mid-cycle numbers are not verdicts | suppression may not come back -> same verdict, 8 h later | set-v3 sessions reduc vs v8; heads must not regress |
| 3 | **Gate operating point, held-out protocol** | fit calibrator + PresenceGate knobs (keep_bias, T, taus, b_hi) on 90d/270d + non-held QVF, read 180d ONCE; then wire as opt-in inference flag | the ep31 head only works behind self-calibration; one -5.25 dB interjection deletion is the open cost; scenario3 actuation exists (-1.9 dB) | overfitting six sessions is the trap the protocol exists to prevent | 180d single read: violations 0, suppression > fixed arm |
| 4 | **M6 / boundary banks for the HEAD** | swap/union `hybrid_rir_16k_m6_20260804` + `boundary_20260811` banks into the head-training recipe only | bank axis's real defect is COVERAGE (zero at the 1-2 m decision boundary) and M6's DRR gradient matches measurement -- the head's whole job is that boundary | RIR-axis non-transfer x3 on record; M6 banks already measured FLAT on every separator wall (m6bank runs, 08-04..10) -- expect head-metric gains or nothing | scenario3 / 180d AUC and boundary-band discrimination, NOT synthetic suppression |
| 5 | **Reference architecture (layer 2)** | train the reference in: room-embedding conditioning (two segments, same room, embed one -- possibly bystander-only -- judge the other) or a contrastive nearer/farther objective | every failure on record is a missing comparison point; the model demonstrably compares (rho +0.599) and never reads absolutes; layer 1 proved the offset structure but cannot manufacture evidence at t=0 | architecture-level lift; RIR->real transfer must be earned again on the head side | cold-start lone-far on real recordings -- the one number nothing has ever moved |

Recommended order: **1 (offline part) immediately** -- it is nearly free and its
outcome re-ranks everything else; **3 in parallel** (also free, uses existing
caches); then decide 2 vs 4 vs 5 with those answers in hand. If chain factors
claim the rest of the scenario3 gap, v11c = v11b + those knobs and axis 5 waits;
if they come back empty, axis 5 is the main line and deserves the design page.
