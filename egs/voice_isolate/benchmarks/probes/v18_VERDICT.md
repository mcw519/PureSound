# v18 (three loss/label floor flags) — NEGATIVE on the field block. Flags stay in the tree, default off.

2026-09-03. v18 = v16 recipe + {MRSTFT relative_floor, SDRLoss inactive_mode=relative,
EnergyVAD eps_mode=relative}, warm-start v16 ep19, 20 epochs, judged on the ep15–19 block
(`benchmarks/field_test_vector/FIELD_BLOCK.md` protocol).

| block | sessions | cold-far none | cold-far ambient | keep near viol/worst | dt median |
|---|---|---|---|---|---|
| v16 ep19 (init) | −9.73 | −1.61 | −12.65 | 3 / −8.34 | −1.11 |
| v16 ep39 (+20 ep, no flags) | −11.64 | −1.26 | −14.43 | 4 / −15.68 | −1.65 |
| **v18** | −8.62 | **−0.84** | −10.76 | 4 / −12.11 | −1.79 |

Paired vs v16 ep19: **cold-far `none` worse, +0.53 dB, p=0.017**; sessions +0.46 (p=0.44);
ambient +2.37 (ns); keep near/dt/s2f-guard all unchanged (p>0.1). Vs the +20-epoch control
(v16 ep39) it is behind on every suppression axis and ahead only on keep severity.
Nine-stage single point: in-domain +8.19 (same), Dawn 0.268/del 0.182 (better than v16's
0.296/0.206, single-checkpoint), turn-taking SUPPRESS −16.28 vs −19.29 (shallower).

Reading. The three flags were bundled as a "bug-fix round", so the harm is not attributable
by measurement — but the mechanism points at `inactive_mode: relative`: the historical
absolute-energy term has gradient ∝ 1/‖enh‖ and keeps pushing lone-far residuals down
without limit; scoring "reduction achieved" instead caps that incentive, and every
suppress-side number (sessions, cold-far, turn-taking SUPPRESS) moved shallower while the
keep side did not move. The "kill harder forever" term was doing useful work.

Consequences:
* `inactive_mode: relative` is NOT a free correction; leave it off. `mean` (length-invariant
  floor, same objective) remains the safe variant if the 3 s/30 s spread ever matters.
* The STFT relative floor and VAD relative eps were never isolated; they stay default-off
  as latent-bug fixes for any future level-varying corpus, untested as training changes.
* The v17 re-examination question is answered: lifting the floors does not unlock the
  zero-context column either. Cold start remains a deployment-side problem.
