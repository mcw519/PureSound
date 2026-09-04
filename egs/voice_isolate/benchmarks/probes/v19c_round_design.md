# v19c round design — pre-registered plan (2026-09-04)

Provenance: produced by an 18-agent workflow (5 diagnostic measurements, 4 independent proposals, 8 adversarial
verdicts, 1 synthesis), all opus/high, on 2026-09-04. Companion file with the five diagnoses and the four killed
proposals: `v19c_diagnosis.md`. Verified by hand before commit: the synthetic wrong-anchor tables
(`anchor_synthetic` arm `bgonly`/`utt`, v8 and v16), and the `eq_probe.py` compressor bug (its `comp()` returns the
gain curve from `compressor_gain` instead of `x * gain`, so the "compression" condition fed the model a gain curve,
not compressed audio — `eq_probe_README.md` Verdict 2 is withdrawn, erratum appended there). Diagnostic scripts and
tables currently live in the session scratchpad `v19_plan/{onset_profile,anchor_synthetic,training_data_audit,
chain_readability,objective_landscape}` and are promoted into this directory as part of R0.4 below.

# v19 pre-registered plan — one axis: wrong-anchor deletion, loss-only

**Recommended round: `v19c-INHERIT`** — an onset-window, within-row *no-anchor-inheritance* hinge on the achieved foreground gain. Warm-started from v16 ep19. No data change, no architecture change, no suppression-term change.

Everything below is pre-registered. Numbers in parentheses are the source; `[D:x]` = diagnosis section, `[F:path]` = repo file, `[V:lens]` = an adversarial verdict's own measurement.

---

## 0. What the diagnosis actually licenses

| Failure | In synthetic training distribution? | Gradient to shape? | Verdict |
|---|---|---|---|
| **Cold onset deletion** (first talker, no anchor) | No — median −0.25 to +0.30 dB fg-projection excess on both WER sets, both checkpoints; real is 11–23× larger `[D:onset_profile]` | **No** | Not a loss axis. Not a v19 round. |
| **Wrong-anchor deletion** (near talker right after another talker's anchor) | **Yes** — Δon1 −1.25 dB (v16) / −1.71 (v8), p≈3e-24, 173/200 rows down, same-talker control null +0.04 `[D:anchor_synthetic]` | **Yes** | **This is the round.** |
| **Anchor forgotten across silence** | Shape absent (re-entry after ≥5 s = 0.3% of rows, 0.0% of the 3 s bucket) `[D:training_data_audit]`, and the deletion is already gone at that gap (−0.49/−0.46 dB at 5/10 s) `[V:evidence, FGMEM]` | No (and coupled to the above) | Not this round; see §6. |
| **QVF chain inversion** (distance readout 0.255) | Features carry the cue (device-fitted probe reads QVF 0.827); the trained head inverts `[D:chain_readability]` | Readout, not data | **Zero-GPU first** (§2), training only if that fails. |

Consequence: this round targets **one** failure, the one with gradient, and it does it with a loss on the unchanged distribution. Cold onset stays covered by the shipped `puresound/system/onset_guard.py` until a real-data round exists.

---

## 1. Sequence, GPU-days, decision points

| Stage | What | GPU-days | Engineer-days | Decision point |
|---|---|---|---|---|
| **R0** Baseline + free wins | Strongest-baseline table; offline readout surgery on the existing agcache; guard replay; two bug fixes; promote the probes | 0 (≤2 GPU-h if a cache needs rebuilding) | 4 | **A**: if a fitted readout behind a session-relative margin beats `pna`, ship it — Round 2A is cancelled |
| **R1-P** Pre-flight | Score the new term at init on train rows, Dawn, and Dawn-through-the-guard; gradient share; DDP safety | 0.1 (~2 GPU-h) | 2 | **B**: kills the axis before any cycle |
| **R1-S1** 3-epoch probe | 2 arms (M_C = 6, 10), warm start v16 ep19 | 0.33 | 0 | **C**: kills the axis; also forks to R2B on a transfer null |
| **R1-S2** Full cycle + block eval | 20-epoch cosine, judged at the ep19 trough; 5-checkpoint block (ep15–19) | 1.06 + 0.20 | 1 | **D**: deploy / fallback |
| **R1 total** | | **≈1.7** | **7** | |
| **R2A** *(conditional)* readout head, head-only | Only if R0 says a trained per-frame readout is needed | 1.1 + 0.2 | 4 (0.5 of it patching the instrument) | — |
| **R2B** *(conditional)* real end-to-end rows, quarter scale | Only if R1's real number does not move | 1.1 per arm | 5 | — |

Epoch cost is measured, not estimated: v16's checkpoint timestamps give 38.1 min/epoch on 2 GPUs = 1.27 GPU-h/epoch `[V:buildability, DECIDE]`.

---

## 2. Round 0 — zero GPU, and it can cancel work

### R0.1 The strongest-baseline table (assembled from records, no runs)

Every v19 gate is read against **both** columns. This is the single amendment every adversarial lens demanded.

| Metric | v8 | v16 ep19 | v16 + `pna` guard | v16 + `drr3` | raw mix |
|---|---|---|---|---|---|
| Dawn deletion (fw large-v3 harness) | 0.230 | **0.207** | **0.111** | 0.094 | 0.084 |
| Dawn WER | 0.333 | 0.297 | 0.217 | 0.188 | — |
| Dawn insertions | — | 0.014 | 0.036–0.050 | 0.036–0.050 | 0.052 |
| Keep violations /148 (`anchor_gate_sim`, fit split) | 26 | 20 | 7 | 3 | — |
| Field sessions suppression, block | −11.78 | −9.73 ±1.41 | −11.0 (fit far median) | — | — |
| Cold-far `none` / `ambient`, block | −1.06 / −12.57 | −1.71 / −13.19 | — | — | — |
| Field keep near violations / worst (`eval_field_block`) | 4 / −7.97 | 3 / −8.34 | — | — | — |
| QVF clip distance AUC | 0.265 | 0.255 | — | — | device-fit probe **0.827** |

Sources: `[F:benchmarks/probes/anchor_gate_README.md §1,§2,§6]`, `[F:benchmarks/field_test_vector/FIELD_BLOCK.md]`, `[F:benchmarks/field_test_vector/COLDSTART_V2.md]`, `[D:chain_readability]`.

Note for the write-up: `eval_dawn_chorus.py` has no onset-guard flag (verified: only `--context`, `--dry-blend`, `--asr`, `--asr-model`), so the guarded column is an **offline gate replay** in `anchor_gate_sim.py` on the agcache, not a paired baseline from the same script. Say so wherever the two are compared.

### R0.2 Readout surgery — CPU, hours, on caches that already exist

`benchmarks/probes/readout_surgery.py` over `scratchpad/agcache/{v8,v16,v11b}/field/*.npz` (`feat` = bottleneck pooled over frequency, 100 fps — the vector any per-frame head would read).

Fit on **device recordings only**, evaluate leave-chain-out on QVF and device leave-one-group-out:

| Variant | What it tests |
|---|---|
| R0 raw logistic | the recorded 0.827 (v16) / 0.789 (v8) ceiling |
| R1 per-clip mean-centred | is the chain shift an additive offset? |
| R2 per-clip standardised | offset + scale |
| R3 affine-free LayerNorm over C | the exact normalisation a head would use |
| R4 within-clip pairwise ranking, scored against the clip's own midpoint | the exact self-calibrated read |
| R5 **SiLU-MLP** on R3 features | covers a *nonlinear* head; without it a probe pass does not transfer `[V:buildability, chain_inv]` |

Report at **clip scope** (per-clip median over labelled spans, AUC over the 9 keep × 6 suppress QVF clips, bootstrap over clips — the measured half-width is ±0.14, so pooled frame AUC and the 4-group count are both under-resolved: plumbing is one clip vs one clip) **and at session scope** (the deployment condition, and the one that collapses: device 0.757/0.781, QVF 0.357–0.517).

### R0.3 Guard replay — **Decision point A**

Patch `anchor_gate_sim.py` with `--feature fitted` and replay the best offline readout behind a **session-relative** margin (absolute dB margins died twice: pooled QVF 0.474 vs within-session 0.886/0.911; v8's 3 dB margin costs v16 ten suppress passes `[F:presence_selfcal_README.md]`, `[F:anchor_gate_README.md §2]`).

- **Ship it, cancel R2A** if the fitted readout beats `pna` by ≥0.03 Dawn deletion **or** ≥3 keep violations /148 at equal far-suppression median.
- **R2A is justified** only if clip scope works but session scope collapses (<0.60 QVF, <0.75 device sessions) *and* a per-frame construction on the same frozen features is measurably better.
- **R2A is closed** otherwise.

### R0.4 Free code, independent of every verdict

1. **Label bug fix**: a row whose gated target is exactly zero is relabelled `target_absent=True`. Measured: 1.0% of rows, all on the 3 s bucket, where SDR/MR-STFT treat the row as lone-far (mask derived from the waveform, `siso.py:305`) while `ResidualReferenceLoss(target_present_only)` still counts it present `[D:training_data_audit]`.
2. **`scripts/check_training_data.py --split train`** (default stays `valid`, so nothing changes for existing callers) + the audit's temporal columns: target onset, interferer-before-onset, longest target-free gap, re-entry rate, target-absent provenance. Today `report_separability` measures **no** temporal property, which is why this distribution was never on the record. The audit's `sample_rows.py` did 1206 rows in 375 s.
3. **Promote the instruments**: `onset_real.py`, `onset_synth.py`, `anchor_synthetic_probe.py` into `egs/voice_isolate/benchmarks/probes/` with committed v16 ep15–19 block baselines. Two of the round's primary gates currently live only in a session-scoped `/tmp` path.
4. **Metric of record**: the foreground-projection gain `P = 10log10((⟨e,r⟩/⟨r,r⟩)²)` replaces span energy for every keep-side claim. Measured separation: interferer removed with the user intact reads −0.00 dB on P and −6.41 dB on span energy; user attenuated 6 dB with the interferer kept reads −6.12 vs −0.77 `[D:decision proposal, verified]`. This closes the span-energy conflation the user caught.

### R0.5 Eligibility and contrast on the **train** loader

Measure, over ≥600 rows through the real v16 dataloader (seed 7, the audit's method): the share of rows with ≥0.5 s and with ≥1 s of interferer speech strictly before the user's first active frame, split by length bucket and row type. Audit's upper bound: any interferer before onset 15.8%, ≥1 s 6.0%, ≥2 s 4.6%.

---

## 3. Round 1 — the one axis

### 3.1 The objective (one new loss class, nothing else)

`AnchorInheritanceLoss` in `puresound/nnet/loss/inherit.py`, `required_inputs = ("enhanced", "target", "batch", "vad_target")`. One config entry at `weighted: 0.25`, ramped 0 → 0.25 over 2 epochs. Total loss weight 7.0 → 7.25.

**Frame grid** — 400/160, identical to the `EnergyVADLabeler` that produces `vad_target`; verified that `vad_reference` is cloned at `ns.py:583`, i.e. *after* the speed-perturb block (`ns.py:487–510`), so `vad_target` is on the post-speed grid. `consistency_noise = noisy_speech − target_speech` (`ns.py:631`, collated `:765`) is on the same grid.

**Scores**, per frame, floored:

```
a_t = <E_t,R_t>/max(<R_t,R_t>, 1e-10)      # achieved gain on the user
q_t = <E_t,N_t>/max(<N_t,N_t>, 1e-10)      # achieved gain on what was there before
pos(z) = softplus(16 z)/16
P_t = clamp(10 log10(pos(a_t)^2 + 1e-12), min = -30, max = 0)
Q_t = clamp(10 log10(pos(q_t)^2 + 1e-12), min = -30, max = 0)
```

The **−30 dB floor is load-bearing**, not cosmetic. Measured on the diagnosis's own Dawn rows: unfloored, mean L rises to 14.89 with median 2.90 — the worst 10% of onsets carry 53% of the mean, so the unfloored statistic is a phase/anti-correlation tail detector as much as a deletion detector; flooring at −30 dB drops the mean to 9.41 (−37%). Also measured: `softplus(16·)` does not remove the dead zone, it *moves* it to `a ≤ −0.85` (dP/da = 0.0069 at a = −1.0), and max|dP/da| is 138.87, not the 100.2 originally claimed `[V:buildability, DECIDE]`. Every statistic is reported as a **median** (and winsorised mean), never a bare mean.

**No per-talker plumbing is needed.** During the bystander's solo run the user is silent by construction, so `mix − ref` over those frames *is* the bystander plus noise. So `N_t` comes from `consistency_noise` — already in every batch, already aligned. This deletes the whole `far_targets` emission the earlier proposals needed, and with it the measured 5%-of-row-length misalignment (`background_speech_reference` is a pre-speed snapshot) and the single-scalar-gain approximation that a linear-but-filtered chain breaks.

**Row eligibility** (evaluated per row, no dataset change):
- `batch["n_interferers"] ≥ 1` (`voice_isolation.py:555`), and
- a *row-level* test that ≥1 s of interferer speech precedes the user's first active frame — this is the **only** use of `background_vad_target`, and a ±5% timing error cannot flip a 1 s row-level test, and
- the user's first active frame is ≥100 frames (1 s) into the row, and
- ≥50 frames of user activity after it.

Fallback pre-registered at R1-P: if the ≥1 s rule yields <5% of rows or a zero median hinge, drop to ≥0.5 s and report the 0.5–1 s and ≥1 s strata separately (the anchor knee is at 1 s: 0.5 s arms only −3.67/−4.14 dB vs 1 s −12.61/−9.72 `[D:anchor_synthetic]`, so the shorter stratum is weaker, not free).

**Frame sets** (both from aligned tensors):
- `O` = the first 50 frames (0.5 s) of `vad_target` activity at the onset — the window the Dawn number is quoted at.
- `B` = frames before the onset with `vad_target == 0` and `consistency_noise` frame energy within 25 dB of its own max over that prefix (so a noise-only prefix cannot make the hinge trivially satisfiable).

**The hinge**:

```
Pbar_on  =  sum_O  w_t P_t / sum_O w_t         w_t  = <R_t,R_t>
Qbar_pre = detach( sum_B w'_t Q_t / sum_B w'_t )   w'_t = <N_t,N_t>
L_inherit = mean over eligible rows of relu( M_C - (Pbar_on - Qbar_pre) )
M_C = 10 dB   (arm A);  6 dB (arm B)
```

Three properties, each answering a recorded failure:
- **Only a difference appears.** No absolute dB anywhere. Absolute calibrations died twice (chain, checkpoint) and relative/self-calibrated survived twice `[F:presence_selfcal_README.md]`.
- **`Qbar_pre` is detached.** The hinge cannot be satisfied by suppressing the bystander *less* — that would buy the keep number with far suppression, which is the v11b / OnsetGuard trade this round exists to avoid.
- **It is a hinge on an onset window, not a row mean.** Measured: the row-mean form is inert — v16's median `Pbar − Qbar` over whole rows is 36.74 dB with 88.2% of rows already above 15 dB, and a 1 s event moves a row mean by ~0.1 dB `[V:evidence, DECIDE]`.

**Nothing else changes.** No suppression term, no floor, no `inactive_mode` (v18: relaxing the floor made every suppression number shallower, cold-far +0.53 dB, p = 0.017). No `OverSuppressionLoss` reweighting (a row-head onset weight would land on 81.7% of rows where onset is at t≈0 and the correct action is already KEEP — 90.5% of such a mask's mass sits where no deletion is measured `[V:evidence, RETURN]`). No length-schedule change (v16's header records 10.5 rows/batch = 88% of v8's scene diversity *deliberately*, so touching it re-imports v15's budget-shortfall confound).

**Zero new parameters** → `preflight_ckpt_recipe.py` must report 0 missing / 0 unexpected against v16 ep19, the streaming ONNX export is byte-identical, and `config/infer_dpcrn.yaml` needs no change (which is what silently no-op'd an earlier round).

### 3.2 Expected effects, each with its citation

| # | Expected change | Basis | How it fails |
|---|---|---|---|
| 1 | Synthetic wrong-anchor Δon1 (other2/other3) moves from −1.25/−1.33 dB toward ≥−0.6 dB, same-talker control (self3 +0.04) held within ±0.1 | The failure is in-distribution at p≈3e-24, 173/200 rows down, and the hinge is that contrast written as an inequality on the frames where the damage sits (Δlate is only −0.10/−0.12 dB) `[D:anchor_synthetic]` | The 6% natural dose is too thin, or the probe's spliced anchor is a construction the training rows never present — R1-P measures exactly this |
| 2 | Dawn bystander-first onset excess @0.5 s (n=217) improves ≥1.0 dB from −6.07 dB, paired | This is the condition the guard's `pna` clause fixes, and the guard proves the frames are recoverable on these very weights (deletion 0.207 → 0.111) `[F:anchor_gate_README §2,§6]` | The 11–23× real/synthetic ratio does not close; this is the transfer null, and it forks to R2B |
| 3 | Dawn deletion (guard OFF) falls below 0.18 with insertions held at ≤0.020 | ~0.13 absolute deletion is onset + wrong-anchor deletion `[F:anchor_gate_README §6]`; **but** no training round on record has put unguarded Dawn deletion below ~0.175 `[V:evidence, RETURN]`, so 0.18 is the honest target and 0.12 is a stretch, not a pass bar | The model learns the guard, not the decision — pass-through scores −0.06 dB on P, so leakage satisfies a keep metric. Insertions and Dawn total WER are gates for exactly this |
| 4 | Far suppression does **not** pay the guard's price: sessions block within 1.5 dB of −9.73, cold-far `ambient` within 2 dB of −13.19 | The hinge saturates on a contrast (silent once a row decides correctly), the suppress side is detached, the floors are untouched, and the added weight is 0.25 of 7.25 — below the 0.1-of-7.0 level at which v11's aux heads already biased the operating point, but not far below `[F:presence_head_v11_README.md]`, `[F:v18_VERDICT.md]` | The contrast is satisfied from the wrong side (lifting `Pbar_on` = onset leakage). Guarded by insertions, the s2f keep curve, and cold-far |
| 5 | **No** change to cold onset deletion, and **no** change to QVF distance AUC | Cold onset is absent from the synthetic median (−0.25 to +0.30 dB); QVF is a chain/readout problem — the compressor is inert on that readout (|dAUC| ≤ 0.013), linear channel consistency is *already* trained in both v8 and v16 at the proposed settings, and QVF scenario clips are context-insensitive `[D:onset_profile, chain_readability]` | Claiming either would be the round's easiest way to look like a win and then fail the block scorecard |

### 3.3 Round 1 pre-flight — **Decision point B** (~2 GPU-h, before any cycle)

| Test | Instrument, n | Kill |
|---|---|---|
| **P0** eligibility + median hinge at init on v16 ep19 | patched `check_training_data.py --split train` + the loss's scoring mode, ≥600 train rows, seed 7 | eligible share <3% of rows at the ≥0.5 s rule, **or** the median hinge over eligible rows is 0 (we would be training a tail, not a behaviour) → **the axis is closed for loss-only; go to R2B** |
| **P1** does the score measure the known fix? | median floored `L_inherit` on 450 Dawn utterances, unguarded vs passed through `onset_guard.py` | guarded ≥ unguarded, or <30% reduction → the score does not measure the failure; redesign the score, do not train |
| **P2** is it a targeted hinge or a global reweighting? | gradient-norm share of the new term at init on a real batch | >50% of total (v18 in reverse). Re-scale below 20% |
| **P3** DDP safety | a batch with no eligible row | must return a graph-carrying zero (`enhanced.sum()*0.0`, the `dist.py:70–71` idiom); a `None`/NaN here severs a DDP job, as the background-VAD crash already did once |
| **P4** alignment | speed-perturbed 30 s row | the loss's onset index must coincide with `vad_target`'s within 2 frames |

### 3.4 Round 1 Stage 1 — **Decision point C** (3 epochs × 2 arms, 0.33 GPU-days)

Read only the fast instruments — no ASR, no field block:

| Read | Kill |
|---|---|
| `anchor_synthetic_probe.py --arm utt` (200 utts, ~10 min/ckpt): Δon1 other2 | has not moved ≥0.4 dB toward zero from −1.25, or self3 falls below −0.3 dB (the model started deleting after its *own* anchor) |
| `eval_coldstart_v2.py` far `ambient` (blow-up detector) | shallower by >2 dB — the v18 column, and the cheapest detector that suppression got shallower |
| `onset_real.py` on a fresh Dawn agcache: bystander-first excess @0.5 s | **the fork, not a kill**: if the synthetic number moved and Dawn is inside ±0.5 dB, stop the round and open R2B. That null is the answer — the same wall that closed the RIR axis, made gate-only adaptation a no-op on real recordings and turned v17 negative — not a "train longer" |

Three epochs is enough to see the sign because this is output-side behaviour on n = 200 paired (baseline −1.25 dB, p≈3e-24) and n = 217 paired (−6.07 dB); v17 and v18 both showed their sign inside the first third of their cycle. Three epochs cannot decide the *price*, which is what Stage 2 buys.

### 3.5 Round 1 Stage 2 — the pre-registered scorecard (5-checkpoint block, ep15–19)

Every row is judged against **v16 ep19 and v16 + `pna`**. No BUT-OFFICE (n = 200, CI ±0.03, no resolution). No single-checkpoint field verdict (same-run ep8–19 cold-far spans −0.35…−3.39 dB, single clips swing 19.9 dB).

| # | Metric | Instrument, n | Pass | Kill |
|---|---|---|---|---|
| **P1** | Dawn bystander-first onset excess @0.5 s, **paired per-utterance delta** vs the v16 block | `benchmarks/probes/onset_real.py` on a per-checkpoint agcache; n = 217; paired Wilcoxon | ≥ +2.0 dB improvement, p < 0.01 | ≤ +0.8 dB, or p > 0.05. *Absolute medians are forbidden as the gate: the marginal bootstrap half-width is ±1.694 dB, the paired one ±0.11 dB* |
| **P2** | Synthetic Δon1 (other2, other3) + self3 control | `anchor_synthetic_probe.py`, n = 200, block median | ≥−0.6 dB both, self3 within ±0.1 | worse than −1.0 dB, or self3 below −0.3 |
| **P3** | Keep violations /148, guard OFF | `anchor_gate_cache.py field` + `anchor_gate_sim.py sweep --split fit`, all 5 block checkpoints (~4 GPU-h, budgeted) | ≤10/148 (between v16's 20 and the guarded 7), and 180d `--split held` not worse | >16/148. *Two-proportion test pre-registered; binomial sd at p≈0.13 is ~4 spans* |
| **G1** | Field keep near violations / worst | `eval_field_block.py compare`, its own 3–5 span scale, harmonic peak–valley (`audit_span_metric.py`) on every span that moved ≥1 dB | ≤3 and worst ≥−7.0 dB, no confirmed speech removal | ≥5, or any confirmed speech removal, or worst past −10 dB |
| **G2** | Sessions suppression, block | `eval_field_block.py`, n = 6 sessions, paired CI (not "p > 0.05" — at n = 6 Wilcoxon cannot go below p = 0.031, so failure-to-reject is not equivalence) | CI lower edge above −8.2 dB | block median shallower than −7.5 dB |
| **G3** | **Cold-far `none` and `ambient`**, block | `eval_coldstart_v2.py` blocks, paired per clip | not shallower than v16 by >1.5 dB | shallower by >2.0 dB. *This is the column v18 died on and no earlier proposal gated it* |
| **G4** | Dawn deletion **and total WER and insertions**, guard OFF | `eval_dawn_chorus.py --context none` and `--context background`, recognizer pinned (`--asr faster-whisper --asr-model large-v3`), deltas vs mix; Azure row from `AZURE_2026-09.md` | deletion ≤0.18 in `none`, WER ≤0.26, insertions ≤0.020 | deletion >0.207, or WER >0.297, or insertions >0.030 |
| **G5** | Moderate-reverb WER (primary WER gate) | `eval_wer.py --set-dir data_report/wer_set_moderate_test --asr azure`, printed bootstrap CI, uncapped subset n = 127 reported separately (the 10 s cap inflates absolute deletion) | delta CI upper edge ≤ +0.01 vs v16 | CI lower edge > +0.01 |
| **G6** | s2f keep curve | `eval_s2f_keep_probe.py` vs the **block** baseline −0.47/−0.60/−0.56/−0.65/−1.11 (not the superseded single-checkpoint row) | no point >0.8 dB below, no new cliff | any point >1.5 dB below, or a new cliff |
| **G7** | Extreme reverb + synthetic far-only | `run_full_benchmark.sh` stages 2/4/8 | far-only within 3 dB of v16; extreme-reverb ΔWER CI not above zero | either fails |
| **M** | monitor only | `on_validation_epoch_end`: median `L_inherit`, eligible-row share, per-term loss curves (`verbose: True` — it is False in both shipped recipes, so the curves the round promises to read would not be emitted) | — | — |

**Decision point D.** A pass means: P1 + P2 + P3 pass **and** every G holds **and** the arm beats v16 + `pna` on at least one of {keep violations at equal far suppression, Dawn deletion at equal far suppression}. A round that only matches the shipped guard at the guard's suppression price produced nothing the guard did not — that is exactly why v14 was killed.

### 3.6 Files to change for Round 1

Paths verified this session.

| # | File | Change |
|---|---|---|
| 1 | `puresound/nnet/loss/inherit.py` **(new)** | `AnchorInheritanceLoss`, `required_inputs = ("enhanced","target","batch","vad_target")`. Adds **no** provider — `consistency_noise`, `n_interferers`, `background_vad_target` all arrive through `batch` (`siso.py:_loss_providers`, verified) |
| 2 | `puresound/nnet/loss/__init__.py` | export in `__all__` — that list is what `loss_func[].type` resolves against |
| 3 | `egs/voice_isolate/config/exp/train_dpcrn_v19c_inherit.yaml` **(new)** | v16 verbatim + one loss entry (`weighted: 0.25`, 2-epoch ramp), `verbose: True`, `work_folder: exp/dpcrn_v19c_inherit`, `max_epochs: 20`, `T_0: 20`. Header states the one knob and the warm-start command |
| 4 | `test/test_losses/test_inherit_loss.py` **(new)** | the floored-P envelope table as assertions; invariance under `P,Q → +b` to 1e-6; zero on a satisfied row; positive on an inherited-deletion row; graph-carrying zero on a batch with no eligible row; onset-index alignment on a speed-perturbed 30 s row |
| 5 | `test/test_system/test_loss_dispatch.py` | register the new loss in the provider-contract test (it reads `siso._loss_providers`) |
| 6 | `egs/voice_isolate/scripts/check_training_data.py` | `--split {train,valid}` (default `valid`) + the temporal columns — from R0, reused as this round's data-shape instrument |
| 7 | `puresound/system/siso.py` | `on_validation_epoch_end` logging median `L_inherit` + eligible-row share on a frozen validation subset (`validation_step` logs losses only today). No provider change, no perturbation change |
| 8 | `egs/voice_isolate/benchmarks/probes/{onset_real.py, onset_synth.py, anchor_synthetic_probe.py}` | promoted from the session scratchpad, with committed v16 ep15–19 block baselines |
| 9 | `egs/voice_isolate/benchmarks/probes/v19c_round_design.md` **(new)** | this pre-registration, committed **before** launch |

**Not touched, deliberately:** `dpcrn.py`, `lobe/heads.py`, `lobe/trivial.py`, `task/overlap_gating.py`, `task/ns.py`, `task/noise_stage.py`, `config/augmentation.py`, `streaming/`, `config/train_dpcrn.yaml`, `config/infer_dpcrn.yaml`, `trainer.length_schedule`.

**Launch** (positional config, `--training` required — `runner.py:build_arg_parser`):

```
cd /home/milowu/A4Audio/PureSound/egs/voice_isolate && uv run python main.py \
  config/exp/train_dpcrn_v19c_inherit.yaml --training \
  --pretrained_ckpt_path /work/any_exp_link/puresound_exp/dpcrn_v16_lengthmix/\
lightning_logs/version_0/checkpoints/epoch=19-step=10000.ckpt
```

Weights only (`strict=False`), so the cosine restarts and ep19 of the new run is a trough comparable to v16 ep19. Never re-run the last stage from scratch: the eight-stage ladder is load-bearing (last stage alone reaches −1.65 dB real far suppression against v8's −9.83), and since no parameter is added the ladder does not need re-running.

---

## 4. What is NOT to be done, and why

| Not done | The recorded result that forbids it |
|---|---|
| Raise `overlap_control.turn_taking_prob` / `far_first_prob` / far-first row dose | **Already run as v10's T2** (`config/exp/train_dpcrn_coldstart.yaml`: 0.30→0.45, far_first 0.5→0.75, row-initial-far 10%→23%): turn-taking KEEP violations 6 → 10 and moderate-range WER 0.024 worse than v8, CI clear of zero. Also: v8 ships *with* realfar 0.3 / realnear 0.5 turn-taking and has the *worse* Dawn deletion of the two checkpoints (0.230 vs 0.207) |
| `augmentation_row_initial_ambient` (room / ambience lead-in) | **v17, negative not null**: cold-far `none` +0.35 dB worse (p = 0.020) and five QVF keep clips degraded 1.7–6.9 dB; keep near worst −8.34 → −11.61. Plus the ERRATUM: room tone is a measured null (−0.04/+0.03 dB, p = 0.08/0.76) — the 90D "ambience" that unlocked cold start contained unlabelled speech |
| Digital silence as a gap filler or a "neutral" prefix | −5.90 dB (v8) / −19.22 dB (v16) suppress bias on the lone-interferer arm, −2.33 dB keep-side onset deletion on v16, and −35.46 dB dt keep median in the field. It stays a **probe** (it is the one synthetic prefix that reproduces Dawn's tail: sil1 p5 −70.2 vs −75.6) and, if ever used in training, only as an explicitly KEEP-labelled stressor row — never as incidental padding |
| Long-gap re-entry rows as the wrong-anchor carrier | The deletion is already gone at that gap: −0.49 dB at 5 s, −0.46 at 10 s, and statistically dead by 20 s (p = 0.24 / 0.93). The row would carry almost none of the failure it is built to teach |
| Raise the target-absent / lone-far rate | Dawn WER 0.392 → 0.626, deletion 0.286 |
| Relax any suppression floor (`inactive_mode=relative`, `stft relative_floor`, VAD `eps_mode`) | v18: every suppression number got shallower, cold-far +0.53 dB, p = 0.017. The "kill to the floor" term does useful work |
| A decay-free / lengthened anchor memory | Persistence and wrong-anchor deletion share one clock (field §6.2, confirmed synthetically: both close by 20 s). Lengthening memory lengthens the window in which the next talker is deleted. Also: the ~10 s forgetting is currently the only thing stopping one bystander utterance from arming suppression for a whole session |
| Wire any presence readout to the mask (FiLM, gate, gain) in this round | b-gate @0.50: turn-taking KEEP violations 6 → 65, SI-SDR +5.42 → −0.23, four gates FAIL — and the field set showed zero cost because it is one room. Prerequisite on record: presence must be invariant *before* it may drive gain |
| Any absolute dB threshold or margin | Died twice (chain, checkpoint): pooled QVF 0.474 vs within-session 0.886/0.911; v8's 3 dB `drr` margin costs v16 ten suppress passes; v11b's factory threshold died from logit drift ep19 → ep31 |
| Input-side chain augmentation for the QVF inversion | The envelope compressor is inert on that readout (|dAUC| ≤ 0.013 at every operating point), the `|x|^p` waveshaper degrades but never inverts (0.989 → 0.880), no modelled stage reproduced 0.26 — and the linear channel-consistency term is **already trained in both v8 and v16** at exactly the proposed settings with QVF still inverted |
| A row-mean keep/leak margin term | Measured inert: v16 median row margin 36.74 dB, 88.2% of rows already past 15 dB, and a 1 s event moves the mean ~0.1 dB |
| A row-head onset weight | 81.7% of rows have the target onset within 0.1 s and the correct action there is KEEP; ~90% of such a mask's mass lands where no deletion is measured (synthetic onset excess −0.25 to +0.30 dB). That is a global reweighting wearing a decision label — v18's shape |
| `dry_blend` as a cold-start actuator | The ceiling never binds: median gain 0.03 dB. A gain moves it (−0.39 → −10.01 dB) but costs a KEEP violation |
| Touch `trainer.length_schedule` in the same round | v16's header records 10.5 rows/batch = 88% of v8's scene diversity as a deliberate choice; changing it re-imports v15's budget-shortfall confound (a 0.10/0.20/0.35/0.35 schedule lands at 7.2 rows = 69% of v16) |
| BUT-OFFICE as a WER gate; span energy as a keep verdict; single-checkpoint field deltas | No resolution (n = 200, CI ±0.03, neither v8 nor v10 provably beats not processing); span energy conflates speech/noise/reverb and user listening has already caught a "violation" that was noise; same-run block spread −0.35…−3.39 dB with 19.9 dB single-clip swings |
| `eq_probe_README.md` Verdict 2 and its priority ordering | Withdrawn: that probe never multiplied its compressor gain in, so the model was fed a constant DC signal (std 2.3e-10). Codec, noise gate and room/mic geometry go **back** onto the candidate list; the standing "+5.19 dB" claim belongs to the `|x|^p` waveshaper, not to `compressor_gain` |

---

## 5. What "real-only" implies for synthetic objectives, and the admissible real path

The diagnosis splits the two failures cleanly, and the plan follows the split:

- **Wrong-anchor deletion is on both sides** of the recording chain (synthetic Δon1 −1.25 dB, p≈3e-24; Dawn bystander-first −6.07 dB). A synthetic objective is legitimate here, and Round 1 is that objective. The magnitudes still differ ~5×, so the *pass gate is read on real recordings*; the synthetic probe is an early-stopping and sanity signal, never the verdict.
- **Cold onset deletion is real-only** (11–23× on the median, 3.9–7.9× on the p5 tail, two checkpoints, two measures; and the synthetic median is *positive* on `indomain_wer_set`). Therefore: **no synthetic objective may claim it**, and any candidate that makes the synthetic onset number worse while the real one improves is behaving as designed, not regressing. The interim answer is the shipped `onset_guard.py`, whose price (far median −15.2 → −12.0 dB, insertions back toward the raw mix) is a product choice to be stated, not hidden.
- **The QVF cross-chain wall is a chain problem**: scenario clips are context-insensitive, the readout inverts while the features carry the cue. It is out of scope for every data round and is addressed, if at all, by R0.3's offline readout or R2A.

**Admissible real-data path (R2B, only if Round 1's real number does not move):** VOiCES, RealMAN, DiPCo, AMI and NOTSOFAR are in-house with commercial-compatible licences and can be *trained* on; `test_vec/` and every private recording stay eval-only and never leave the machine. The row shapes that R2B would build are the two the real record supports — real near-keep rows (round-1's deletion explosion was fixed only by adding them) and real bystander-first turns — at quarter scale, with the standing guardrail that real far-field rows taught real suppression with zero cross-corpus transfer until real near-keep rows were present. The paired corpora also supply the one thing the existing per-frame cache cannot: two talkers per session and the same talker across channels, which is the prerequisite probe for any future talker-identity slot (samespk already costs −0.44 dB, ≈30% of the diffspk effect, so the template is part channel signature today).

---

## 6. Round 2 branches — one axis each, and their gates

- **R2A (readout, head-only).** Only if R0.3 shows a fitted readout cannot ship but a per-frame construction is better on the same frozen features. Train a per-frame, session-relative proximity head; **do not** wire it to the mask (v11's pattern: heads training-only, inference bit-identical). Prerequisite build: `anchor_gate_cache.py` must stash the per-frame head output and `anchor_gate_sim.py` needs a `--readout` branch — today `load_head` hard-loads `dist_head_net.pt` and `head_apply` is a two-layer MLP, so the primary instrument **cannot read a conv head at all** (0.5 engineer-days). Judged by the guard replay (keep violations, Dawn deletion at equal suppression), not by AUC — three rounds already won an AUC and shipped nothing.
- **R2B (real end-to-end rows).** As §5.
- **Not scheduled: explicit foreground memory (a slot / FGMEM).** Gated behind two prerequisites the repo pre-registered itself: (i) a measured talker-vs-channel separation in the bottleneck on paired corpora with a real margin (not merely >0), and (ii) a stated mechanism that breaks the persistence / wrong-anchor coupling *without* extending memory. Both mechanisms in the published analogues (arrival-order caches, reliability-filtered self-enrolment) make persistence *stronger*, which on today's model makes wrong-anchor deletion worse — so they come after Round 1 has fixed *which* talker the anchor should be, not before.


## Open questions carried into R0/R1

- Does the new hinge fire at all on the rows the training set actually contains? The failure is measured at -1.25 dB on a probe construction (spliced 2 s anchor, 200 moderate-set utterances), but the natural dose in training is 6.0% of rows at >=1 s of interferer-before-onset and nobody has measured the model's onset behaviour on those specific rows. Round 1 pre-flight P0 answers it for ~2 GPU-hours and can close the axis before any cycle.
- Has any training change in this repo ever moved unguarded Dawn deletion below ~0.175? Every recorded round (v9, v10, v11b, v14, v16, v17, v18) sits in 0.175-0.240 and only inference-side gating reaches 0.09-0.12. If the answer is no, gate G4's 0.18 target may be at the frontier's edge and the honest deliverable of Round 1 is a smaller deletion move at unchanged suppression, not the guard's number.
- Can a fitted readout behind a session-relative margin actually ship? Round 0.3 decides it, but two sub-questions are open: whether session-scope readability (device 0.757/0.781, QVF 0.357-0.517) is usable at all, and whether the fitted readout survives checkpoint drift the way v11b's absolute threshold did not.
- Is the far-only frame set derived from consistency_noise clean enough? During the bystander's solo run mix-ref is the bystander plus noise exactly, but on rows where the pre-arrival stretch is noise-dominated Q would measure noise suppression and satisfy the hinge trivially. The 25 dB relative energy rule plus the row-level background-speech gate are the guard; their sufficiency is unmeasured.
- Real-near rows carry the recording's own room tone (target 5th-percentile frame -38.9 dB vs -56.5 dB synthetic, active fraction 0.98 vs 0.80). On those rows P is partly computed over room tone and the -25 dB style masks matter. The diagnosis's strict-mask control only moved Dawn from -3.85 to -2.87 dB, so the phenomenon survives; the training-side effect of the mask choice is untested.
- Does the wrong-anchor template's channel component (samespk -0.44 dB, about 30% of diffspk -1.47 dB) shrink under a loss that never mentions channels? Expected: less than the talker component. If it does not shrink at all, the residual is a chain problem and belongs to R2B, not to any loss round.
- Is there a real-recording construction that reproduces the cold-onset failure in a trainable form at all? Today the only synthetic prefix that reproduces Dawn's tail is digital silence, which is forbidden as a filler. Until R2B answers this, cold onset has no training axis and the shipped guard is the answer, with its suppression price stated as a product choice.
- What is the actual wall-clock and memory cost of the new loss? Two unfolds and two inner products per row on resident tensors should be under 2% of step time, but this was not measured, and the 6 s and 12 s buckets already peak at 19.0-19.2 GiB on 23 GB cards.

## AMENDMENT (2026-09-05) — noise-aware re-read; one diagnosis row downgraded, one gate replaced

The user objected that onset/keep dB deltas were compared across sets without accounting for the noise level.
Re-measured in `v19c_diagnostics/snr_strat/` (both scripts re-run by hand):

1. **Dawn Chorus has no usable waveform reference** — `speech` is not waveform-consistent with `mix` (median
   |corr| 0.30 at best lag, polarity flips, lags up to thousands of samples). Every Dawn SI-SDR and every
   foreground-projection number on Dawn is invalid, including the diagnosis's "−3.85 / −3.90 dB @0.5 s".
   Dawn is ASR-only from here (deletion / insertion / WER, paired per utterance).
2. **§0 row 1 is downgraded from "real-only, not a loss axis" to "unresolved."** The 11–23× ratio rested on the
   invalid Dawn numbers and on an SNR-unmatched comparison (synthetic onset windows sit at +0.6 dB local SNR).
   On synthetic data with the background rescaled, the onset-specific excess stays within ±1.4 dB in the
   median down to −15 dB SNR while the *whole utterance* is deleted (g_on −5 dB at −5 dB, −8/−9 at −10 dB);
   the energy-based excess is biased positive at low SNR. Whether real recordings carry an onset-specific
   effect beyond that cannot be measured with today's instruments; the ASR fact that the `pna` guard (dry for
   ~1 s + 2 s release) cuts Dawn deletion 0.230 → 0.123 is the only valid real-recording evidence and it
   stands.
3. **Gate P1 is replaced.** Not `onset_real.py` fg-projection on Dawn. Instead: paired per-utterance ASR
   deletion (`eval_dawn_chorus.py --context none` and `--context background`, large-v3, Wilcoxon) vs the v16
   block, pass = deletion in `background` improves ≥ 0.03 absolute with p < 0.01 and the unguarded-vs-guarded
   deletion gap shrinks by ≥ 1/3; kill = < 0.01 or p > 0.05. Gate G4 unchanged.
4. **Metric policy, binding for every table in this plan**: onset/keep effects are reported as paired
   within-utterance deltas (prefix vs none, or onset vs rest of the *same* utterance) and stratified by local
   SNR of the window; no absolute-dB comparison across sets or chains; energy-based deltas are labelled
   noise-confounded wherever the window's SNR is below +5 dB; the synthetic wrong-anchor result (Δon1) is
   unaffected because it is paired at identical SNR.
5. The round's *axis* is unchanged (wrong-anchor deletion, loss-only) — it never depended on the Dawn
   projection numbers. R0.4 item 4 ("P replaces span energy") is restricted to data with an aligned clean
   reference (synthetic rows, VOiCES/RealMAN-style corpora), never Dawn or the private field recordings.
