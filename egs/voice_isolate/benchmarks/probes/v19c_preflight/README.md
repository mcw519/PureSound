# v19c Round 1 pre-flight — Decision point B (2026-09-04)

Pre-registered as `v19c_round_design.md` §3.3 (with the AMENDMENT of 2026-09-05 replacing P1's
instrument) and as stage **S0** of `v20_self_enrolling_foreground_design.md` §7. Budget: ~2 GPU-h.
The question it answers, in the design's own words:

> Does the new hinge fire at all on the rows the training set actually contains?

## Verdict

| Test | Result | Pre-registered decision |
|---|---|---|
| **P0** eligibility + median hinge at init | eligible 4.93% (≥1 s) / **5.10%** (≥0.5 s fallback); **median hinge = 0.000** at both arms | **KILL** (criterion 2: "we would be training a tail, not a behaviour → the axis is closed for loss-only; go to R2B") |
| **P1** does the score measure the known fix? | median `L_inherit` **rises** 22.2% (M=10) / 38.2% (M=6) under the guard, needed a ≥30% fall | **FAIL — redesign, do not train** ("the score does not measure the failure") |
| **P2** gradient share of the new term | median **50.4%** of ‖g_total‖ on the batches where L>0, worst **104.5%** | **KILL** (>50% of total) |
| **P3** DDP safety | 5/5 no-eligible-row batches returned a graph-carrying zero; 16/16 forward+backward through `siso.compute_loss`; 0 exceptions, 0 NaN | **PASS** |
| **P4** onset alignment | worst \|loss_onset − vad_target_onset\| = **0 frames** over 60 long rows (3 provably speed-perturbed) | **PASS** |

**Three of five gates fire. `v19c-INHERIT` must not be launched as specified.** P0 is the
pre-registered kill for the axis; P1 says the score is not a measurement of the failure; P2 says the
term at `weighted: 0.25` owns the gradient wherever it is non-zero. P3 and P4 confirm the
implementation is sound — the objective is what is wrong, not the code.

The deliverables (loss class, tests, config, patched data instrument) are committed anyway: they are
what makes this verdict reproducible, and `row_scores` is the instrument any successor objective is
read with.

## Why it fails, in one paragraph

The hinge's contrast `Pbar_on − Qbar_pre` is dominated by **how hard the model suppresses the
pre-onset material**, not by how much of the user it deletes. On the training distribution the
pre-onset material is a *far* interferer that v16 already crushes to `Qbar_pre` = −27.06 dB while
keeping the user at `Pbar_on` = −1.30 dB, so the inequality has +23.93 dB of headroom and the term is
silent on 81% of the rows that qualify for it. On the probe construction where the wrong-anchor
deletion was *measured*, the prefix is a *near* talker v16 keeps (`Qbar_pre` = −7.92 dB), so the
inequality is violated on the median row (contrast +0.43 dB, hinge 9.57 dB). The two populations do
not overlap. A term trained on the first cannot learn the behaviour measured on the second — and the
binding shape constraint is not the loss's to fix: only **6.83%** of training rows have the user's
first active frame ≥1 s into the row at all, because 93.0% of rows start the target within 0.5 s.
**That row shape has to come from data.**

## P0 — eligibility and the median hinge at init

3000 rows of the **train** loader of `config/exp/train_dpcrn_v19c_inherit.yaml`, seed 7, forward on
v16 ep19, `cuda:0`. (A 700-row pilot agreed: 33/700 = 4.71%, median hinge 0, non-zero in 30.3% of
eligible rows.) Full table: `results/p0_TABLE.txt`, per-row records in the scratchpad, summary
`results/p0.json`.

```
>=1.0 s interferer-before-onset (the rule)  :  148/3000  4.93%  95% CI [4.21, 5.77]%
>=0.5 s (the pre-registered fallback)       :  153/3000  5.10%  95% CI [4.37, 5.95]%
```

Both are above the 3% kill line, so **criterion 1 passes**. Criterion 2 does not:

| over the 148 eligible rows | median | p25 | p75 | winsorised mean |
|---|---|---|---|---|
| `Pbar_on` | −1.30 | −4.90 | −0.35 | −4.69 |
| `Qbar_pre` | −27.06 | −27.26 | −23.00 | −24.14 |
| contrast `Pbar_on − Qbar_pre` | **+23.93** | +15.36 | +26.59 | +19.60 |
| hinge, M = 10 | **0.000** (bootstrap CI [0.000, 0.000]) | 0.000 | 0.000 | 1.347 |
| hinge, M = 6 | **0.000** (CI [0.000, 0.000]) | 0.000 | 0.000 | 0.713 |

Non-zero hinge in **28/148 = 18.9%** (CI [13.4, 26.0]) of eligible rows at M = 10, **19/148 = 12.8%**
(CI [8.4, 19.2]) at M = 6 — i.e. **0.93%** and **0.63%** of all training rows. Mean hinge over
eligible rows 1.805 dB (M = 10) / 1.171 dB (M = 6).

The 28 rows that do fire are the right rows — the model deletes the user almost as hard as it
suppresses the bystander — which is why the tail is real but thin:

| | n | median `Pbar_on` | median `Qbar_pre` |
|---|---|---|---|
| eligible rows where the hinge fires (M = 10) | 28 | **−20.08** | −20.49 |
| eligible rows where it is silent | 120 | −0.63 | −27.17 |

### Where eligibility lives

| length bucket (s) | 2.86 | 3.00 | 5.71 | 6.00 | 11.43 | 12.00 | 28.57 | 30.00 |
|---|---|---|---|---|---|---|---|---|
| n | 234 | 1190 | 186 | 930 | 63 | 297 | 19 | 81 |
| eligible ≥1 s | 0.4% | 1.3% | 9.7% | 8.2% | 9.5% | 8.4% | 5.3% | 7.4% |

| row type | n | eligible ≥1 s | median contrast | median hinge M10 |
|---|---|---|---|---|
| synthetic | 2058 | **1.1%** | +25.88 | 0.00 |
| real-far | 499 | 10.0% | +25.14 | 0.00 |
| real-near | 443 | 17.2% | +21.92 | 0.00 |
| turn-taking (cross-cuts the three above) | 369 | **32.2%** | +23.49 | 0.00 |

Cross-tabulated, the shape exists *only* inside the turn-taking rows: real-near **0/214** without
turn-taking vs **76/229 = 33.2%** with it; real-far 1.9% vs 30.7%. Row type is read off emitted
scalars (`RowPlan.use_realfar/use_realnear` are not emitted) — see `preflight_common.row_type`.

### Which clause binds

| clause (each on its own, over all 3000 rows) | share |
|---|---|
| `n_interferers >= 1` | 79.87% |
| user's first active frame ≥ 100 frames | **6.83%** |
| ≥ 50 user-active frames from the onset | 95.03% |
| ≥ 100 frames of interferer speech before the onset | 6.20% |
| ≥ 50 frames of interferer speech before the onset (fallback) | 8.47% |
| any interferer speech before the onset | 18.03% |
| frame set `B` non-empty | 25.00% |

## P1 (amended) — does the score measure the guard's fix?

The design asked for 450 Dawn utterances. The AMENDMENT withdraws every Dawn *waveform* number
(`speech` is not waveform-consistent with `mix`: median |corr| 0.30 at the best lag, polarity flips,
lags into the thousands of samples), and `P`/`Q` **are** foreground projections, so Dawn cannot carry
this test. It is run instead on the construction the failure was measured on: 400 synthetic
wrong-anchor rows = the 200 utterances of `data_report/wer_set_moderate_test`, each with an `other2`
and an `other3` prefix (2 / 3 s of a different speaker's ref-active mix), built by importing
`v19c_diagnostics/anchor_synthetic/anchor_synthetic_probe.py`'s own helpers. Eligible by
construction: 400/400 in both arms. Table: `results/p1_TABLE.txt`.

| median over 400 paired rows | (a) ungated v16 ep19 | (b) through `OnsetGuard()` defaults | paired Δ | Wilcoxon p | down/up |
|---|---|---|---|---|---|
| `Pbar_on` | −7.10 | −2.25 | **+4.62** | 2.7e-67 | 0/400 |
| `Qbar_pre` | −7.92 | −0.36 | **+7.43** | 2.7e-67 | 0/400 |
| contrast | +0.43 | −1.69 | **−2.50** | 0.0029 | 244/156 |
| `L_inherit`, M = 10 | 9.566 | **11.691** | +22.2% (needed −30%) | 0.0101 | 156/244 |
| `L_inherit`, M = 6 | 5.566 | **7.691** | +38.2% | 0.145 | 156/244 |

By prefix length, M = 10: 2 s 9.52 → 11.39; 3 s 9.63 → 11.98. What the guard did (gain 1.0 = dry,
model not consulted): median mean gain over `B` 0.905 with the median row 100% dry there, median mean
gain over the onset window 0.549.

**Read plainly.** The guard *does* recover the user's onset — `Pbar_on` improves 4.62 dB on every one
of 400 rows. It recovers the pre-onset bystander by more, because handing back the dry signal is
exactly "stop suppressing", so the score's *difference* moves the wrong way. This is not a tuning
problem: the design detaches `Qbar_pre` precisely so the model cannot buy the keep number with far
suppression, and the consequence is that the score is **blind by construction** to the only
intervention with real-recording evidence behind it (Dawn ASR deletion 0.230 → 0.123). Per the
pre-registration: say so, do not retune.

## P2 — gradient share of the new term

16 real train batches (3 s and 6 s buckets only — the two SSL encoders plus a backward do not fit
beside a 12 s or 30 s bucket at fp32 on a 23 GB card), `weighted: 0.25`, fp32. The module is in
`train()` mode because cuDNN refuses `RNN backward` outside it, with all 12 dropout / normalisation
modules pinned to `eval()` so every pass sees identical activations. One term per forward pass, graph
released each time. Table: `results/p2_p3_TABLE.txt`.

11/16 batches had ≥1 eligible row; **6/16 had `L > 0`** (an eligible row whose hinge is already
satisfied contributes an exact zero and no gradient at all, so that is the stratum where the term
exists).

| share of ‖g_total‖ | value |
|---|---|
| the six `L > 0` batches, sorted | 17.72, 20.48, 50.01, 50.86, 84.74, **104.51** % |
| median on `L > 0` | **50.43%** (p25 27.86, p75 76.27, winsorised mean 54.01) |
| worst single batch | **104.51%** (>100% is possible: the two gradients partially cancel) |
| median over the 11 eligible-row batches | 17.72% |
| median over all 16 batches | 0.00% |
| median share of the *summed* per-term norms, `L > 0` | 29.74% (max 85.20%) |

Per-term gradient norms, median over the 16 batches:

| term | weight | median ‖g‖ | median weighted value |
|---|---|---|---|
| SDRLoss | 1.0 | 1072 | −10.23 |
| MultiResolutionSTFTLoss | 0.5 | 64.95 | 0.344 |
| OverSuppressionLoss | 3.0 | 12.28 | 0.040 |
| ResidualReferenceLoss | 0.2 | 0.328 | 0.0027 |
| ASRFeatureLoss (HuBERT) | 1.0 | 69.89 | 0.331 |
| ASRFeatureLoss (WavLM) | 1.0 | 69.89 | 0.331 |
| DistHeadRegressionLoss | 0.3 | 4.229 | 0.0056 |
| **AnchorInheritanceLoss** | 0.25 | 0 (silent on 10/16) | 0 |
| ‖g_total‖ | | 1328 | total −5.16 … −9.07 |

On the six firing batches the weighted term is 0.295–3.538 (raw hinge 1.18–14.15 dB) against a total
loss of −5.16 to −9.07. **Kill fires** (>50%). The cause is dimensional, not a weight mistake: the
hinge is measured in dB with `max|dP/da| = 138.87`, while every other term is an O(1) signal loss, so
0.25 × a dB-valued hinge is not a small perturbation. Any successor objective on this contrast needs
its scale fixed at the definition, not at the weight.

## P3 — DDP safety

```
batches with no eligible row      : 5
  returned a graph-carrying zero  : 5
  returned None / NaN / no graph  : 0
forward+backward completed        : 16/16
exceptions                        : 0 []
```

Plus five unit cases in `test/test_losses/test_inherit_loss.py`: onset too early, no interferer, no
interferer speech before the onset, only 0.4 s of it, and a batch with no `background_vad_target` key
at all (the shape the collate emits when no row in the batch carries background speech). Each returns
`0.0` with a live `grad_fn`, and `backward()` leaves zeros on the input rather than raising. This is
the failure mode that severed a DDP job once already (the background-VAD crash).

## P4 — onset alignment

60 rows from the ≥20 s buckets of the train loader, on the 400/160 label grid. No model: the onset
index depends only on `vad_target`. Table: `results/p4_TABLE.txt`.

| | all 60 long rows | the 3 provably speed-perturbed |
|---|---|---|
| \|loss_onset − label_onset\| | max **0**, median 0, >2 in 0/60 | max **0**, >2 in 0/3 |
| \|label_onset − waveform_onset\| | max **1**, median 0, >2 in 0/60 | max 0, >2 in 0/3 |

Only 3/60 rows can be *proved* speed-perturbed: the speed draw is `{0.95, 1.00, 1.05}`, a slowed row
is cropped back to the bucket length, and 1.00 is a no-op — so only `1.05` leaves a visible trace
(30.00 s → 28.57 s). The second column is the property the gate protects: `vad_reference` is cloned
*after* the speed block in `task/ns.py`, so the label sits on the post-speed grid and agrees with the
post-device-chain target waveform to within one frame. Had it been a pre-speed snapshot a ±5% draw
would put the label up to ~150 frames away on a 30 s row and the loss's 0.5 s onset window would be
measuring the wrong half second. The pytest covers the perturbed case explicitly by construction.

## Training-data temporal shape (`check_training_data.py --split train`)

600 train rows, seed 7. Reproduces the diagnosis's own audit (`v19c_diagnostics/training_data_audit`,
1206 rows) within sampling noise, from the shipped instrument rather than a scratchpad script. Table:
`results/train_temporal_TABLE.txt`, JSON: `results/check_training_data_train.json`.

| | this run (600 rows) | audit (1206 rows) |
|---|---|---|
| target onset median | 0.00 s | 0.00 s |
| onset within 0.5 s | 93.0% (558/600) | 90.7% |
| onset ≥ 1.0 s | 4.5% (27/600) | — |
| any interferer before onset | 17.7% (106/600) | 15.8% |
| interferer before onset ≥ 0.5 s | 7.0% (42/600) | — |
| interferer before onset ≥ 1 s | 5.3% (32/600) | 6.0% |
| interferer before onset ≥ 2 s | 4.5% (27/600) | 4.6% |
| longest target-free gap median | 0.56 s | 0.53 s |
| gap ≥ 5 s | 0.8% (5/600) | 0.3% |
| re-entry after ≥ 5 s | 0.2% (1/600) | 0.3% |
| **silent target but labelled `target_present`** | **1.8% (11/600)** | 1.0% |

The last row is R0.4's label bug, independently reproduced: 11 rows whose gated target is exactly
zero still carry `target_present = 1`, so the waveform losses treat them as lone-far while
`ResidualReferenceLoss(target_present_only)` counts them present. 7 more rows are silent *and*
labelled absent (correct), and 0 rows are non-silent but labelled absent.

`--split valid` (the default) is unchanged: the printed output is byte-identical to `HEAD`'s over all
139 lines, with the new `TEMPORAL SHAPE` block appended, and `report.json` keeps `manifests` and
`separability` exactly as they were while gaining `split` and `temporal`.

## Checkpoint / recipe pre-flight

Zero new parameters, as pre-registered:

```
$ cd egs/voice_isolate && uv run python scripts/preflight_ckpt_recipe.py \
    --ckpt /work/any_exp_link/puresound_exp/dpcrn_v16_lengthmix/lightning_logs/version_0/checkpoints/epoch=19-step=10000.ckpt \
    config/exp/train_dpcrn_v19c_inherit.yaml config/exp/train_dpcrn_v16_lengthmix.yaml
  train_dpcrn_v19c_inherit.yaml              ok
  train_dpcrn_v16_lengthmix.yaml             ok
```

Explicit counts against the same checkpoint: **0 missing, 0 dropped backbone weights**, 801 098
parameters. The 4 "unexpected" checkpoint keys are loss-function STFT window buffers
(`loss_func_list.1.stft_losses.*.window`, `loss_func_list.2.hann_window`), which
`preflight_ckpt_recipe.py` ignores by design because `init_siso_model` builds no loss list — v16's own
recipe reports the same four.

## Exact commands

```
cd /home/milowu/A4Audio/PureSound/egs/voice_isolate

# P0  (~5 min for 700 rows, ~20 min for 3000, cuda:0)
uv run python benchmarks/probes/v19c_preflight/p0_eligibility.py \
    --n-rows 3000 --max-batches 2000 --num-workers 8 --seed 7 --out <dir>

# P1  (~30 s on 400 rows, cuda:0)
uv run python benchmarks/probes/v19c_preflight/p1_guard_sensitivity.py --out <dir>

# P2 + P3  (~5 min, cuda:0; set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python \
    benchmarks/probes/v19c_preflight/p2_p3_gradient_ddp.py \
    --n-batches 16 --max-batches 120 --num-workers 6 --out <dir>
# add --firing-only for the conditional distribution (skips the exact-zero batches;
# P3's no-eligible-row case then needs a run without the flag)

# P4  (CPU only)
CUDA_VISIBLE_DEVICES= uv run python benchmarks/probes/v19c_preflight/p4_alignment.py \
    --n-rows 60 --num-workers 6 --max-batches 600 --out <dir>

# training-data temporal shape
uv run python scripts/check_training_data.py \
    config/exp/train_dpcrn_v19c_inherit.yaml \
    --split train --n 600 --dump 0 --num-workers 6 --seed 7 --out <dir>

# tests
cd /home/milowu/A4Audio/PureSound
uv run pytest test/test_losses/test_inherit_loss.py test/test_system/test_loss_dispatch.py -q
```

## Files this stage produced

| path | what |
|---|---|
| `puresound/nnet/loss/inherit.py` | `AnchorInheritanceLoss` — the objective and `row_scores`, the instrument every number above is read with |
| `puresound/nnet/loss/__init__.py` | one import + one `__all__` entry (that list is what `loss_func[].type` resolves against, and what the provider-contract test in `test/test_system/test_loss_dispatch.py` enumerates — no edit was needed there) |
| `test/test_losses/test_inherit_loss.py` | 27 cases: the floored-P envelope table, the offset invariance, the sign on both rows, DDP zeros, speed-perturbed alignment |
| `egs/voice_isolate/config/exp/train_dpcrn_v19c_inherit.yaml` | v16 verbatim + the one loss entry; `max_epochs: 20`, `T_0: 20`, `work_folder: exp/dpcrn_v19c_inherit`, `verbose: True`. **Not launched** |
| `egs/voice_isolate/scripts/check_training_data.py` | `--split {train,valid}` (default `valid`) + the temporal columns |
| `benchmarks/probes/v19c_preflight/*.py` | the four probes and their shared plumbing |
| `benchmarks/probes/v19c_preflight/results/` | the tables and JSON summaries above |

## Caveats, stated

* **No ramp is configurable.** The design asked for 0 → 0.25 over two epochs. `LossConfig`
  (`puresound/config/recipe.py`) is `{type, weighted, args}` and `reduce_losses` multiplies by a
  static float; there is no schedule hook on the loss list anywhere in the repo. The weight would be
  constant at 0.25 from step 0. Given P2 this matters: a ramp is not what fixes a dB-scaled term.
* **`on_validation_epoch_end` monitoring (design §3.6 item 7) was not implemented.**
  `puresound/system/siso.py` was being edited by another session; `verbose: True` gets the per-term
  curves (`train_step_loss_{i}` / `valid_step_loss_{i}`) but not the median-hinge / eligible-share
  monitor. `row_scores` is the call it needs.
* **Concurrent tree.** These numbers were measured while another session was landing the v20
  session-rows infrastructure. `augmentation_session_rows` is absent from the v19c config, so
  `SessionRowBuilder.active` is False and it consumes no randomness — the distribution is v16's.
  While the recipe block had landed ahead of the dataset side, every voice-isolation dataloader
  raised `TypeError: unexpected keyword argument ['session_rows_args']`;
  `preflight_common._BlockFilteringRecipe` drops **empty** blocks the dataset does not accept (and
  refuses to drop a configured one), which is a no-op now that the dataset accepts it.
* **P2's n is small** (6 firing batches) and restricted to the 3 s / 6 s buckets. The kill rests on
  the worst batch exceeding 50% and on the dimensional argument, not on the median's precision.
* **P4's perturbed subset is 3 rows** for the reason given above; the pytest covers the case by
  construction.

## What this sizes for the next round

1. **The row shape is a data problem, not a loss problem.** 6.83% of rows have a user onset ≥1 s in;
   93.0% start inside 0.5 s; the eligible population lives almost entirely inside the turn-taking
   rows (32.2% there, 1.1% in plain synthetic rows). No reweighting of the existing distribution
   reaches the behaviour, which is what `v20 §7`'s S0 was asked to establish.
2. **A contrast against `consistency_noise` is keyed to the bystander's proximity class**, not to
   deletion: −27 dB when the pre-onset material is a far interferer, −8 dB when it is a near talker.
   Any successor must either condition on that class or measure the user's onset against the *same
   utterance's* own later frames (a within-row paired delta), which is also what the AMENDMENT's
   metric policy requires.
3. **`OnsetGuard` remains the only intervention with real-recording evidence** (Dawn ASR deletion
   0.230 → 0.123) and P1 shows a detached-suppress-side contrast cannot credit it. A score that is
   meant to reproduce the guard internally has to be allowed to see the guard's mechanism, or be
   judged on ASR instead of on a projection.
4. **dB-valued hinges need their scale set at the definition.** `max|dP/da| = 138.87`; at weight 0.25
   the term reached 104.5% of the total gradient norm.
