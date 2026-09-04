# AnchorGate, steps 1–2: can an inference-side supervisor stop the wrong-anchor deletions?

2026-09-04. Scripts: `anchor_gate_cache.py` (step 1: per-frame cache of bottleneck features,
DistHead readouts, mix and enhanced audio for every field clip × context condition and every
Dawn utterance × {none, background}; v8 and v16 ep19) and `anchor_gate_sim.py` (step 2: offline
gate simulation on that cache, no model calls). Follows `reference_matrix_README.md` §6: the
streaming state is a recent-talker template, so the harmful failures are deletions of a near
talker who follows a *different* anchor, and cold start with no anchor is passthrough.

Rule under test, in words: **trust the model's attenuation only when the current anchor is a
confirmed near talker; otherwise, while speech is present, blend the output back to the dry mix.**
Anchor A = the model's own DistHead readout (sliding MLP over a window of speech-active
bottleneck frames) over the most recent ≥ 1 s of sustained speech, held through silence; current
C over the last W_c s; protect when C reads clearly nearer than A (relative rule) or when no
anchor exists yet (`pna`). Actuator = PresenceGate-shaped integrator (attack 0.05 s, release
τ_dn), out = g·mix + (1−g)·enh, exactly dry or exactly enh outside transitions. Operating point
chosen on the FIT set (90d / 270d / 0d / all qvf_*); **180d read once**. All numbers below were
re-run independently of the agent that wrote the script and match.

## 1. Is the anchor's distance readable from the state?

AUC of the sliding readout, keep spans (30–50 cm) vs suppress spans (2–3 m), W = 1 s:

| | v8 dist | v8 DRR | v16 dist | v16 DRR |
|---|---|---|---|---|
| device chain, isolated clips | 0.989 | **0.997** | 0.977 | 0.990 |
| device chain, streaming sessions | 0.695 | 0.757 | 0.709 | 0.781 |
| QVF chain, isolated clips | **0.265 (inverted)** | 0.799 | 0.255 | 0.674 |
| QVF chain, sessions | 0.357 | 0.517 | 0.357 | 0.441 |

Prefix (anchor) reads, v8, W = 1 s: device near_speech 0.53 m / 11.6 dB DRR, far_speech
0.87 m / 1.9 dB, the 90D utterance 0.56 m / 10.9 dB, floor 0.62 m / 3.0 dB. QVF: near_speech
reads *farther* (0.87 m) than far_speech (0.71 m).

Reading: **yes on the device chain, no on the QVF chain.** Distance is not weak on QVF, it is
reversed; DRR is the only readout that survives QVF (and is the better one on device too). Both
degrade hard in streaming sessions (0.99 → 0.70–0.78) — the deployment condition. Five clips
never arm an anchor (longest speech run < 1 s: 180d_near6, 270d_near3/5, qvf_gym_far1,
qvf_price_far1).

## 2. What the gate buys and costs (FIT set)

Keep objective: near+dt violations (< −3 dB) over near_speech / far_speech / stream / floor
prefixes + sessions (148 spans). Suppress objective: passes (≤ −6 dB) over near_speech / event /
stream + sessions (46 spans).

| arm | v8 keep viol | v8 supp pass / median | v16 keep viol | v16 supp pass / median |
|---|---|---|---|---|
| ungated (dry_blend 1.0) | 26/148 | 40/46 / −15.23 | 20/148 | 37/46 / −15.77 |
| `pna` only (no anchor ⇒ dry; comparison off) | 13/148 | 39/46 / −11.95 | 7/148 | 36/46 / −11.81 |
| DRR relative, margin 3 dB, W_c 1 s, τ_dn 2 s, pna (chosen on v8) | **6/148** | 37/46 / −11.00 | 3/148 | **27/46 / −6.80** |
| DRR relative, margin 6 dB (v16's own frontier) | 13/148 | 39/46 / −11.38 | 6/148 | 35/46 / −11.51 |
| distance relative, r 0.7, W_c 0.5 | 11/148 | 36/46 / −11.00 | 7/148 | 32/46 / −9.82 |
| distance relative, pna off | 24/148 | 38/46 / −13.58 | 19/148 | 33/46 / −12.25 |

Three readings:

1. **Half the benefit is the "no confirmed anchor ⇒ don't delete" clause alone** (26 → 13,
   20 → 7), which needs no distance readout at all and therefore does not depend on §1. Those are
   onset deletions: the model attenuates the very first second of a near talker before any anchor
   exists.
2. The relative comparison adds the other half, DRR beats distance at every frontier point, but
   **the margin is an absolute dB quantity and does not transfer across checkpoints**: v8's 3 dB
   costs v16 ten suppress passes; v16 wants 6 dB. Same failure mode as
   `presence_selfcal_README` Finding 4.
3. The price is suppression depth: −15.2 → −11.0 dB median on the fit set, and on **180d
   (held out, nothing to fix: 0/40 violations ungated) the gate is pure cost**, −19.4 → −11.6
   (v8) and −22.3 → −10.0 (v16), 1–2 suppress passes lost.

## 3. Named cases (v8, DRR gate, ungated → gated, dB)

| case | ungated | gated |
|---|---|---|
| 90d_near4 after 3 s far speech (`stream`) | −3.53 | −1.78 |
| 90d_near3 after other-recording near speech | −14.07 | −2.12 |
| qvf_keep_in_touch_near1 after its own floor lead-in | −28.80 / −29.55 | −2.17 / −0.01 |
| qvf_keep_in_touch_near1 cold | −3.64 | −0.01 |
| qvf_price_near1 cold | −6.65 | −0.09 |
| 0d_near1 sentinel cold | −4.21 | −2.95 (still borderline) |
| `silence` prefix catastrophe, dt keep median | −35.46 | −1.15 |
| cost: device far clips after near anchor | −12.6..−27.1 | −11.0..−16.0 (all still pass) |
| cost: qvf_gym_far1 after near anchor | −13.45 | **0.00** (QVF inversion = false protect) |
| cost: qvf_price_far2 after near anchor | −13.71 | −5.04 (loses pass) |

## 4. Dawn Chorus, proxy only (no ASR yet)

Deletion proxy = energy of the output relative to the mix over reference-active frames, paired
per utterance, n = 450.

| | v8 none | v8 background | v16 none | v16 background |
|---|---|---|---|---|
| proxy, enh → gated | −1.94 → −0.95 (+0.85) | −2.28 → −1.05 (+0.84) | −1.68 → −0.75 (+0.80) | −1.92 → −0.99 (+0.74) |
| SI-SDR, enh → gated (paired) | −0.25 | −0.09 | −0.39 | −0.18 |

The bystander-context penalty on the proxy (background − none, paired median) halves on v8
(−0.115 → −0.050) and narrows on v16 (−0.138 → −0.106). Whether that is the +0.03 Dawn deletion
of `reference_matrix_README` §5 coming back needs the ASR run (step 3).

## 5. What steps 1–2 settle

* **The state does carry the anchor's distance on the device chain**, readably enough for a
  supervisor (AUC 0.99 isolated, 0.7–0.8 in sessions). So layer 2 (teach the LSTM with
  long-gap / bystander-first rows) is not asking it to read something it cannot see.
* **On the QVF chain the distance readout is inverted and DRR is marginal** — the compression
  axis (`eq_probe`, v11b) has to be fixed in the representation; no supervisor reads through it.
* **The chain-agnostic half of the gate — "until a ≥ 1 s talker has been heard, do not
  delete" — is the deployable part**: it removes half the keep violations on both checkpoints
  with no readout, no margin, no checkpoint dependence, at a cost of −3..−4 dB median far
  suppression on the fit set and one lost pass. The relative-DRR half is worth its extra keep
  gain only if the margin can be made relative too (self-calibrated per session), otherwise it
  re-creates the fixed-threshold problem.
* Step 3 (only if wanted): ASR on the gated Dawn outputs for the `pna`-only arm and the DRR
  arm, both checkpoints, `none` + `background` — four large-v3 runs, ~1 h — to turn §4's proxy
  into deletion numbers; then wire `AnchorGate` beside `PresenceGate` as an opt-in flag.

## 6. Step 3 — real ASR on the gated Dawn outputs (`anchor_gate_asr.py`)

faster-whisper large-v3, n = 450, both checkpoints, gated audio built from the step-1 cache with
the step-2 gate (same functions), ungated re-transcribed in the same session (reproduces
`reference_matrix_README` §5 within 0.002). Arms: `pna` = protect-when-no-anchor only (comparison
disabled); `drr3` / `drr6` = DRR relative rule with 3 / 6 dB margin; all W_c 1 s, τ_dn 2 s.

| v8 | none WER / sub / ins / del | background WER / sub / ins / del |
|---|---|---|
| raw mix | 0.183 / 0.048 / 0.052 / 0.084 | (same) |
| ungated | 0.333 / 0.090 / 0.014 / 0.230 | 0.373 / 0.101 / 0.015 / 0.257 |
| pna | 0.232 / 0.061 / 0.049 / 0.123 | 0.267 / 0.080 / 0.040 / 0.147 |
| drr3 | **0.194** / 0.053 / 0.046 / **0.096** | **0.193** / 0.058 / 0.036 / **0.099** |
| drr6 | 0.223 / 0.055 / 0.050 / 0.118 | 0.243 / 0.071 / 0.037 / 0.136 |

| v16 | none | background |
|---|---|---|
| ungated | 0.297 / 0.077 / 0.013 / 0.207 | 0.335 / 0.085 / 0.015 / 0.235 |
| pna | 0.217 / 0.059 / 0.047 / 0.111 | 0.246 / 0.076 / 0.037 / 0.133 |
| drr3 | **0.188** / 0.051 / 0.044 / **0.094** | **0.196** / 0.057 / 0.040 / **0.099** |
| drr6 | 0.210 / 0.055 / 0.046 / 0.108 | 0.233 / 0.068 / 0.038 / 0.127 |

Paired per-utterance deletion, arm − ungated: every arm, both contexts, both checkpoints
p < 1e-26 (e.g. v8 drr3 background: 297 utterances down / 32 up / 121 unchanged). Bystander
penalty (background − none deletion): ungated +0.027 / +0.028, `pna` +0.025 / +0.022,
`drr6` +0.018 / +0.018, **`drr3` +0.003 / +0.005 (p 0.30 / 0.34, closed)**.

How often the gate is actually dry while the user speaks (v8, fraction of reference-active
frames with g > 0.5, median): `pna` 0.20 (none) / 0.00 (background); `drr3` 0.39 / 0.48;
`drr6` 0.25 / 0.00. The `pna` gain under `background` therefore comes from the 2 s release
tail leaking a partial dry blend into the first seconds of the utterance, not from a fully
open gate. In every gated arm the insertion rate returns from 0.014 to 0.036–0.050 — most of
the way back to the raw mix's 0.052: what the model stopped hearing, the gate lets back in.

### Reading, plainly

* The model's Dawn deletion (0.23 → 0.10 with `drr3`) is mostly **onset deletion and
  wrong-anchor deletion**, and an inference-side gate that refuses to attenuate until a near
  talker has been confirmed removes most of it, on both checkpoints, with no retraining.
* `drr3` reaches WER 0.19 — within 0.01 of not processing at all — by bypassing the model
  ~40–50 % of the user's speaking time; it wins on deletion and gives back most of the
  insertion reduction that is the product's purpose. `pna` keeps the model on 80–100 % of the
  time and still halves the deletion, but leaves the bystander penalty untouched.
* Neither arm is a version decision. The gate is a keep-side safety belt whose cost is
  measured in far suppression (fit set −15 → −11 dB median; held-out 180d −19 → −12) and in
  insertions. The right operating point is a product choice between "hear too much" and
  "delete the user", and both ends are now on the same table.

## 7. Productised: `OnsetGuard` (`puresound/system/onset_guard.py`), 2026-09-04

The `pna` clause is now an inference option: `SISO.forward(..., onset_guard=OnsetGuard())`,
`--onset-guard` on every eval script that takes `--presence-gate`. Operating point decided by
`onset_guard_sweep.py` on the step-1 cache: causal floor tracker (2 s running minimum, ≤ 3 dB/s
rise, +8 dB margin, 0.2 s hold, 0.1 s confirmation, no lookahead; 91–95 % frame agreement with the
oracle detector, recall 1.0, zero false arming on 174 room-tone prefixes), `t_arm` 1.0 s, release
2.0 s, `t_forget` 5.0 s (not resolvable from the data — set from the model's measured anchor
half-life). Same choice is optimal on v8 and v16. Fit set v8: keep violations 26 → 13, suppress
passes 40 → 37, median −15.2 → −11.8; held-out 180d 0 → 0 violations, 12/13 → 12/13, median
−19.3 → −11.6. Guard vs the probe's strict implementation: max |Δgain| 5.6e-16. ORT runtime
wiring is documented in the module docstring and not yet applied.
