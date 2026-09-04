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

## 5. What this settles

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
