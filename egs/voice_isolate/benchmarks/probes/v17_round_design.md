# v17 round design — cold start is a missing ROOM REFERENCE, and half the wall is the instrument

2026-08-31. Produced by a 5-way investigation + 3-lens adversarial review (11 agents, all
claims below measured, mostly on v16 ep19). Working artifacts: session scratchpad
`v17_investigation/` and `review_*/`; the decisive probe scripts worth keeping are named
inline. Supersedes the "absolute level" premise this round started from.

## 1. Causally REFUTED this round (do not revisit without new evidence)

* **Absolute level as a cue.** DPCRN's first op is InstantLN (per-frame LN over ch×freq,
  `norm.py:53`) and the complex mask multiplies the un-normalised STFT -> the network is
  exactly scale-equivariant: f(a·x)=a·f(x). Measured: ±32 dB gain on all 20 field cold-far
  clips moves suppression ≤0.06 dB; same on v8. Corollaries: level-swept guards are vacuous;
  data-side re-levelling is a no-op as a cue and only re-weights scale-dependent losses
  (in the deletion direction: OverSuppression/ResidualReference scale as a¹).
* **Speech-to-floor ratio (s2f) as the cue.** Removing noise from training lone-far rows
  DEEPENS the kill (−59.8 -> −68.0); adding white OR real DNS-5 ambience to field cold-far
  clips does not unlock (−1.34 -> −1.36/−2.32, ~0.6 dB of which is arithmetic). Note: the
  "training s2f 12 dB vs field 26 dB" numbers were partly an estimate_floor artifact
  (5th-pct frames on continuous rows measure speech pauses, not floor).
* **H-chain as written** ("unseen chain => shallow"). Three never-seen chains give
  −19 / −2 / −1 dB (DiPCo / AMI / NOTSOFAR arrays). DRR explains the split: AMI/NOTSOFAR
  table arrays measure at KEEP-class DRR (−1.4/−1.8 vs VOiCES keep −1.9) — the model treats
  them as keep because acoustically they are (corpus_pair_README already said this).
  What SURVIVES: at matched DRR, in-domain VOiCES is killed 20–27 dB deeper than any unseen
  chain — chain familiarity is a large multiplier, but not the gate.

## 2. The two load-bearing discoveries

### 2a. The instrument: single-checkpoint field cold-far readings are unreadable
Scored all 12 saved v16 checkpoints (ep8–19, same run, no knob change) on the field set:
cold-far median spans **−0.35..−3.39 dB**; n(≤−6 dB) 2..7; PARTIAL 0..4; per-clip swings up
to 19.9 dB across adjacent epochs; session-mean sd 1.95 dB; KEEP violations 2..5 depending
on epoch and dry_blend. **Every cold-far delta this project has ever read off a single
checkpoint — including "v16 ep19 −1.21 vs v8 −0.33" — sits inside this null band.**
Count-based KEEP guards are additionally blind to catastrophic deepening (ep19->ep39:
qvf_keep_in_touch_near1 −3.16 -> −13.26 with NO count change) and to chain-selective damage
(all ep19->ep39 keep damage is QVF-chain; device chain moved ≤0.12 dB).

### 2b. The mechanism: 3 s of REAL ROOM AMBIENCE unlocks cold start (no anchor needed)
`pad_type_probe.py`, v16 ep19, 14 device cold-far clips, 3 s lead-ins:

| lead-in                    | median | ≤−6 dB |
|---|---|---|
| none (current benchmark)   | −1.21  | 3/14 |
| digital silence            | −5.71  | 7/14 |
| white noise at floor level | −0.28  | 1/14 |
| **real ambience (no voice)** | **−16.11** | 5/6 |
| near-user speech (anchor)  | −20.19 | 14/14 |

Real room tone alone recovers most of the anchored performance; a synthetic floor at the
same level recovers nothing. The model self-calibrates against the room's true signature,
not against energy. This (i) confirms `selfcal`/`dist-cue` ("根治=給參照") causally at the
output, (ii) means the deployed streaming model — whose state has always heard the room —
faces a much smaller wall than the zero-context benchmark measures, and (iii) hands the
round its training knob.

## 3. The round

### Part A — rebuild the instrument first (eval only, ~1 GPU-day, no training)
1. **Cold-start protocol v2**: every reading = mean over the last 5 checkpoints × context
   conditions {0 s, 3 s ambient lead-in ×3 draws}, absolute residual + sir_out, paired
   Wilcoxon vs the control's matched block; steady-state (t≥3 s) co-reported. The 0 s
   column is "worst-case stream start"; the ambient column is "deployment steady state".
2. **Guards**: keep the KEEP count but add per-chain × per-clip severity deltas; adopt the
   s2f-swept KEEP probe (measured baseline is NOT flat: −0.25 -> −1.02 monotone, with a
   −51.76 dB cliff on qvf_keep_in_touch_near1 at s2f 8 — regressions judged against that
   curve, not against "flat"); drop the level-swept guard (vacuous, §1); add a
   context-stability spread (three same-length pads, report median shift AND spread).
3. **Re-read the standings** (v8, v16 ep19/ep39 blocks) with this instrument before any
   new training is judged; freeze as the scorecard's cold-start section.

### Part B — training knob (ONE change): row-initial ambient lead-ins
`row_initial_ambient`: with prob ~0.3, a training row OPENS with 1–4 s of its own scene's
noise/room signature before any speech (both keep and lone-far rows — class-blind by
construction, so it cannot leak labels). Teaches exactly the behaviour 2b showed the model
already half-has: calibrate on the room, then decide. Implementation sits in the overlap
gating/turn-taking machinery (a scene-solo lead-in). Warm-start v16 ep19, 20 epochs,
self-stopping + chained benchmark; judged ONLY with the Part-A instrument. Success = the
0 s column moves toward the ambient column (the model calibrates faster from less), with
Dawn deletion / KEEP severity / session block-band / s2f-swept-keep all held.

### Parked (specced, not in this round)
* **DiPCo far pool** — the one genuinely new far chain (arrays at DRR −11..−12, ~2 h solo
  audio after merged-span extraction). Full build spec from review: Wiener-deconv DRR gate
  ≤−7 dB (close-talk level ratios do NOT work — headset-gain dominated), cross-talk gate on
  other participants' close mics, windows cut to wavs (AudioIO.open has no offset), weight
  field in the pool loader, session+speaker+device quarantine to keep corpus_pair_probe
  alive (train S02/S04/S05/S09/S10 on U02–U05 only). AMI/NOTSOFAR arrays are KEEP/boundary
  material, not suppress — any future use goes on that side.
* **Loss-floor debt**: the live gradient floor on lone-far rows is the MR-STFT magnitude
  clamp (|X| pinned at 1e-4; ~39% of bins pinned at −70 dBFS output), NOT inactive_sdr's
  eps (~41 dB from binding). Fix belongs to a loss round, with the per-row renormalisation
  recipe already verified in `floors_and_fix_probe.log`.
* **Deployment note (no training needed)**: never reset streaming state on silence; prime
  new streams with buffered room audio when available. This is most of the product-side
  cold-start mitigation, available today.

---

## VERDICT (2026-09-01, ep19 block) — NEGATIVE on the pre-registered primary. Axis closed for training.

`row_initial_ambient` (prob 0.30, 1–4 s leads, verified in the data by ear and by
measured lead lengths) trained 20 epochs from v16 ep19. Judged on the v2 instrument:

| | v16 ep19 block | v16 ep39 block (+20ep control) | **v17 block** |
|---|---|---|---|
| far `none` | −1.71 | −1.52 | **−1.48** (p=0.15 / 0.18 — no change) |
| far `ambient` | −13.19 | −14.47 | **−10.85** (p=0.69 — if anything weaker) |
| near worst (block) | −8.34 | — | −11.61 |
| Dawn WER/del (single ckpt) | 0.296/0.206 | 0.320/0.215 | 0.290/0.184 |

The zero-context column did not move — now measured properly across v8, v16 (two
cycles) and a directly-targeted round. Reading: the ambient column proves the
calibrate-on-room capability exists and saturates; the `none` column offers NO
reference to calibrate on, and more examples of "reference then decide" cannot
conjure one. Zero-context cold start is treated from here as a DEPLOYMENT problem
(never reset streaming state; prime new streams with buffered room audio — worth
+11..13 dB on every model generation), and the `ambient` column is the
deployment-truth metric. The knob stays in the tree (default off), harmless.

---

## ERRATUM (2026-09-03) — §2b is withdrawn

The "real ambience (no voice)" pads of §2b were drawn from 90D 98.4–107.0 s, which holds an
unlabelled utterance at 101.4–103.9 s (−47 dBFS, verified by level profile and whisper). Sliding
a 2 s window across the gap: floor windows −0.2..−4 dB, the utterance window −22..−26 dB (both
v8 and v16). True floor is a null on 5-checkpoint blocks (Δ −0.04 / +0.03 dB, p 0.08 / 0.76).
So the model does not "self-calibrate against the room's true signature"; it re-uses a recent
talker (any talker through the same chain) as anchor, for ~1–10 s. §1's refutations stand; §2a
stands; §3's Part B (row-initial ambient lead-ins) was targeting a mechanism that does not exist,
which is consistent with its negative verdict. Full matrix: `reference_matrix_README.md`.
