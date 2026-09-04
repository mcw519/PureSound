# v20 design — a model that knows who the user is (self-enrolling foreground, trained in)

2026-09-05. Status: **design + theory, pre-registration of the programme**. Nothing here is trained yet.
Companion records: `reference_matrix_README.md` (anchor mechanism on real recordings),
`anchor_gate_README.md` (what an inference guard can and cannot do), `v19c_diagnosis.md` +
`v19c_diagnostics/` (five measurements: onset profile, synthetic anchor probe, training-data audit,
chain readability, objective landscape), `v19c_round_design.md` (the minimal loss-only round and the
not-to-do list), `v19c_diagnostics/snr_strat/` (metric policy: paired, SNR-stratified, Dawn ASR-only).

This document is about **training-side, capability-level change**. The OnsetGuard (`puresound/system/onset_guard.py`)
is an inference safety belt and is out of scope except as a baseline the trained model must beat *with the guard off*.

---

## 1. The theory: what the current model actually computes

Every measurement of the last three weeks fits one statement:

> **The model has no representation of "the user". It has a recency template: whatever near-sounding
> speech it heard in the last few seconds becomes the foreground, everything unlike it is attenuated,
> and the template is forgotten within about ten seconds of quiet.**

Evidence, all measured on both v8 (deploy default) and v16 ep19 (training mainline):

| Property of the template | Real recordings (field set v3, Dawn) | Synthetic (moderate WER set, paired per utterance) | Source |
|---|---|---|---|
| Forms after ~1 s of a talker, saturates at 2 s | 0.5 s ≈ 0, 1 s −13 dB, 2 s −19 dB | 0.25 s −0.5, 0.5 s −3.7, 1 s −12.6, 2 s −13.9 | `reference_matrix_README` §3; `v19c_diagnostics/anchor_synthetic` |
| Forgotten across quiet | half at 5 s, gone at 10 s | half-life 4–5 s, ~90 % gone by 20 s | same |
| Any same-chain talker will do; room tone, noise, cross-chain speech do not | floor null (Δ −0.04 / +0.03, p 0.08 / 0.76); the "ambience" that unlocked v17 was an unlabelled utterance | room-tone analogue −1.4/−0.6, bystander-only −0.1, far talker +0.9 (protective) | same |
| Transfers to the next talker: a new near talker after somebody else's anchor is attenuated | device near clips 0 → 4–6 keep violations / 17 after a 2–3 s other-talker prefix | Δon1 −1.25 / −1.71 dB, p≈1e-25, 176/24 utterances; same-talker prefix null (+0.01) | same |
| Is partly a channel signature, not only a voice | — | same talker, different room draw: −0.44 dB ≈ 30 % of the different-talker effect | `anchor_synthetic/TABLE_ident.md` |
| Without any template the model passes everything through | cold-start lone far talker −1.1 / −1.7 dB (4/20 pass) | lone interferer −6.3 dB (already suppressed: synthetic cold start is easier) | `COLDSTART_V2.md`; `anchor_synthetic` |
| The proximity cue survives to the bottleneck but the trained readout is chain-specific | device AUC 0.99, QVF 0.26 (inverted); a device-fitted probe on the same features reads QVF 0.79–0.83 | — | `anchor_gate_README` §1; `v19c_diagnostics/chain_readability` |
| Under strong interferers the user is deleted globally | Dawn ASR deletion 0.21–0.23 (ASR is the only valid Dawn instrument) | g ≈ −5 dB at −5 dB SIR, −8/−9 at −10 dB, whole utterance | `snr_strat/README.md` |

Two consequences follow and both were confirmed:

1. **Persistence and wrong-anchor deletion are one variable.** Because the template is a fading recency
   state, anything that makes it last longer also lengthens the window in which the next talker is
   deleted (`reference_matrix_README` §6.2; synthetic decay and Δon1 close on the same clock). No
   knob on this architecture can buy memory without buying deletion.
2. **An inference guard can only refuse to act.** "Do not attenuate until a talker has been heard for
   1 s" halves keep violations and Dawn deletion, at −4 dB of far suppression and insertions back near
   the raw mix (`anchor_gate_README` §6). It does not change what the model believes.

## 2. Why the current training produces exactly this

The training-data audit (`v19c_diagnostics/training_data_audit`, 1206 rows through the v16 pipeline):

| Dimension | Value | What it teaches |
|---|---|---|
| Target's first onset | median 0.00 s; 90.7 % within 0.5 s; 6.6 % at ≥ 1 s | "the user is whoever is already talking at t = 0" |
| Interferer speaks first for ≥ 1 s | 6.0 % of rows (turn-taking prob is 0.0; only real-recording overrides fire) | almost never has to decide *against* the first voice |
| Longest target-free gap inside a row | median 0.53 s, p90 1.81 s; ≥ 5 s in 0.3 % | never has to hold an identity across silence |
| Target re-entry after ≥ 5 s absence | 0.3 % (0.0 % of the 3 s bucket = half of all rows) | never has to *recognise* the user coming back |
| Same talker rendered through two chains in one row | 0 % | free to key the template on the channel |
| Realised SIR | median +0.5 dB, p25 −2.8, p5 −8.1 | the loud-bystander regime is a 5 % tail |
| Loss terms weighting time or segments | none of 7 (SDR whole-row, MR-STFT/OverSup/ResidualRef means, DistHead global pool); ASRFeatureLoss's 6 s crop covers t = 0 with probability 0.000 on 12/30 s rows | a one-second mistake costs ~0.1 dB of a row mean |
| Anything defining "user" beyond the row's `target` | nothing: no identity, no proximity target that is relative, no cross-time consistency | the cheapest solution — recency of the dominant near voice — is optimal |

With this distribution the recency template is not a bug the model has; it is the **correct minimiser**
of the objective on the data. To get a different model, the thing that is optimal has to change.

## 3. The capability to build

Operational definition, used for labels, losses and gates: **the user is the near talker who persists
across the session.** Near = highest direct-to-reverberant / shortest source distance *among the talkers
in this session* (relative, never absolute metres). Persistent = the same voice across silences and
across the user's own movement.

The trained model must, with no enrollment and the guard off:

| Req | Statement | Instrument (exists today) |
|---|---|---|
| R1 | Not attenuate the user's first second after a bystander has been talking | synthetic Δon1 (paired), field anchored near clips, Dawn paired ASR deletion in `--context background` |
| R2 | Not adopt the first voice it hears as the user when a nearer voice arrives later | same as R1 plus the bystander-first field clips |
| R3 | Keep the user after 5–20 s of quiet and keep suppressing bystanders through and after the gap | synthetic decay curve (gap-corrected), sessions in the field block |
| R4 | Key the user on voice + proximity, not on the channel draw | synthetic `ident` ladder (self / self-late / same-speaker-other-room / different-speaker) |
| R5 | Suppress a lone bystander once a user has been established, and keep the user under a louder bystander | field sessions and cold-far `anchor` column; SNR sweep g_on at −5..−10 dB; moderate-set ASR |

Explicitly **not** claimed: suppressing a lone far talker at true cold start with no user ever heard (there
is no evidence any talker-agnostic cue does this on our chain — `reference_matrix_README` §6.1), and the
QVF cross-chain wall (chain/readout problem, `eq_probe_README` erratum, `chain_readability`).

## 4. The three components

They are one design. §5 says why each fails alone.

### 4.1 Data: sessions, not clips

Replace the 3–30 s "two people already talking" row with a **session row** (30–60 s; keep the length
schedule's audio-seconds budget, i.e. fewer rows per batch — v16's header records the budget deliberately):

- **Talkers**: one user U (near: 0.3–1.0 m draw), 1–2 bystanders B (1.5–4 m draw), optional noise at
  the realism recipe's levels. SIR drawn so the tail reaches −10 dB more often than today's 5 %.
- **Turn structure** (drawn per row, all four shapes present at non-trivial rates):
  (a) U opens; (b) **B opens for 2–8 s, U enters** (the wrong-anchor case); (c) **U speaks, 5–20 s of
  true room floor, U returns** (persistence / re-entry) with B allowed to speak inside the gap;
  (d) overlap segments (double talk) as today's turn-taking machinery produces.
- **The user moves**: with probability ~0.3 the row re-renders U through a second draw of the same room
  and device chain (different RIR from the same near range, different chain draw) at a turn boundary.
  Same voice, different channel — the only way to make "channel ≠ identity" learnable.
- **Labels per frame**: `user_active[t]` (from U's dry signal, the same energy VAD grid as `vad_target`),
  `bystander_active[t]`, plus per-turn talker ids and the proximity scalars the pipeline already emits
  (`fg_dist`, `itf_dist`, `fg_drr`).
- **Never**: digital silence as filler (measured suppress bias −5.9/−19.2 dB), target-absent labelling of a
  row that merely has a long user gap (the Dawn deletion explosion when lone-far rate rose), relaxing any
  suppression floor (v18).

This is not "raise `turn_taking_prob`" (v10's T2, which raised turn-taking keep violations 6 → 10): that
changed dosage inside the old objective, which can only respond by suppressing more. Here the new shapes
arrive together with objectives (4.2) that say *what* to keep.

### 4.2 Objective: identity and proximity as relative, cross-time targets

Two loss terms on top of the unchanged separation losses (no floor, no `inactive_mode`, no
`OverSuppression` reweighting is touched — v18 stands):

**(i) Foreground identity consistency** — on pooled bottleneck vectors. Let `z_t` be the bottleneck
pooled over frequency (128-d, 100 fps; the vector `anchor_gate_sim` already reads). Define turn-level
embeddings `e_k = LN(mean_{t∈turn k} z_t)` for each labelled turn (user turns and bystander turns).
Supervised contrastive loss inside the row:

```
pos(k) = user turns other than k (including the user's re-rendered turns after the move)
neg(k) = bystander turns in the same row
L_id = mean over user turns k of  -log  Σ_pos exp(s(e_k,e_p)/τ) / (Σ_pos exp(s(e_k,e_p)/τ) + Σ_neg exp(s(e_k,e_n)/τ))
```

with `s` = cosine, `τ` = 0.1. Because positives include the same user through a different chain draw and
negatives include bystanders through the *same* chain draw, the cheapest solution is no longer the
channel. Rows without a bystander turn or without a second user turn contribute zero with a graph-carrying
zero (the `dist.py:70–71` DDP idiom).

**(ii) Relative proximity** — replace the absolute-metres regression of `DistHead` with a within-row
ranking on the same pooled window: for each (user turn u, bystander turn b) pair,
`L_prox = softplus(m − (p(e_u) − p(e_b)))`, `p` a linear readout on `LN(z)`, `m` = 1 (unitless: the head only
has to order the talkers of *this* row). Chain-consistency view: the ordering must agree between the two
chain draws of the same row (`|Δ(p_u − p_b)|` penalty). The metres regression stays as an auxiliary
scalar only if the `dist_head` gates need it; it is not what the memory reads.

Both terms live in `puresound/nnet/loss/` as new classes fed from `batch` (turn ids, `user_active`,
`bystander_active`) through the existing provider table; they do not modify the separation terms.

**Why relative.** Absolute thresholds and absolute-metre readouts died three times (chain offset,
checkpoint drift, QVF inversion) and self-calibrated / relative judgements survived each time
(`presence_selfcal_README` findings 1 and 4; `anchor_gate_README` §2). Within-row contrast is the
training-time form of the same rule.

### 4.3 Architecture: an explicit foreground memory with a proximity-gated write

A slow state alongside the inter-time LSTM — the state the model currently fakes with recency:

```
inputs per frame t:  z_t (128-d pooled bottleneck),  w_t = write gate in [0,1]
memory:              M_t = M_{t-1} + η · w_t · (LN(z_t) − M_{t-1}),   M_0 = 0,  η ≈ 0.02 (≈ 0.5 s time constant while writing)
                     (no decay term: silence writes nothing and forgets nothing)
read:                FiLM(γ_t, β_t) = MLP(M_{t-1}) applied to the bottleneck before the decoder (the existing
                     dvec/FiLM path, `embedding_size=dvec_dim`, currently unused), γ zero-initialised, β zero-initialised
write gate:          w_t = σ(a · (p(z_t) − p̄_t) + b) · user_presence_t
```

where `p` is the relative-proximity readout of 4.2(ii), `p̄_t` a slow running mean of `p` over speech
frames in the session (so the gate compares to *this session's* talkers, not to a constant),
`user_presence_t` the per-frame near-presence logit (the v11 head, retrained here on `user_active`), and
`a, b` learned. Two properties are enforced by construction:

- **The write is proximity-gated, not recency-gated.** A bystander who opens the session scores below the
  session mean once the user has spoken and is written out; before the user has spoken the gate has
  nothing to compare to and writes weakly. This is the mechanism that separates persistence from
  wrong-anchor deletion — the thing the recency template cannot do.
- **The read starts as the identity.** Zero-initialised FiLM ⇒ warm-start from v16 ep19 is bit-identical;
  the memory earns influence only through gradient, on the session rows.

Auxiliary losses on the memory, cheap and well-defined on synthetic rows: `L_mem = ‖LN(M_t) − e_U‖²`
averaged over frames after the user's first turn (the memory should converge to the user's turn
embedding and stay there through gaps and through the user's move), and a *negative* term keeping
`M_t` away from bystander embeddings.

Streaming export: `M` is one more state tensor of size 128 plus the running mean scalar; the FiLM MLP is
a few thousand parameters; CPU cost is negligible next to the LSTMs. It is exportable with the same
manifest machinery as the head EMA states.

## 5. Why each component fails alone (and what the record says)

| Alone | What happens | Recorded instance |
|---|---|---|
| 4.1 session rows only | The old objective can only answer "more bystander-first rows" by suppressing more → keep violations rise | v10 T2: turn-taking keep 6 → 10, moderate WER +0.024 |
| 4.2 objectives only | Identity is learned but the mask has no slow state to read it from; the LSTM still forgets in ~5 s | v11 presence heads: 0.95 in-domain, never coupled; every actuator that used them failed or was inert |
| 4.3 memory only (recency-written) | Longer memory = longer deletion window for the next talker | `reference_matrix_README` §6.2, confirmed synthetically; the FGMEM verdict |
| 4.3 with an absolute write threshold | Dies on chain offset and checkpoint drift | `presence_selfcal_README` finding 4; `anchor_gate_README` §2 |
| Any of them from scratch | The curriculum ladder is load-bearing | last-stage-only training −1.65 dB vs v8 −9.83 |

## 6. Instruments and metric policy

Training-time (minutes, synthetic, paired within utterance so the noise term cancels — the policy set in
`snr_strat/README.md`): `anchor_synthetic_probe.py` Δon1 / decay / `ident` ladder; readability AUC
(`anchor_gate_sim.py readability`, device and QVF, clip and session scope); the SNR sweep
(`onset_snr_sweep.py`) for global deletion at −10..−5 dB.

Gates (5-checkpoint blocks, guard **off**, judged against v16 ep19 *and* v16 + `pna` guard):

| # | Metric | Instrument | Pass | Kill |
|---|---|---|---|---|
| P1 | Synthetic Δon1 other2/other3 | `anchor_synthetic_probe.py`, n = 200 | ≥ −0.4 dB, self3 within ±0.1 | worse than −1.0 |
| P2 | Gap-corrected decay at 10 s / 20 s (lone bystander after a 3 s user anchor) | same | ≤ −6 / ≤ −4 dB (v16: −2.0 / −0.7) **with** P1 held | no gain, or P1 fails |
| P3 | `ident` ladder: same-speaker-other-room ≥ 70 % of self | same | met | same-speaker-other-room below 50 % of self |
| P4 | Dawn paired ASR deletion, `--context none` and `background`, large-v3 | `eval_dawn_chorus.py --context` | `background` ≤ 0.18 and gap to `none` ≤ 0.01; unguarded-vs-guarded gap shrinks ≥ 1/2 | deletion > v16's 0.207 / 0.235 |
| P5 | Field keep violations, `anchor_gate_sim` fit split, guard off | `anchor_gate_cache.py` + `sweep` | ≤ 7 / 148 (the guarded arm's number) | > 13 |
| G1 | Field sessions suppression block | `eval_field_block.py`, paired CI | within 1.5 dB of −9.73 | shallower than −7.5 |
| G2 | Cold-far `none` / `anchor` (renamed from `ambient`) blocks | `eval_coldstart_v2.py` | not shallower by > 1.5 dB | > 2 dB |
| G3 | Moderate-set WER delta (Azure), insertions | `eval_wer.py`, CI printed | CI upper ≤ +0.01; insertions ≤ 0.02 | CI lower > +0.01 |
| G4 | Extreme-reverb WER, synthetic far-only, s2f keep curve | `run_full_benchmark.sh` stages 2/4/8, `eval_s2f_keep_probe.py` | no regression outside CI / band | any |
| G5 | Streaming export parity + RTF | `streaming_onnx.py verify`, `benchmark` | bit-parity within 1e-6; RTF within budget | either |

No cross-set absolute dB. Dawn SI-SDR is never quoted. BUT-OFFICE is a monitor only.

## 7. Staged programme

| Stage | Content | Cost | Kill / fork |
|---|---|---|---|
| **P0** prerequisite probe (no training) | On the paired corpora (DiPCo / AMI / NOTSOFAR sessions: two talkers per session, same talker across channels) and on synthetic same-talker-two-chain pairs, measure whether the frozen v16 bottleneck separates talker identity from channel: LOGO logistic/cosine AUC for same-talker-other-chain vs different-talker-same-chain | 1 day, CPU + one GPU pass | If identity is not separable at all (AUC < 0.6), 4.2(i) is the first thing to train and 4.3 waits; if separable (≥ 0.75), R1 and R2 can be merged |
| **R1** data + objectives, architecture unchanged | Session-row generator (4.1) + `L_id`, `L_prox` (4.2) at small weights (0.1–0.25 each, ramped), warm-start v16 ep19, 20 epochs | ~1.5 GPU-days (+ ~10 % for the second chain view), 5 engineer-days | P1/P3 and readability must move; G1–G3 must hold. If P1 moves on synthetic and Dawn `background` deletion does not: real-chain wall → fork to real session rows (R2B in `v19c_round_design.md` §5) before R2 |
| **R2** memory + FiLM | 4.3 on top of R1's checkpoint, zero-init, write gate reading R1's proximity head, `L_mem` | ~1.5 GPU-days, 7 engineer-days (module, scan, export ports, tests) | P2 (persistence) is the new requirement; P1 must not regress (the coupling test) |
| **R3** consolidation | Re-run the affected ladder stages if R2 changed the operating point; full nine-gate + ASR blocks; release decision | 2–3 GPU-days | deploy only if it beats v16 + guard on keep at ≥ equal far suppression |

Every stage: one axis, 5-checkpoint block, paired tests, pre-registered thresholds written to the round's
design file **before** launch. A stage that only matches the shipped guard at the guard's suppression
price produced nothing (the v14 rule).

## 8. Risks named in advance

1. **Real-chain transfer.** Every synthetic-only success so far stopped at the recording chain (RIR axis,
   gate-only adaptation, v17). Identity and relative proximity are *less* chain-bound than absolute
   distance, and the session objective is exactly what real paired corpora can also supply — the fork in
   R1 is planned, not hoped away.
2. **Write-gate failure on real recordings.** The v11 presence head inverted on the QVF chain. The gate
   compares within the session (relative) and is trained with chain views; but if `user_presence` is
   unreadable on a chain, the memory must default to *not writing* (passthrough), never to writing the
   loudest voice.
3. **Budget confound.** Session rows are long; keep audio-seconds per batch fixed and report rows/batch,
   or the round re-imports v15's shortfall.
4. **Deletion explosion.** Long user gaps must never be labelled target-absent; the separation losses see
   `target` as before, only the new terms see turn structure.
5. **Coupling regression.** R2 is judged on P1 *and* P2 together; a memory that improves persistence while
   P1 regresses is the recency template with a longer time constant and is killed.

## 9. Relation to v19c

`v19c` (`AnchorInheritanceLoss`, one hinge on the onset window of bystander-first rows) is a strict subset
of 4.2 on today's data: it penalises inherited deletion without giving the model any state to prevent it.
It remains the cheapest experiment that tells whether the *loss side* alone moves Δon1 on this
distribution, and its pre-flight P0 (does the hinge fire on the rows the training set contains) is still
worth running first — it costs 2 GPU-hours and its answer sizes how much of R1 must come from data.
