# v20 design (rev 2) — a model that knows who the user is

2026-09-05, revised after the Codex review of rev 1 (disposition: AMEND before implementation — accepted).
Status: **design + theory + pre-registration of the programme**. Nothing here is trained. Rev 1's inline
review comments are resolved in §0 and folded into the text; rev 1 is in git history (`e3fd1a1`).

Companion records: `reference_matrix_README.md`, `anchor_gate_README.md`, `v19c_diagnosis.md` +
`v19c_diagnostics/` (incl. `snr_strat/` metric policy), `v19c_round_design.md` (minimal loss-only round,
not-to-do list). Scope: **training-side, capability-level change**. `OnsetGuard` is a baseline the trained
model must beat with the guard off, and — see §3.3 — its commit logic is what the model has to internalise.

---

## 0. Review log (Codex, 2026-09-04) and disposition

| # | Review point | Disposition | Where |
|---|---|---|---|
| 1 | Amend before implementation; run v19c pre-flight and an amended P0/R1 first | **Accepted** | §7 order |
| 2 | Claims overstated: "any same-chain talker" → "near-like same-chain speech"; "correct minimiser" and "no knob on this architecture can…" are hypotheses, not impossibility results | **Accepted**; verified: far same-chain speech is weak/protective (real −2.75/−1.09 dB; synthetic +0.85/+1.65) | §1 |
| 3 | Causal identifiability: candidate vs committed user, no-read-before-commit, and a rule for when proximity ordering changes after the user moves | **Accepted**; product contract written | §3.3 |
| 4 | `L_id` positives are all near, negatives all far → a near/far scalar satisfies both losses; need speaker-disjoint counterfactual pairs, exclude overlap frames, stop-gradient targets, speaker-disjoint validation with EER/pair AUC | **Accepted** | §4.2 |
| 5 | Session rows cut independent scenes per update (v16: 10.5 rows/batch, 19 GiB peaks); pre-register sessions/speakers per update; prefer state-carrying truncated segments; memory smoke test | **Accepted** | §4.1, §7 |
| 6 | Render semantics: the device chain is applied once per row jointly to noisy and target (`ns.py:589`); a per-talker chain draw breaks mixture/target consistency; use emitted keys `foreground_distance`, `foreground_drr`, `nearest_interferer_distance` | **Accepted**; the "user moves" is an RIR change only; cross-chain identity comes from cross-row pairs | §4.1 |
| 7 | No differentiable bottleneck provider: `last_bottleneck` is off by default and detached (`dpcrn.py:297`); provider table has no bottleneck entry; `DistHead` is utterance-global | **Accepted**; new graph-carrying provider + per-frame heads are R1a deliverables | §4.2, §7 |
| 8 | Fatal first-talker write: with one talker `p≈p̄`, `w≈σ(b)`, a 2–8 s bystander opening fills the slot; separate candidate/committed slots, hard no-read-before-commit | **Accepted** | §3.3, §4.3 |
| 9 | The existing FiLM is utterance-level, pre-inter-RNN, `scale*x+bias` (zero ≠ identity), not invoked in the streaming path; implement as a new zero-init causal residual branch after block 2 and prove step-0 identity | **Accepted**; verified in `dpcrn.py` / `streaming/dpcrn.py` | §4.3 |
| 10 | Export scope understated: positional state packing, only VAD head triples registered; new states need ports, `state_from_tensors` slices, warm-up hold, parity/long-stream/RTF tests | **Accepted** | §4.3, G5 |
| 11 | Gates: P3 not computable (`ident` self ≈ 0); P5 is a single-checkpoint cache, not a block; G4/G5 need numeric bands; use the strongest baselines (v8 sessions −11.78, v16+guard 7/148 & 0.111) at matched far suppression | **Accepted**; verified `TABLE_ident.md` self1/self3 = +0.02/+0.01 | §6 |
| 12 | Revised order: v19c pre-flight → P0 (speaker-disjoint, balanced) → R1a (heads only, output bit-identical) → R1b (memory trained without read) → R2 (read) | **Accepted** | §7 |
| 13 | (own re-examination, not in the review) Rev 1 under-weighted a *chain-robust proximity readout* as a capability in its own right; the "absolute died twice" lesson is about fixed thresholds on drifting readouts, not about a readout trained to be chain-invariant — and the features already carry the cue on QVF (device-fit probe 0.79–0.83) | **Added** as capability A | §3.1, §4.2(ii) |

---

## 1. Working hypothesis: what the current model computes

> **H1.** The model carries no representation of "the user". Its foreground decision is driven by a
> **recency state**: near-like speech through the current chain in the last few seconds becomes the
> foreground; unlike speech is attenuated; the state fades within ~5–10 s of quiet. **H2.** Under the
> current training distribution and losses this state is a (not necessarily unique) low-cost solution;
> nothing in the objective rewards a persistent, selectively updated identity.

H1/H2 are falsifiable hypotheses about behaviour, not theorems about the architecture: the record rules
out "a longer recency time constant" as a fix; it does not show the inter-LSTM could never learn a
selectively written state under different supervision (that is precisely R1b's question).

Evidence, on v8 and v16 ep19:

| Behaviour | Real recordings | Synthetic (paired per utterance) | Source |
|---|---|---|---|
| Forms after ~1 s of near-like speech, saturates at 2 s | 0.5 s ≈ 0, 1 s −13, 2 s −19 dB | 0.25 s −0.5, 0.5 s −3.7, 1 s −12.6, 2 s −13.9 | `reference_matrix_README` §3; `anchor_synthetic` |
| Fades across quiet | half at 5 s, gone at 10 s | half-life 4–5 s | same |
| Near-like same-chain speech arms it; **far** same-chain speech is weak or protective; room tone, noise, cross-chain speech do nothing | far speech −2.75 / −1.09; floor null (p 0.08 / 0.76) | far talker +0.85 / +1.65; room-tone analogue −1.4 / −0.6 | same |
| Transfers to the next talker | device near clips 0 → 4–6 violations / 17 after a 2–3 s other-talker prefix | Δon1 −1.25 / −1.71 dB, p≈1e-25; same-talker prefix null | same |
| Partly a channel signature | — | same talker, different room: −0.44 dB ≈ 30 % of the different-talker effect | `TABLE_ident.md` |
| No state ⇒ passthrough | cold lone far talker −1.1 / −1.7 dB | lone interferer −6.3 dB (synthetic cold start is easier) | `COLDSTART_V2.md` |
| Proximity cue present in features, chain-specific in the readout | device AUC 0.99, QVF 0.26; device-fitted probe reads QVF 0.79–0.83 | — | `chain_readability` |
| Global deletion under loud bystanders | Dawn ASR deletion 0.21–0.23 | g ≈ −5 dB at −5 dB SIR, −8/−9 at −10 | `snr_strat` |

Two corollaries, both measured: persistence and wrong-anchor deletion move on the same clock (any change
that lengthens the state lengthens the next talker's deletion window); and an inference guard can only
refuse to act (halves violations and deletion at −4 dB far suppression and raw-mix-level insertions).

## 2. Why the training distribution makes this cheap

`v19c_diagnostics/training_data_audit`, 1206 rows through the v16 pipeline: target onset median 0.00 s
(90.7 % within 0.5 s); interferer speaks first ≥ 1 s in 6.0 % of rows; longest target-free gap median
0.53 s, ≥ 5 s in 0.3 %; re-entry after ≥ 5 s in 0.3 %; no row renders the same talker twice; realised SIR
median +0.5 dB (p5 −8.1); none of the seven losses weights time; nothing defines "user" beyond the row's
`target`. Under this distribution "whoever is already talking, keyed on recent near-like sound" is a
low-cost solution. To change the model, change what is cheap.

## 3. Capabilities and the product contract

### 3.1 Two capabilities, not one

- **A. Chain-robust proximity.** A per-frame readout of "how near is the current talker relative to this
  session's talkers", stable across capture chains (device, QVF-like production, codec). This is what a
  cold-start decision on a lone far talker would have to rest on — no identity exists yet — and what the
  commit rule (3.3) reads. Evidence it is trainable: the cue is in the bottleneck on QVF (0.79–0.83) while
  the metres-regression head inverts it (0.26). Evidence that *absolute* forms fail: three deaths of fixed
  thresholds/metres readouts. So: relative within the session, trained with cross-chain views, and never
  acted on through a fixed dB margin.
- **B. Persistent identity.** Once a user is committed, keep them through silence and their own movement,
  do not hand the foreground to the next voice, and keep suppressing bystanders through and after gaps.

A without B is QVF-like proximity focus with the wrong-anchor problem intact; B without A commits the
first voice it hears. Rev 1 folded A into B; rev 2 keeps them separate because they have different
instruments, different failure modes, and A can ship earlier.

### 3.2 Requirements (guard off, no enrollment)

| Req | Statement | Instrument (exists today unless marked) |
|---|---|---|
| R1 | Do not attenuate the user's first second after a bystander has been talking | synthetic Δon1 (paired); field anchored near clips; Dawn paired ASR deletion `--context background` |
| R2 | Do not commit the first voice heard as the user when a nearer voice arrives later | same, plus a write-decision replay on the field/Dawn caches (R1b deliverable) |
| R3 | Keep a committed user across 5–20 s of quiet, keep suppressing bystanders through the gap | synthetic gap-corrected decay; field sessions |
| R4 | Key identity on voice + proximity, not on the channel draw | speaker-disjoint identity EER / pair AUC with clip-level CI (new, §6) |
| R5 | Suppress a lone bystander once a user is committed; keep the user under a louder bystander | field sessions, cold-far `anchor` column; SNR sweep g_on at −10..−5 dB; moderate-set ASR |
| R6 | Proximity readout ordering agrees across chains | readability AUC device *and* QVF, clip and session scope |

Not claimed: suppressing a lone far talker with no talker ever heard **unless** capability A reaches a
pre-registered readability on real chains (R6) — then it becomes a product decision, not a model claim.
The QVF cross-chain wall stays a chain problem.

### 3.3 The commitment contract (causal, no enrollment)

A causal system cannot know that the first voice is a bystander until contrary evidence arrives. The
model therefore keeps two slots and the separator may only *act* on one of them:

| State | Written by | May condition the mask? | Leaves the state when |
|---|---|---|---|
| **candidate** | any sustained speech run (≥ 1 s) | **No** — output is the unconditioned separator (today's behaviour); the candidate only accumulates evidence | promoted to committed, or superseded by a nearer candidate |
| **committed** | a candidate whose relative proximity `p − p̄_session` has stayed above the session's running statistics for ≥ `T_commit` (1–2 s of *its own* speech) with presence confirmed | **Yes** | demoted only by the replacement rule below |

Rules, all future-independent:
- **No read before commit.** Until a committed slot exists, the memory read is exactly zero. This is the
  training-time form of the shipped guard's "do not delete before a talker is confirmed", and it protects
  the later user's first second by construction rather than by a loss.
- **Replacement (what wins when proximity and persistence disagree).** The committed user persists. A new
  voice replaces them only if it reads nearer than the committed user's stored proximity by a margin that is
  itself session-relative (a quantile of the session's own `p` spread, never a dB constant), sustains that
  for ≥ `T_replace` (3 s), **and** the committed user has been silent for ≥ `T_replace`. During double talk
  both are kept (the turn-taking KEEP guard). The user moving away (same voice, lower proximity) does *not*
  demote them: identity similarity to the committed embedding blocks replacement by a nearer *stranger*
  unless the stranger also satisfies the silence condition. This is a product decision and is recorded
  here so it can be argued with, not discovered in a scorecard.
- **Failure default.** If presence or proximity is unreadable on a chain (the v11 QVF inversion case), the
  gate writes nothing and the model is the unconditioned separator; it never writes the loudest voice.

## 4. The three components (revised)

### 4.1 Data: sessions, with correct render semantics and a fixed diversity budget

- **Session**: 30–60 s of material, **trained as state-carrying truncated segments** (e.g. 4 × 10 s or
  2 × 15 s with the recurrent and memory states carried across segments inside the row), so the number of
  *independent sessions and speakers per optimiser update* stays at v16's level (10.5 rows/batch is the
  pre-registered budget, not audio-seconds alone). A 30 s / 60 s peak-memory smoke test (19.0–19.2 GiB
  today on the 6/12 s buckets, 23 GB cards) precedes costing R1a.
- **Talkers**: user U (near draw 0.3–1.0 m), 1–2 bystanders B (1.5–4 m), noise per the realism recipe; SIR
  tail reaching −10 dB more often than today's 5 %.
- **Turn shapes** (all four at non-trivial rates): U opens; **B opens 2–8 s, U enters**; **U speaks, 5–20 s
  of true room floor (B allowed inside the gap), U returns**; overlap segments from the existing turn-taking
  machinery.
- **The user moves = an RIR change only.** One device-chain draw per session, applied jointly to `noisy`
  and `target` exactly as `ns.py:589` does today. With p ≈ 0.3 U's later turns use a second near-range RIR of
  the same room. Cross-*chain* identity pairs come from **cross-row** pairs (same speaker id in another row
  of the batch / an embedding queue), never from a per-talker chain inside one row.
- **Counterfactual rows for identity (speaker-disjoint by construction):** the same speaker serves as U in
  some rows and as B in others; bystanders are drawn to match U's distance/level/RIR class in a fraction of
  rows (so "near" alone cannot separate them); speaker ids are emitted per turn.
- **Labels per frame**: `user_active`, `bystander_active` (energy VAD on the dry signals, the `vad_target`
  grid), per-turn speaker id and role, and the pipeline's emitted scalars `foreground_distance`,
  `foreground_drr`, `nearest_interferer_distance`. Overlap frames are flagged and **excluded from turn
  pooling**.
- **Never**: digital silence as filler (−5.9 / −19.2 dB suppress bias); labelling a long user gap as
  target-absent (deletion explosion on record); relaxing suppression floors (v18); changing
  `trainer.length_schedule` in the same round (v15 confound).

### 4.2 Objectives: identity and proximity as relative, cross-time, speaker-disjoint targets

Infrastructure (R1a): a **graph-carrying bottleneck provider** (`"bottleneck"` in `siso._loss_providers`,
materialised only when a configured loss requests it), a per-frame **identity head** `g(z_t) → 64-d`
(small causal conv + LN) and a per-frame **proximity head** `p(z_t) → scalar` on `LN(z_t)`, both training
heads in the v11 pattern (inference output bit-identical until R2). Provider-contract and alignment tests
accompany them.

**(i) Identity.** Turn embeddings `e_k = mean_{t∈turn k, non-overlap} g(z_t)`, L2-normalised. Targets come
from a **stop-gradient EMA teacher** of `g` (no collapse by co-adaptation). Supervised contrastive loss
whose positives and negatives are chosen to break the near/far shortcut:

```
pos(k): other turns of the same speaker id -- in this row (incl. after the RIR move) AND in other rows of
        the batch / queue where that speaker was rendered through a different chain and possibly as B
neg(k): other speakers -- including bystanders matched to U's distance class, and U-role turns of other
        rows at U's own distance
L_id = mean_k  -log  Σ_pos exp(cos(e_k, ē_p)/τ) / Σ_{pos∪neg} exp(cos(e_k, ē_·)/τ),   τ = 0.1
```

Validation is **speaker-disjoint**: EER and pair AUC of same-vs-different speaker at matched proximity, with
clip/session-level bootstrap CI. Frame-pooled AUC alone is not accepted (item 4).

**(ii) Relative proximity.** Within a row, for (user turn u, bystander turn b): `softplus(m − (p̄_u − p̄_b))`,
`m` = 1 in the head's own units; **cross-chain consistency**: the same source material rendered in two rows
through different chains must give the same ordering and a bounded `|Δ(p̄_u − p̄_b)|`. The utterance-level
metres regression (`DistHead`) is kept only as an auxiliary; nothing downstream reads metres.

Neither term touches a separation loss; both zero-contribute with a graph-carrying zero when a row lacks
the needed turns (the `dist.py:70–71` idiom, DDP-safe).

### 4.3 Architecture: candidate/committed memory, read through a new zero-init residual branch

```
per frame t (all causal):
  z_t         pooled bottleneck (128-d);  q_t = teacher-free identity g(z_t);  p_t = proximity;  a_t = presence
  session stats: running mean/quantiles of p over speech frames (the relative reference)
  candidate:  C_t  <- EMA of q_t over the current sustained speech run (>= 1 s), with its proximity summary
  commit:     when C's proximity summary exceeds the session reference by the session-relative margin for
              T_commit of its own speech and presence is confirmed  ->  M <- C
  replace:    per §3.3 (nearer by session-relative margin for T_replace AND committed silent for T_replace)
  read:       x'_2 = x_2 + F(x_2, M)  after DPRNN block 2, F ends in a zero-initialised projection
              (or a learned scalar alpha initialised at 0)  ->  step-0 output bit-identical to the warm start
              M = 0 (no committed user)  ->  F(x_2, 0) is trained to be 0 (explicit penalty), i.e. no read
```

- The committed slot has **no decay**; silence writes nothing and forgets nothing. Replacement is the only
  way out.
- `L_mem` (R1b): after the user's first committed turn, `‖M − ē_U‖²` (teacher target) small; `M` far from
  bystander embeddings; **precision of "the opening bystander is never committed"** and **coverage of the
  first user second by no-read** are logged as validation metrics on synthetic sessions and replayed on the
  field/Dawn caches (the write decision is a discrete event that can be audited without any ASR).
- Streaming export (R2 deliverable, budgeted): new state ports for `M`, candidate `C`, session statistics,
  commit/replace timers, and any head caches; explicit `state_from_tensors` slices; **all decay-free states
  held during the look-ahead warm-up** so phantom frames are never integrated; tests: train-/infer-config
  checkpoint pre-flight, offline-vs-streaming numerical parity beyond warm-up (≥ 400 frames, tolerance
  stated), 5–10 min long-stream stability, ONNX state round-trip, RTF on the target CPU.

## 5. Why each component fails alone

| Alone | What happens | Recorded instance |
|---|---|---|
| Session rows only | the old objective answers "more bystander-first rows" by suppressing more | v10 T2: turn-taking keep 6 → 10, moderate WER +0.024 |
| Objectives only | identity is learned but nothing slow reads it; the inter-LSTM still fades in ~5 s | v11 heads 0.95 in-domain, never usable as an actuator |
| Memory with a recency write | longer memory = longer deletion window | `reference_matrix_README` §6.2; synthetic decay/Δon1 share a clock |
| Memory with an absolute write threshold | dies on chain offset / checkpoint drift | `presence_selfcal_README` F4; `anchor_gate_README` §2 |
| Read before commit | the opening bystander becomes the anchor with more authority than today | review item 8 |
| Any of them from scratch | the ladder is load-bearing | last stage alone −1.65 dB vs v8 −9.83 |

## 6. Instruments, baselines and gates

Metric policy (`snr_strat/README.md`): paired within-utterance deltas, SNR-stratified; no cross-set
absolute dB; Dawn is ASR-only; energy deltas labelled noise-confounded below +5 dB window SNR.

**Baselines every gate is read against** (strongest first): v8 block (sessions −11.78 dB), v16 ep19 block
(−9.73 ± 1.41), v16 + `pna` guard (7/148 keep violations single-cache, Dawn deletion 0.111 / 0.133), v16 +
`drr3` (0.094 / 0.099, at −11.0 far median), and the best free inference knob of the round. A candidate
passes on keep/deletion **at matched far suppression** or not at all.

| # | Metric | Instrument, n | Pass | Kill |
|---|---|---|---|---|
| P1 | Synthetic Δon1 other2/other3, self3 control | `anchor_synthetic_probe.py`, 200 utts, block median | ≥ −0.4 dB; self3 within ±0.1 | worse than −1.0, or self3 < −0.3 |
| P2 | Gap-corrected decay at 10 / 20 s with P1 held | same | ≤ −6 / ≤ −4 dB (v16 −2.0 / −0.7) | no gain, or P1 fails |
| P3 | Identity: speaker-disjoint EER and pair AUC at matched proximity, clip-level bootstrap CI | new probe (R1a deliverable) on synthetic held-out speakers and on DiPCo/AMI/NOTSOFAR pairs | EER ≤ 15 % synthetic, pair AUC ≥ 0.80 real, CI half-width ≤ 0.05 | EER ≥ 25 % or real AUC ≤ 0.65 |
| P4 | Dawn paired ASR deletion, `--context none` and `background`, large-v3 | `eval_dawn_chorus.py --context`, n = 450, Wilcoxon | `background` ≤ 0.18; `background − none` ≤ 0.01; unguarded-vs-guarded gap shrinks ≥ 1/2 | deletion above v16 (0.207 / 0.235) |
| P5 | Keep violations, guard off, **five caches** (block checkpoints) aggregated per span | `anchor_gate_cache.py field` × 5 + `anchor_gate_sim.py sweep --split fit`, two-proportion test | ≤ 7/148 block mean | > 13/148 |
| P6 | Write-decision audit (R1b+): opening-bystander commit rate; first-user-second no-read coverage | replay on synthetic sessions + field/Dawn caches | commit rate ≤ 2 %; coverage ≥ 95 % | commit rate ≥ 10 % or coverage ≤ 80 % |
| R6 | Proximity readability AUC, clip scope, bootstrap over clips (±0.14 today) | `anchor_gate_sim.py readability` on new head | device ≥ 0.95 kept; QVF ≥ 0.75 target, ≥ 0.50 minimal | QVF < 0.40 or device < 0.90 |
| G1 | Field sessions suppression, block, paired CI | `eval_field_block.py` | CI lower edge ≥ −11.2 dB vs v8's −11.78 (matched), never shallower than −8.2 vs v16 | median shallower than −7.5 |
| G2 | Cold-far `none` / `anchor` blocks | `eval_coldstart_v2.py` | not shallower than v16 by > 1.5 dB | > 2.0 dB |
| G3 | Moderate-set WER delta (Azure) with CI; insertions | `eval_wer.py` | CI upper ≤ +0.01; insertions ≤ 0.020 | CI lower > +0.01 |
| G4 | Extreme-reverb ΔWER; synthetic far-only; s2f keep curve | `run_full_benchmark.sh` 2/4/8; `eval_s2f_keep_probe.py` | ΔWER CI upper ≤ +0.01; far-only within 3 dB of v16; no s2f point > 0.8 dB below block, no new cliff | any exceeded |
| G5 | Streaming parity and cost | `streaming_onnx.py verify/benchmark` | max abs diff ≤ 1e-5 beyond warm-up; long-stream drift ≤ 1e-4; RTF ≤ 1.1 × v16 on the target CPU | any exceeded |

## 7. Staged programme (revised order)

| Stage | Content | Cost | Kill / fork |
|---|---|---|---|
| **S0** v19c pre-flight | does a loss-only onset hinge even fire on the rows the training set contains (`v19c_round_design.md` §3.3 P0–P4) | 2 GPU-h | sizes how much of R1a must come from data |
| **P0** prerequisite probe | speaker-disjoint, chain/proximity-balanced identity probe on the frozen v16 bottleneck: DiPCo/AMI/NOTSOFAR pairs (same talker across channels, two talkers per session) and synthetic same-talker-two-chain pairs; pair AUC with CI | 1 day, CPU + 1 GPU pass | AUC < 0.60 ⇒ identity is not in the features: R1a's `L_id` is the first thing to train and R1b waits; ≥ 0.75 ⇒ proceed to R1a as planned (**not** merged with R2) |
| **R1a** representation round | fresh v20 session rows (4.1), actual per-turn RIR distances, a training-only proximity head/loss, frame-level presence head/loss, and explicit post-synthesis paired chain views; identity and onset-keep objectives stay off; separator output **bit-identical at inference** (heads training-only); warm-start v16 ep19; 20 epochs | implementation + CPU checks + a no-update GPU memory smoke; training is a separate authorised action | P1/R6 and the fixed 12/30 s acceptance set must move; G1–G3 must hold. A bounded/scale-free proximity ablation or `OnsetKeepLoss` is a later recipe, never silently mixed into this baseline |
| **R1b** memory without read | candidate/committed slots and `L_mem`, read branch absent; audit the write decision on every real chain and on bystander-first / user-first / re-entry / overlap cases (P6) | ~1 GPU-day, 5 engineer-days | P6 fails ⇒ fix the commit rule, not the read |
| **R2** read | zero-init residual branch after block 2 on R1b's checkpoint; export ports and tests | ~1.5 GPU-days, 7 engineer-days | P1 must not regress while P2 improves (the coupling test); G5 |
| **R3** consolidation | affected ladder stages if the operating point moved; full nine gates + ASR blocks; release decision | 2–3 GPU-days | deploy only if it beats v8 and v16+guard on keep/deletion at matched far suppression |

Every stage: one axis, 5-checkpoint block, paired tests, thresholds committed before launch. A stage that
only matches the shipped guard at the guard's suppression price produced nothing (v14 rule).

### 7.1 Executable v20 baseline amendment (2026-09-05)

The previous R1a draft was removed before training. The current executable
definition is `config/exp/train_dpcrn_v20_r1a.yaml`: `scale_free: false`,
`distance_matched_bystander_prob: 0`, legacy `pair_prob: 0`, and
`paired_view_prob: 0.5`. The second view is nested under `paired_view` and is
consumed only by a generic SISO paired-consistency dispatcher; it never enters
the ordinary separation or presence reduction. The loss compares actual rendered
turn distances, excludes unknown/padded turns and gaps below 0.25 m, and does
not assume that role 1 is physically nearer.

`vad_target` is a frame grid supplied by the configured energy labeler. The
session row's `target_present` is a separate scalar and remains one during a
user's floor gap. Fixed speaker-disjoint 12 s/30 s tensors are materialized and
replayed by `benchmarks/probes/v20_session_rows/session_validation.py`; the
ordinary six-second validation is retained as a regression check. The
no-update memory smoke and 300-row data audit are in the same directory.

Nothing in this amendment starts a training run. `OnsetKeepLoss` remains
archived/off, and no backbone or streaming export path is changed.

## 8. Risks named in advance

1. **The first-talker ambiguity is irreducible without enrollment.** The model can only defer (no read
   before commit) and revise; a product that wants the first voice suppressed at t = 0 needs capability A
   to reach R6 on that chain — or enrollment.
2. **Real-chain transfer.** Every synthetic-only success stopped at the chain. Identity and relative
   proximity are less chain-bound than metres, and the session objective is what real paired corpora also
   supply; the R1a fork is planned.
3. **Write-gate failure on a chain** ⇒ default to no write (passthrough), never to the loudest voice.
4. **Diversity/memory budget** ⇒ state-carrying segments, sessions-per-update pre-registered, smoke test
   before costing.
5. **Deletion explosion** ⇒ long gaps never target-absent; separation losses unchanged.
6. **Coupling regression** ⇒ R2 judged on P1 and P2 together.
7. **Head AUC is not permission to act** ⇒ the read arrives last and separately (v11 lesson).

## 9. Relation to v19c and to the guard

`v19c`'s hinge is a subset of 4.2 on today's data and its pre-flight is S0 here. The shipped `OnsetGuard`
is the external form of §3.3's no-read-before-commit rule; the programme's success criterion is that the
model reproduces that protection internally *and* keeps the far suppression the guard gives up.

---

## 10. Rev 3 addendum (2026-09-04) — S0 and P0 results, and what they change

**S0 (v19c pre-flight, `v19c_preflight/README.md`): the loss-only axis is closed.** On the real train
loader the anchor-inheritance hinge is eligible on 4.9 % of rows but has median 0 (non-zero on 0.9 %
of all rows; reproduced at a second seed), passing the output through the OnsetGuard *raises* the
score instead of lowering it (the contrast is keyed to the bystander's proximity class: training
pre-onset material is a far interferer at −27 dB, the measured failure has a near-talker prefix at
−8 dB), and where non-zero the term owns 50–105 % of the gradient. The row shape (user onset ≥ 1 s
into the row: 6.8 % of rows) has to come from data. This is the pre-registered fork and R1a's
data half (`v20_session_rows/README.md`, committed) is that data: session rows with all four turn
shapes, per-turn labels, one chain draw per row, target-absent share unchanged, no memory cost.

**P0 (`identity_probe/README.md`): the frozen bottleneck does not carry talker identity across
channels.** Real corpora: best cross-channel pair AUC 0.59 (per-recording centred; raw 0.15–0.49
because the chain offset dominates), leave-speakers-out linear ceiling 0.73–0.76 — below the P3
target of 0.80. Synthetic: identity is present when chain, room and distance are all fixed (0.805)
and one chain swap removes it (0.012); factor ladder distance > chain > room ≫ talker. **The repo's
frozen SV backbone (`speaker-verification-ps-spk-v1-1`, 192-d) scores 0.94–0.999 on the very same
pairs**, and costs 78 ms per 2 s window single-threaded on this host's CPU (≈ 8 % of one core at
1 Hz, 16 % at 2 Hz; 15 M parameters), measured 2026-09-04.

**Design change (supersedes §4.2(i) and the identity part of §4.3):**

1. **Identity is supplied, not learned from scratch.** The identity carrier for the candidate /
   committed slots is the frozen SV embedding computed on single-talker speech runs (sliding 2 s
   window, ≥ 1 Hz, only while presence is confirmed and `bystander_active`-style overlap is not
   flagged). The DPCRN learns what P0 shows it can: proximity (capability A) and presence. The
   memory stores an SV embedding; the commit / replacement rules of §3.3 are unchanged; identity
   similarity in the replacement rule is SV cosine (the same quantity that scored ≥ 0.94 across
   DiPCo's close/array channels).
2. **`L_id` becomes distillation, optional.** If the SV side stream is too expensive on the target
   device, `IdentityContrastiveLoss`'s EMA teacher is replaced by the SV embedding of the turn
   (frozen teacher, same loss code path: one `teacher` provider), i.e. the bottleneck head is
   trained to *predict* SV identity rather than to invent one. Whether that costs separation is a
   measured question for R1a, not an assumption; the head stays training-only either way.
3. **R1a scope, as it launches:** session rows ON; `RelativeProximityLoss` ON (capability A; gate
   R6 on the QVF readability AUC and the device AUC); presence head retrained on `user_active`;
   `IdentityContrastiveLoss` ON only in its distillation form and at 0.1 (or OFF for the first
   cycle if the SV provider is not wired in time — state which in the round file). P3 is re-based:
   for the SV carrier it is met by construction (0.94–0.999 measured) and the gate moves to the
   *write decision* (P6); for the distilled head it stays as written.
4. **R1b stores SV embeddings.** The write gate reads the R1a proximity head and presence; the
   candidate/committed logic and P6 audit are unchanged. The streaming export gains the SV model
   as a second session beside the DPCRN graph (or the distilled head if item 2 is taken); the
   cost budget for G5 must include it.

What does not change: sessions as the training material, relative proximity as the write criterion,
no-read-before-commit, the read as a zero-initialised residual branch in R2, every guard in §6, and
the real-chain fork in R1a.

## 11. Binding constraint: real-time is the only deployment target (2026-09-04, user)

Everything in this programme is judged as a **causal, per-hop, CPU streaming** system or not at all.
Concretely, and these override anything above that reads otherwise:

1. **No component may exist offline-only.** Every state the model keeps (candidate / committed
   slots, session statistics, timers, any identity carrier) is a streaming state with an export port
   and a `state_from_tensors` slice, held through the look-ahead warm-up; every rule in §3.3 is
   evaluated on past frames only. Offline-vs-streaming parity (G5) is a pass/kill gate for every
   round from R1b on, not a release-time check.
2. **Cost budget is part of the gate.** G5: RTF ≤ 1.1 × v16 **on the target device CPU**, measured
   there, not on this server. The recurrent separator is ~0.8 M parameters; the record already shows
   a 2.1–2.7 × CPU cost being "over budget" (MambaInter). Anything added has to fit inside the 10 %.
3. **Identity carrier, in cost order.** (a) Default: a **distilled in-graph identity head** — the
   `IdentityHead` (depthwise conv + linear on the 128-d pooled bottleneck, ~10 k parameters, negligible
   at 100 fps) trained to predict the frozen SV embedding of the turn (SV as a frozen teacher at
   training time only). (b) Fallback, opt-in: the SV model as a side session at ≤ 1 Hz on a 2 s window
   over speech-only frames (78 ms single-threaded per call on this host ≈ 8 % of one core here; the
   target CPU number decides). The SV model never runs per hop and never gates the mask per frame:
   identity is read only at commit / replacement decisions, which are ≥ 1 s events by design, so a
   1 Hz identity stream adds no latency to the audio path.
4. **Normalisation must be causal.** P0's per-recording centring (the thing that lifts identity AUC
   0.57 → 0.87 synthetically) is offline; the deployable form is a causal running mean over speech
   frames with a warm-up hold, like the OnsetGuard floor tracker. Before R1b, re-run the P0 pair test
   with causal EMA centring (τ ∈ {5, 10, 20} s) and report the AUC loss vs offline centring.
5. **Latency.** The audio path keeps the existing algorithmic delay (`streaming_delay_frames`); the
   memory read in R2 uses `M_{t-1}` (one hop late by construction); no new look-ahead anywhere.
6. **Training mirrors streaming.** Session rows are trained with the same causal recurrence; if
   state-carrying truncated segments are introduced later, the carried state must be exactly the
   exported state set (no training-only state).
