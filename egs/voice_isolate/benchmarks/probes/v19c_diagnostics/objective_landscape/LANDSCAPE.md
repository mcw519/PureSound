# Objective / architecture landscape for "persistent, correctly-assigned foreground identity without enrollment"

Label `objective_landscape`, 2026-09-04, CPU only. No code changed, nothing trained.
Every repo number below is quoted from the file named; every external number from the paper/product page named.
My own measurements are in §5 (scripts beside this file).

---

## 0. What the repo already has (conditioning / memory machinery inventory)

| asset | where | state | what it gives a v19 mechanism |
|---|---|---|---|
| FiLM conditioning in BOTH DPRNN blocks | `puresound/nnet/dpcrn.py:36-47,122-135` (`embedding_size=dvec_dim`, `fused_type="film"`), `nnet/lobe/trivial.py:129-167` | built only when `dvec_dim is not None`; **None in every voice_isolate config** (v8, v16) -> dead path, zero cost today | a ready injection point for any identity/anchor vector; `dvec` is L2-normalised before use (`dpcrn.py:264`), i.e. scale-free by construction |
| MISO conditional trainer + frozen SV backbone | `puresound/system/miso.py` (`EncDecCondMaskBase`, siamese encoders, `jointed_trained=False` freezes c-side), `nnet/ecapa_tdnn.py`, `egs/target_speaker_extraction/config/default_config.yaml:377` (`EcapaTdnnExtractor`) | frozen legacy (memory: TSE/SV frozen) | a teacher for distilling identity into the bottleneck; and the reference implementation of "second audio input -> embedding -> FiLM" |
| metric-learning losses | `nnet/loss/spk.py` (`AAMsoftmax`, `SphereFace2`, `GE2ELoss`, `TripletLoss`) | present, unused in voice_isolate | contrastive/identity objectives do not need new loss code |
| per-frame VAD head with multi-scale EMA bank + streaming state | `nnet/lobe/heads.py:78-215` (`ema_taus_s`, debiased EMA in fp32, `initial_stream_state`/`step`) | v11 used it (taus 50 ms/250 ms/1 s/4 s, BCE 0.1); **off in v16** | the exact "fast average vs slow average = envelope modulation / DRR-fill" primitive, already export-safe for streaming |
| background VAD head (twin) | `heads.py` via `background_vad_head`, `loss/vad.py:207` | present; **does not transfer** (see §3 E7) | -- |
| DistHead + regression loss | `heads.py:218-248` (mean over F **and T** -> MLP, 3 outputs), `loss/dist.py`, v16 config `dist_head.enabled: True` w 0.3 | live in v16; **utterance-level pooled**, training-only | distance/DRR supervision already flows from free dataset scalars; a per-frame variant is a small change |
| per-frame foreground activity labels | `vad_label: {used: True, backend: energy}` in both `config/train_dpcrn.yaml:286` and v16 `:296`; consumed only by the overlap-gating labeler ("no loss consumes these labels", v16 config comment line 295) | computed every row, **unconsumed by any loss** | a segment/onset-weighted loss needs no new labelling pass |
| turn-taking / conversational row generator | `puresound/task/overlap_gating.py:166-227` (`_turn_taking`, `_turn_script`) | live: v16 `augmentation_speech.turn_taking_prob 0.3`, `augmentation_realnear.turn_taking_prob 0.5` | alternating near/far turns, `far_first_prob 0.5` = "far monologue with no preceding near anchor", gaps, boundary overlaps |
| real near / real far pools with speaker+room+mic identity | `puresound/task/voice_isolation.py:140-330` (`realnear_speaker`, `exclude_speaker`, `(speaker, room, mic)` stitching) | live | same-talker-across-a-gap and different-talker-after-gap rows on ONE real chain are constructible from existing metadata |
| chain-invariance regularizers | `system/siso.py` `channel_consistency` (EQ+gain perturbation, mask-consistency, prob 0.3, weight 1.0); `augmentation_compressor` (v11b) | consistency live in v16; compressor lives in `config/exp/train_dpcrn_v11b_compinv.yaml` | the only confirmed mechanism against the QVF-chain inversion |
| asymmetric (anti-deletion) loss | `nnet/loss/stft_loss.py:230-259` `OverSuppressionLoss` (`mean(ReLU(|T|^p-|E|^p)^2)`, p=0.5), weight **3.0** in v16 | live | already the asymmetric direction VoiceFilter-Lite argues for -- but a **uniform mean over all T-F bins**, no per-frame weight |
| suppression floor terms | `SDRLoss(inactive_mode)`, `ResidualReferenceLoss` | live, `absolute` | the "kill forever" gradient; v18 proved relaxing it is harmful |
| inference actuators | `system/onset_guard.py` (`OnsetGuard`, productised 2026-09-04), `system/presence_gate.py`, `Postprocessor` (`dry_blend`, `spec_floor`, `suppression_ceiling_db`) | shipped/opt-in | the safety belt any trained mechanism must beat, and the fallback if it does not |
| zero-init parallel-branch method + preflight | `nnet/lobe/ssm.py` `MambaInter(zero_init_out=True)`, `inter_type="lstm+mamba"`, `scripts/preflight_ckpt_recipe.py` | kept from the closed v14 axis | how to add a module to a trained net with **bit-identical** step-0 behaviour (v14: `max|diff| = 0.000e+00`) |
| bottleneck stash | `dpcrn.py:250-258` `stash_bottleneck` | opt-in | readout probes and any self-pooled embedding |

---

## 1. Top 5 mechanisms, ranked

### M1 -- Anchor-persistence + wrong-anchor supervision, with an onset-weighted loss  (rank 1)
**What it does.** Two data rows the recipe has never produced, plus one loss weight:
(a) *return rows*: the same real foreground talker returns after 3-10 s of true floor, label = keep (NOT target-absent);
(b) *wrong-anchor rows*: bystander (different talker, same chain) speaks first, then the user starts -- currently generated only with an inter-turn gap <= 0.4 s;
(c) a per-frame weight on `OverSuppressionLoss` (and optionally an activity-masked SDR term) that upweights the first ~1 s after every foreground onset, using the `vad_target` labels that already exist.

**External evidence.**
* **LExt / onset prompting** (arXiv:2505.05114): prepending enrollment at the *waveform* level -- creating an earlier onset for the target -- is enough to condition a generic separator with **no speaker-conditioning module at all**: 24.1 dB SI-SDRi WSJ0-2mix vs USEF-TSE 23.3; 18.3 vs 16.1 on WHAMR!; **22.0 dB SI-SDRi from a 0.5 s prompt**. So "earliest onset defines the foreground" is a learnable, strong cue -- which is also the honest description of what our streaming state already does by accident. (LExt itself is offline; see E8.)
* **Weighted speech-distortion losses** (arXiv:2001.10601, Xia et al.): splitting the loss into speech-active / speech-inactive terms with separate weights is the standard cure for the deletion/suppression trade; **frequency- and transient-weighted losses** (arXiv:2506.18714, arXiv:2606.21635) weight by spectral flux precisely to protect onsets/plosives.
* **VoiceFilter-Lite** (Interspeech 2020) established that "most WER degradation is false deletions = over-suppression" and answered it with an **asymmetric loss** plus an inference-time over-suppression compensation -- i.e. our `OverSuppressionLoss` + `dry_blend`, already here. The unexploited half is *where* the asymmetry is applied.
* **TSOS** (ICASSP 2023 DNS-5 PSE): a signal-level target-speaker-over-suppression metric, "critical for models to have very low TSOS"; DEL alone was found to miss over-suppression. Worth importing as a scorecard axis (our span-energy metric conflates four layers).

**Repo evidence FOR (this is the strongest FOR on the page).** `probes/anchor_gate_README.md`: an inference-side rule that only refuses to attenuate until a talker has been heard for 1 s removes **half** the keep violations with no readout (v8 26/148 -> 13, v16 20/148 -> 7), and with the relative-DRR arm takes Dawn **deletion 0.230 -> 0.096 (v8) and 0.207 -> 0.094 (v16)**, WER 0.333 -> 0.194 / 0.297 -> 0.188, every arm p < 1e-26 paired. The bystander-context penalty (+0.027/+0.028) closes to +0.003/+0.005 (p 0.30/0.34). `reference_matrix_README.md` §3 localises the same error in training terms: the anchor needs ~1 s and saturates at 2 s, half of it is gone after 5 s of floor and all after 10 s, and the *same* 2-3 s prefix that unlocks far suppression creates 4-6 device keep violations on the next near talker. So the deletion is onset + wrong-anchor, it is worth ~0.13 absolute deletion, and it is currently bought back at inference for -3..-4 dB of far suppression -- the price a trained fix would avoid.

**Repo machinery.** `_turn_script` already has `turn_gap_seconds`, `far_first_prob`, `turn_near/far_seconds`; realnear/realfar pools carry `(speaker, room, mic)`; `length_schedule` has a 30 s bucket (prob 0.20) long enough to hold a 10 s gap; `vad_label` per-frame labels exist and no loss reads them; `OverSuppressionLoss` is one `mean()` away from accepting a weight.

**Collides with.**
* `FAILED_NOTE` §5 / NEXT_STEPS (target_absent) -- **a silence gap must not be implemented as a target-absent row** (Dawn WER 0.392 -> 0.626, del 0.286, SI-SDRi -9.53). The foreground label stays alive through the gap; only the *audio* is silent.
* v18 -- do not touch the loss floors while doing this (`inactive_mode: relative` cost cold-far +0.53 dB, p=0.017).
* v11b -- keep-side healing historically costs about half the anchored far suppression (sessions -12.35/-11.29/-9.25 -> -6.64/-5.95/-6.02). Guard: the session block band, not a count.
* `curriculum-ladder-is-load-bearing` -- warm-start from v16 ep19.
* `FIELD_BLOCK.md` / v17 §2a -- judge on 5-checkpoint blocks with paired Wilcoxon; single-ckpt cold-far deltas are inside a -0.35..-3.39 dB null band.
* `wer_sets/README` -- the 10 s cap inflates deletion; compare deltas.

**My measured support (§5).** In the pipeline-built moderate WER set the foreground's first onset is at **0.06 s median (max 0.14 s; 0/200 rows >= 0.5 s)**, the median longest *internal* foreground gap is **0.335 s**, **0/200 rows have a >= 3 s gap**, and onset frames (first 1 s after >= 1 s of foreground silence) are only **14 % of active frames / 22 % of active energy**. The training row generator's inter-turn gap is capped at **0.4 s** (`OverlapControlConfig.turn_gap_seconds = (0.0, 0.4)`, not overridden in v16). Conclusion: the behaviour the field probes ask for (hold the foreground across a multi-second silence; do not delete the talker who follows a different anchor) is **not in the supervision at all**, and the frames where it would be scored carry at most a fifth of the loss mass.

---

### M2 -- Train the *relative* comparison into a per-frame head, and let the existing actuator read it  (rank 2)
**What it does.** The AnchorGate's best arm is a *relative* rule ("current window reads nearer than the held anchor by a margin"). Make the comparison internal: a per-frame head that predicts "nearer than the anchor / same / farther" (or a per-frame DRR with an anchor-relative target), trained from the row's own geometry labels, and read by `PresenceGate`/`OnsetGuard` with a self-calibrated (session-relative) operating point.

**External evidence.**
* **Streaming Sortformer** (Interspeech 2025, arXiv:2507.18446): an **Arrival-Order Speaker Cache** stores frame-level embeddings of previously observed speakers, ordered by arrival time, updated by selecting the highest-scoring frames from the model's own past predictions; trained with **arrival-time-ordered targets** ("sort loss"). Streaming, low latency, competitive with the offline model; a returning speaker is re-identified from the cache. This is the closest published thing to "a persistent, correctly-assigned slot without enrollment", and it says the persistence must be *supervised by arrival order* and *carried in an explicit, score-updated cache*.
* **EvoTSE** (arXiv:2604.06810): enrollment continuously updated by **reliability-filtered retrieval over high-confidence historical estimates**; reduces speaker confusion, relaxes enrollment-quality requirements, consistent gains **especially out of domain**. Independent evidence that "hold and refresh the anchor from your own confident output" works.
* **VoiceFilter-Lite adaptive suppression strength**: instead of a fixed suppression, predict the condition and scale the strength, smoothed by a moving average -- the commercial precedent for a model-internal, per-frame strength decision rather than a fixed blend.

**Repo evidence FOR.** `presence_selfcal_README`: within-recording ordering survives what pooling cannot (QVF pooled 0.474 vs within-session 0.886/0.911), and the relative rule survived **checkpoint drift** where the fixed threshold died (balanced acc 0.74 -> 0.62-0.70 fixed vs 0.77-0.82 calibrated; suppression -3.2 -> -0.5 dB fixed vs -2.5..-4.2 calibrated). `anchor_gate_README` §1: the anchor's distance IS readable from the state on the device chain (AUC 0.989 dist / **0.997 DRR** isolated; 0.70-0.78 in streaming sessions) -- so this is not asking the network to see something it cannot. `v11b_VERDICT`: scenario3_session 0.596 -> 0.726 crossed into the range the self-calibrated gate needs and produced the first non-zero scenario3 actuation (-1.9 dB, zero keep damage).

**Repo machinery.** `VADHead` + EMA bank (the fast/slow-average ratio is exactly a relative feature) with streaming `step()`; `DistHead`+`DistHeadRegressionLoss` labels; `PresenceGate`; `OnsetGuard`; `train_vad_head_only` (frozen separator, head-only adaptation); `presence_head_judgment.py` protocol with 180D held out.

**Collides with.** `presence_head_v11_README` (the background twin does **not** transfer: real pooled 0.475, scenario3 0.058 -- do not build the double-talk logic on it); `full_gate/v8_gate050_VERDICT` (a gate shipped on a signal that held in one condition); `anchor_gate_README` §2.2 (an **absolute dB margin does not transfer between checkpoints**: v8's 3 dB costs v16 ten suppress passes -> the margin must be self-calibrated or learned); §1 (on QVF the distance readout is *inverted* 0.265 and DRR is marginal -> **M4 is a prerequisite** for the QVF chain); `distance-cue-survives-mask-ignores-it` (DistHead is utterance-pooled; the presence question needs per-frame); `presence_selfcal_README` (through the real actuator the benefit shrinks -- the binding constraint is evidence quality, i.e. the head).

---

### M3 -- Self-conditioning loop: pool the bottleneck into an anchor vector and re-inject it through the existing FiLM path  (rank 3: highest ceiling, highest risk, one unmeasured prerequisite)
**What it does.** Pool the bottleneck over confirmed-foreground frames -> L2-normalise -> feed back as `dvec` through the FiLM path already present in both DPRNN blocks; update only on high-confidence frames, hold through silence with a slow forgetting horizon. Train it with a same/different-talker objective (the `spk.py` losses) and/or by distilling the frozen ECAPA teacher, plus a channel-invariance term so the vector cannot encode the chain.

**External evidence.**
* **Pärnamaa, Interspeech 2024, "PSE Without a Separate Speaker Embedding Model"**: the embedding is the model's *own* internal representation (after the temporal block / final GRU + LayerNorm), frame embeddings averaged. Background-speaker suppression 33.04 dB vs 34.30 dB with a separate speaker model, but **TSOS over-suppression 0.010 vs 0.027 (2.7x less)** and MOS 3.25 vs 3.21; the small model is **1.07 M params, 0.135 ms/frame**. Two things matter for us: the internal representation is a *sufficient* conditioning vector at our parameter scale, and routing it internally *reduced over-suppression* -- our exact failure mode. (Enrollment audio is still used at inference there; the on-the-fly part is ours to add.)
* **EvoTSE** (above) supplies the update rule (reliability-filtered, from own high-confidence output).
* **USEF-TSE** (arXiv:2409.02615) / **SEF-PNet** (ICASSP 2025): frame-level cross-attention instead of a fixed speaker embedding is now SOTA-competitive (USEF-TSE 23.3 dB SI-SDRi WSJ0-2mix) -- i.e. dropping the speaker-recognition model is not a handicap.
* **Mixture-to-set embeddings** (arXiv:2604.03219, Apr 2026): per-speaker embeddings predicted *from the mixture itself*, Hungarian-matched to an ArcFace teacher's embeddings of the clean sources -- enrollment-free identity vectors are learnable; they degrade when the cosine margin between candidates is small; **target selection is left to an external signal** (see E9).

**Repo machinery.** FiLM path + `dvec_dim` (`dpcrn.py`), MISO/ECAPA teacher, `spk.py` losses, `stash_bottleneck`, `channel_consistency`, the v14 zero-init parallel-branch method for adding the pooling/projection module to the trained net without re-initialisation debt.

**Collides with.**
* `real-e2e-round1-verdict` (**channel-signature anchoring**): the round-1 keep column collapsed along the channel domain -- a self-pooled vector on a device chain is at high risk of encoding the *chain*, not the talker. Mandatory: compressor + `channel_consistency` on, and a cross-chain (QVF) validation before any claim.
* `reference_matrix_README` §3.3 / §6.2: today the state keeps "whoever it just heard" and treats a change of talker as a change of foreground. A naive self-enrolment loop **cements** that error; M3 is only safe *on top of* M1's supervision.
* v11/v11b: heads on this bottleneck read the QVF chain inverted (scenario3 0.253 / 0.411) -> M4 prerequisite again.
* Cost/latency: the pooling+FiLM adds params and (if a second forward is needed for consistency) step time against a ~10 ms/frame budget; v14 closed an axis for +110 k params / +2 ms.
* **Unmeasured prerequisite**: nobody has probed whether this bottleneck separates *talker* identity from *channel* identity. The existing per-frame cache cannot answer it -- all 450 Dawn utterances are one talker per recording (26 recordings, ids all `..._h_N`), so same-speaker pairs are always same-channel and any AUC would be confounded. A clean probe needs two talkers in one recording (the field set's own multi-talker sessions, or a paired corpus from `corpus_pair_probe`).

---

### M4 -- Finish chain/compression invariance (prerequisite multiplier, cheapest of the five)  (rank 4 by novelty, rank 1 by cost-effectiveness)
**What it does.** Continue v12 axis 1: offline phantom-distance probes for codec / EQ / noise-gate through the methodology that convicted compression; the factors that move the readout become recipe knobs stacked on `augmentation_compressor` + `channel_consistency`.

**Evidence.** `v11b_VERDICT`: compressor augmentation ALONE moved scenario3 head AUC 0.252 -> 0.365 (ep19) -> 0.411 (ep31) and scenario3_session 0.596 -> 0.726, improved the held-out 180D orientation 0.841 -> 0.889, and healed every QVF keep deletion including a -11.79 dB double-talk cut -- for the price of ~half the anchored far suppression at a mid-cycle checkpoint. `eq_probe_README` says compression explains ~1/3 offline and ~1/2 in training; the remainder is unclaimed. `anchor_gate_README` §1/§5: on the QVF chain distance is **inverted** and DRR marginal -- "no supervisor reads through it", so M2 and M3 both inherit this as a prerequisite. Commercial corroboration: Krisp VI 2.5 explicitly fixed "over-processing in reverberant rooms" and reports competing-speech WER ~36 % -> ~11 % with a clean-audio penalty held at ~0 (2.15 vs 2.10) at 15 ms algorithmic latency, CPU-only; ai-coustics/Krisp both keep a hard keep-side guardrail exactly as our gates do.

**Collides with.** Nothing recorded as negative. Costs: suppression depth (the v11b tax), and the axis is unfinished rather than unproven.

---

### M5 -- Real paired-recording supervision (proxy / pseudo-label from close+far real pairs)  (rank 5 by cost, but the only mechanism with a record of moving the real-chain wall)
**What it does.** Manufacture real-domain targets instead of simulating: close-mic (or projected) reference for a far real recording; train on real conversational mixtures.

**External evidence.** `ctPuLSE` (arXiv:2407.19485, close-talk pseudo-labels for far-field enhancement); NTT C2D projection (arXiv:2606.13109) and CTRnet/PuLSS (arXiv:2605.19695) already logged in `v12_candidate_axes`; **PS4: proxy-supervised joint training for real TSE** (arXiv:2607.08111); **REAL-TSE, SLT 2026** (arXiv:2607.15198): dev set 1,991 samples from AISHELL-4 / AMI / AliMeeting / CHiME-6 / DiPCo, WeSep BSRNN baselines at **TER 0.829/0.838, SIM 0.417/0.443, DNSMOS-P808 2.875/2.756**, separate **online (causality enforced inside the extractor)** and offline tracks, and an official metric set that includes **timing F1** -- a keep/delete-timing metric this repo lacks and should import. Also **REAL-T** (Interspeech 2025) as the underlying real-mixture protocol. Unsupervised fallbacks with 2024-2026 traction: RemixIT / continual self-training, **test-time adaptation for SE** (arXiv:2508.01847 Interspeech 2025; arXiv:2601.14770 mask polarization; arXiv:2509.04280 domain-invariant TTA).

**Repo evidence.** v8's real suppression came from **real recordings on both sides + turn-taking**, not better RIRs (`README.md`); the synthetic-RIR axis is closed three ways; `ct_projection_README` measured an honest projection ceiling of **6.25 dB** with only ~8 % of overlaps qualifying (transplant failed, failure mode benign -- it does not teach deletion).

**Collides with.** `real-e2e-round1-verdict` (real far-field rows taught real suppression -- held-out -13.6 dB, 5 m+ -51 dB -- with **zero cross-corpus transfer** and Dawn deletion 0.50); the DiPCo far-pool build spec parked in `v17_round_design` (Wiener-deconv DRR gate <= -7 dB, cross-talk gating, session/speaker/device quarantine to keep `corpus_pair_probe` alive). Slow and expensive; not a v19 single-axis round.

---

## 2. Reading the ranking

* M1 and M4 are the only two whose target error is **already localised and quantified** (0.13 absolute Dawn deletion for M1; a measured AUC inversion for M4), and neither needs a new parameter. Run them first; M4's offline half is ~0 GPU.
* M2 is the natural second axis because its actuator is already shipped (`OnsetGuard`) and its rule is the best measured keep/suppress trade on record -- what is missing is only that the *margin* lives outside the model.
* M3 has the highest ceiling (it is the literal answer to the question, and Pärnamaa shows it costs nothing at our parameter scale) but it is the one mechanism that can **cement** the repo's known failure and re-run the channel-anchoring disaster. It needs M1 (correct anchor supervision), M4 (a chain-invariant readout) and one new probe (talker vs channel identity in the bottleneck) before it is a round.
* M5 stays the only lever with a record of moving the real-chain wall, and the 2026 literature has converged on it -- but it is a quarter, not a round.

## 3. Exclude list (with the recorded result each collides with)

| # | excluded mechanism | why -- the number that kills it |
|---|---|---|
| E1 | any **absolute-threshold** gate on level / DRR / presence logit / distance | died twice: chain offset (QVF pooled AUC 0.474 vs within-session 0.886/0.911, `presence_selfcal` F1) and checkpoint drift (fixed threshold balanced acc 0.74 -> 0.62-0.70, suppression -3.2 -> -0.5 dB; calibrated 0.77-0.82 / -2.5..-4.2, F4). Also the AnchorGate dB margin: v8's 3 dB costs v16 ten suppress passes |
| E2 | **room / ambience reference** conditioning ("prime with room tone", room-embedding FiLM as originally written in v12 axis 5) | the mechanism does not exist: true-floor pads are a null on 5-ckpt blocks (Delta -0.04 / +0.03 dB, p 0.08 / 0.76), the pads that "unlocked" contained an unlabelled utterance at -47 dBFS; v17 Part B (`row_initial_ambient`) negative on its pre-registered primary (far `none` -1.71 -> -1.48, p=0.15); §2b withdrawn (`reference_matrix_README` §0) |
| E3 | more/better/broken **synthetic RIRs** for real behaviour (bank swap, boundary bank, `direct_smear`) | RIR axis closed 3x (separator 07-08, gate-only 07-10, joint 07-16); and `dist_cue_anatomy` kills the "break the timing cue to force a spectral fallback" premise -- destroying BOTH cues keeps *more* separation (54 %) than destroying timing alone (35 %), so there is no independent backup cue waiting |
| E4 | relaxing the suppression floor / "kill forever" term (`SDRLoss inactive_mode: relative`, MRSTFT relative floor) | v18: cold-far **+0.53 dB worse, p=0.017**; sessions, cold-far and turn-taking SUPPRESS all shallower; keep side did not move |
| E5 | teaching persistence with **synthetic target-absent** silence rows | Dawn WER 0.392 -> 0.626, deletion 0.286, SI-SDRi -9.53 (`FAILED_NOTE` §5). A "user silent for 5 s" row must keep its foreground label |
| E6 | swapping / re-initialising the **context carrier** (Mamba or any inter-path replacement) | v13 lost 8 dB of session suppression; v14's zero-init branch worked mechanically but at matched suppression a free `dry_blend 0.85` on unchanged v11b **halves** its Dawn deletion (0.181/0.092 vs 0.271/0.175); the eight-stage ladder is load-bearing (last stage from scratch -1.65 dB vs v8 -9.83) |
| E7 | **background-VAD-head** double-talk logic | does not transfer: real pooled AUC 0.475, scenario3 0.058, plumbing 0.123, gym 0.136 (`presence_head_v11_README`) |
| E8 | **prepend-enrollment / LExt-style** conditioning as the deployed mechanism | offline by construction -- the authors state that referring to the prepended enrollment's tensors in real time "would be costly", and no frame-online variant is proposed; our product has no enrollment. Keep the *insight* (earliest onset = identity cue -> M1), not the architecture |
| E9 | **mixture-to-set / set-of-embeddings** enrollment-free TSE as published | non-causal, WavLM-backbone teacher-student (orders over a ~1 M-param / 10 ms budget), and target **selection is delegated to external interaction or ASR cues** -- selection is precisely our problem. Degrades when the candidate cosine margin is small |
| E10 | `dry_blend` as the **cold-start** fix | measured no-op at cold start (ceiling never binds, median gain 0.03 dB); only a gain moves it (-0.39 -> -10.01 dB) at the cost of a KEEP violation (`b-detector-works-blend-actuator-cannot`). `dry_blend` remains the right *free knob* for keep-side trades (v14) -- just not for this |

## 4. Protocol / metric imports worth taking with any of M1-M5

| import | source | why here |
|---|---|---|
| **TSOS** (target-speaker over-suppression, signal-level) | ICASSP 2023 DNS-5 PSE; used in Pärnamaa 2024 (0.010 vs 0.027) | our keep-side metric is span energy, which `span-energy-metric-conflates-four-layers` says cannot attribute; TSOS is designed for exactly the "removed target segments" question and does not need ASR |
| **timing F1** | REAL-T / REAL-TSE SLT 2026 official protocol | scores keep/suppress *timing* -- the onset and gap behaviour M1 targets -- without energy pooling |
| **online vs offline tracks with causality enforced inside the extractor** | REAL-TSE | the discipline our streaming export already keeps; useful framing when comparing to published numbers |

## 5. What I measured myself (CPU, this session)

Scripts beside this file. `onset_mass.py` and `first_onset.py` read `data_report/wer_set_moderate_test/*_ref.wav`
(200 items; `ref` = near-reverb foreground). Frame grid 10 ms; activity = frame energy within 30 dB of the file's
95th-percentile frame energy and above -60 dBFS.

| quantity | measured |
|---|---|
| foreground first-onset time | median **0.06 s**, p90 0.08 s, max **0.14 s**; 0/200 rows >= 0.5 s |
| median longest *internal* foreground silence | **0.335 s** |
| rows with an internal foreground gap >= 1 s / >= 3 s / >= 5 s | **2 % / 0 % / 0 %** |
| onset frames (first 1 s after >= 1 s of foreground silence, row start included) | **14.1 %** of active frames, **22.5 %** of active energy (median); ~1 such onset per row |
| same, with a 0.5 s / 0.2 s silence definition | 19.9 % frames / 26.8 % energy; 31.5 % / 46.3 % |
| training row generator's inter-turn gap | `OverlapControlConfig.turn_gap_seconds = (0.0, 0.4)` s (`puresound/config/augmentation.py:57`), **not overridden** in `config/exp/train_dpcrn_v16_lengthmix.yaml` |
| `far_first_prob` | 0.5 (default, not overridden) -- so "far monologue first" IS supervised, but only with a <= 0.4 s gap before the near turn |
| per-frame labels available to a weighted loss | `vad_label` on in v8 and v16 configs; v16 comment (line 295): "no loss consumes these labels" |

**Caveat that matters.** `scripts/build_wer_set.py` is a standalone builder (LibriTTS + BUT real RIRs + DNS-5 noise), **not**
the training dataloader, so the onset/gap numbers describe the *eval* distribution, not training batches. They are quoted
as corroboration of the training-side fact that IS read from the recipe: the inter-turn gap cap of 0.4 s and the absence of
any "foreground returns after multi-second silence" row type. A direct measurement on training batches
(instantiate the v16 dataloader, log per-row foreground-gap and onset statistics) is ~1 CPU-hour and not done here.

**Not measured / explicitly open**
* Whether the DPCRN bottleneck separates **talker** identity from **channel** identity (the prerequisite for M3). The existing
  per-frame cache cannot answer it: 450 Dawn utterances / 26 recordings, all single-talker (`..._h_N`), so same-talker pairs
  are always same-channel.
* Whether an onset-weighted `OverSuppressionLoss` changes anything -- no training was run.
* Any number for M1-M5 as trained mechanisms. Everything above is either a repo measurement already on record, an external
  published number, or one of my two CPU measurements in this section.

## 6. Sources (external)

* LExt / onset prompting -- https://arxiv.org/html/2505.05114v1
* USEF-TSE -- https://arxiv.org/html/2409.02615v3
* EvoTSE -- https://arxiv.org/abs/2604.06810
* Streaming Sortformer (AOSC) -- https://arxiv.org/pdf/2507.18446 , https://www.isca-archive.org/interspeech_2025/medennikov25_interspeech.pdf
* PSE without a separate speaker embedding model (Pärnamaa, Interspeech 2024) -- https://arxiv.org/html/2406.09928v1
* Mixture-to-set enrollment-free embeddings -- https://arxiv.org/html/2604.03219v1
* Positive/negative noisy enrollments (contrastive TSE) -- https://arxiv.org/pdf/2502.16611
* Speaker-embedding-free / disentangled enrollment PSE -- https://arxiv.org/pdf/2505.12288
* VoiceFilter-Lite (asymmetric loss, adaptive suppression strength) -- https://www.isca-archive.org/interspeech_2020/wang20z_interspeech.pdf
* Weighted speech-distortion losses -- https://arxiv.org/pdf/2001.10601 ; transient/frequency-weighted losses -- https://arxiv.org/pdf/2506.18714 , https://arxiv.org/pdf/2606.21635
* Distance-based sound separation (Patterson 2022) -- https://www.isca-archive.org/interspeech_2022/patterson22_interspeech.pdf ; hyperbolic distance-based separation (2024) -- https://minjekim.com/wp-content/uploads/icassp2024_dpetermann.pdf
* ctPuLSE -- https://arxiv.org/pdf/2407.19485 ; PS4 -- https://arxiv.org/html/2607.08111 ; REAL-TSE SLT 2026 -- https://arxiv.org/html/2607.15198v1 ; REAL-T -- https://www.isca-archive.org/interspeech_2025/li25da_interspeech.pdf
* Test-time adaptation / training for SE -- https://arxiv.org/pdf/2508.01847 , https://arxiv.org/html/2601.14770 , https://arxiv.org/pdf/2509.04280 ; RemixIT -- https://github.com/etzinis/unsup_speech_enh_adaptation
* Krisp Voice Isolation 2.5 -- https://krisp.ai/blog/voice-isolation-2-5/ ; Krisp BVC / turn-taking -- https://krisp.ai/blog/improving-turn-taking-of-ai-voice-agents-with-background-voice-cancellation/ ; ai-coustics benchmarks -- https://ai-coustics.com/benchmarks-quantitative
* DNS-5 PSE / TSOS -- https://arxiv.org/pdf/2303.06811
