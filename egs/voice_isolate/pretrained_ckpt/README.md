# voice_isolate — pretrained checkpoints

繁體中文版本：[`README.zh-TW.md`](README.zh-TW.md)

Near-field (<1 m) foreground voice isolation, single channel, no enrollment: keep the near
speaker, suppress far/competing speakers and noise.

Versions are numbered in training order — **`dpcrn_v8.ckpt` is the current default**. Every
version shares the same architecture (DPCRN, complex ratio mask, 16 kHz, ~0.8 M params,
30 ms look-ahead), so one inference config loads all of them:

```bash
uv run python egs/voice_isolate/scripts/demo.py \
    --config_path egs/voice_isolate/config/infer_dpcrn.yaml     # dropdown lists every version
```

## Default: `dpcrn_v8.ckpt` + `dry_blend = 0.9`

The runtime blend is **part of the released configuration**, not an optional extra:

```python
enhanced = model(wav, dry_blend=0.9)      # out = 0.9 * enhanced + 0.1 * input
```

It bounds attenuation at any point to −20 dB. That costs a little residual interferer and
buys a large drop in deletions on capture chains outside the training data — with it the
model *lowers* ASR error on real recordings (Dawn Chorus WER 0.180 vs 0.184 unprocessed);
without it, the same checkpoint raises WER on one of the reverberant test sets. Eval scripts
expose `--dry-blend`; the streaming manifest carries the value under `recommended_inference`.

Known limitation: a blend of 0.9 cannot produce full silence (−20 dB floor by construction),
and suppression of far speech recorded through chains very unlike the training corpora is
shallower than in-domain. For hard muting, the mask itself reaches about −38 dB, but that
path needs a gate rather than a blend.

## Versions

| version | recipe | warm-start | what the stage adds | judged result |
|---|---|---|---|---|
| `dpcrn_v1.ckpt` | `config/exp/train_dpcrn_curriculum_core.yaml` | cold | curriculum core: RT60 0.20–0.45, near/far DRR gap ≥ 6 dB | in-domain SI-SDRi median **+6.98** (ep39) |
| `dpcrn_v2.ckpt` | `config/exp/train_dpcrn_curriculum_expand.yaml` | v1 | widened curriculum: RT60 ≤ 0.65, DRR gap ≥ 3 dB | in-domain **+7.99**; measured-RIR set +5.46; first real-WER win (BUT enh 0.777 < mix 0.796) (ep59) |
| `dpcrn_v3.ckpt` | `config/exp/train_dpcrn_antisup_w1.yaml` | v2 | `OverSuppressionLoss` weight 1.0 (anti-deletion) | in-domain **+8.21**; BUT deletion 0.308 → 0.291 (ep19) |
| `dpcrn_v4.ckpt` | `config/exp/train_dpcrn_antisup_w2.yaml` | v3 | anti-suppression weight 2.0 | in-domain **+8.32**; BUT deletion → 0.276; most even across domains (ep19) |
| `dpcrn_v5.ckpt` | `config/exp/train_dpcrn_antisup_w3.yaml` | v4 | anti-suppression weight 3.0 | in-domain **+8.45**; best in deployment-level reverb, worst in extreme reverb — the domain split point (ep19) |
| `dpcrn_v6.ckpt` | `config/exp/train_dpcrn_wide_antisup.yaml` | v5 | wide RIR domain (RT60 0.20–0.85) + capture realism (media-voice interferer, HPF) | held-out unseen-room **+8.06**; **first streaming-verified** version (see below) (ep19) |
| `dpcrn_v7.ckpt` | `config/exp/train_dpcrn_realE2E_v2c.yaml` | v6 | real recordings on both sides of the decision (real far interferers + real <1 m keep rows), turn-taking far-solo supervision, distance/DRR aux head, channel-perturbation mask consistency | held-out real far-field **−9.45 dB, graded by distance** (1–2 m −3 → 5 m+ −38; v6: −1.76 flat), near-field keep flat (−0.11), **Dawn WER 0.174 < 0.184 raw**, deletion 0.088 ≈ raw floor, reverberant-office WER −0.024 vs mix, in-domain +8.18 (ep19, with `dry_blend 0.9`) |
| **`dpcrn_v8.ckpt`** | **`config/train_dpcrn.yaml`** | v7 | measured-capture realism in synthesis: noise convolved with the speech's own room, an absolute dBFS microphone floor, and part of the synthetic mixture taking its SIR from the scene geometry | real far-field suppression **−15.91 dB** vs v7's −13.72 on identical files (leakage-free subset −17.74 vs −15.68), and the 2–3 m dip in v7's distance response filled in (−4.50 → −16.61) so grading is monotone; near-field keep flat (0.00) with the worst case improved (−5.45 → −1.06); Dawn WER 0.180 < 0.184 raw, deletion 0.094; reverberant-office WER −0.024 vs mix (both interferer counts); turn-taking KEEP 6 violations; in-domain +7.99. Costs: +0.020 WER on the extreme-reverb monitor where v7 was neutral (ep19, with `dry_blend 0.9`) |
| `dpcrn_v9.ckpt` | `config/exp/train_dpcrn_drrcontrast.yaml` | v8 | DRR-contrast augmentation: per fresh RIR channel (prob 0.4) the reverberant tail is rescaled so foreground channels gain up to 4 dB DRR and far channels lose up to 4 dB | the ASR-gate-optimal operating point: reverberant-office WER **0.513, −0.050 vs mix** (v8: 0.539, −0.024; both interferer counts improve — 1 itf 0.480, 2 itf 0.576), **Dawn WER 0.172 / deletion 0.086** (both best of any version), extreme-reverb monitor **−0.003** (v8's +0.020 cost erased), turn-taking KEEP 94/6, in-domain +8.10. Costs: real far-field suppression −12.44 vs v8's −15.91 (paired, 24/71 files >3 dB shallower), turn-taking SUPPRESS 80/20 @ −13.43 vs v8's 87/13 @ −16.14 (ep19, with `dry_blend 0.9`) |
| `dpcrn_v10.ckpt` | `config/exp/train_dpcrn_coldstart.yaml` | v8 | self-calibration curriculum: rows that OPEN with real far-field solo before any near anchor (row-initial-far exposure ~10% → ~23%, anchor-free real lone-far 3% → 7.5%), teaching the model to calibrate against whatever reference exists including the noise floor | **the cold-start axis moved and the deployment gate did not survive it** (ep39, full gate run). Won: bot-idle far suppression 5 of 10 isolated far clips clear −6 dB (v8 1, v9 2) with the deepest single result in the set, best near keep of any version (worst case −0.34 dB), turn-taking SUPPRESS 89/11 @ −15.98 (best ok-rate), Dawn WER 0.173 / deletion 0.088 ≈ v9. Lost: **on the deployment reverberation range it is 0.024 WER worse than v8**, paired on identical utterances, 95% CI [+0.008, +0.041]; turn-taking KEEP violations 6 → 10; in-domain +7.56 (v8 +7.99, v9 +8.10); extreme-reverb back to +0.017 (v9 −0.003). On BUT-OFFICE it reads −0.004 vs v8's −0.024, but that set cannot resolve either number (both intervals span zero at n=200) — it is a monitor, not the gate. The cold-start gain is confined to 200 cm. **Not a deployment candidate** — kept as the cold-start reference and a warm-start point |
| `dpcrn_v16_ep19.ckpt` | `config/exp/train_dpcrn_v16_lengthmix.yaml` | v8 | length schedule: per-batch row length drawn from {3, 6, 12, 30 s} (chapter corpus + real-pool stitching, no zero padding) | field block protocol: keep-violation severity better than v8, session suppression shallower (block −9.73 vs v8 −11.78 dB); Azure WER: lowest deletion of any version on all four sets (Dawn 0.198 vs v8 0.226). Deployment candidate for ASR use; `dpcrn_v8` stays default for suppression depth. |
| `dpcrn_curriculum_v0.ckpt` | `config/exp/train_dpcrn_curriculum.yaml` | **cold** | the whole lineage as ONE run: the knobs the rungs above changed between runs are written as `curriculum` schedules and move while it trains (room pool widening, anti-suppression 0→3, capture realism at ep20, real rows at ep30/40, distance-loss weight with them), mixed row lengths throughout | ~70% of the ladder from a single 120-epoch cold start (ep99): moderate-reverb WER **−0.108** vs the ladder's −0.171, turn-taking SUPPRESS **−15.12** at 84/16 vs −16.14 at 87/13, field sessions **−8.28** vs −11.78, in-domain +6.87. Matches or beats it on two gates: Dawn WER **0.178** / deletion 0.089 (v8 0.180 / 0.094) and keep preservation across the 27 near clips (paired median +0.04 dB, p=0.044). Deeper on the QVF cross-chain far clips (`qvf_price_far1` −17.2 vs −3.6, `qvf_price_far2` −30.1 vs −18.5, n=6) but **not on RealMAN**, where the two are indistinguishable (>1 m median −7.37 vs −7.21) — QVF-specific, not a cross-chain axis. NOT a deployment candidate — it gives up 3.5 dB of field-session suppression, and `dry_blend 1.0` breaks it (Dawn 0.234). Verdict, and the three negative experiments that pin the remaining gap on the warm-start chain itself rather than on budget, room mixture or row length: [`../benchmarks/probes/curriculum_v0_VERDICT.md`](../benchmarks/probes/curriculum_v0_VERDICT.md) |
| `dpcrn_curriculum_v1.ckpt` | `config/exp/train_dpcrn_curriculum_v1.yaml` | `dpcrn_curriculum_v0` | the lineage's second scheduled step: conversational **session rows** (a user and bystanders taking turns, with gaps, re-entry and a seat change), a **paired capture view** of the same row for a consistency term, and two **per-frame heads** — relative proximity supervised by each rendered turn's distance, and presence. Each enters on its own ramp; the file's constants are the end values, so validation measures the final distribution | **the cold-start far-field wall moved, on the internal device chain** (ep39, 5-checkpoint blocks): lone far voice with no near anchor **−7.07 dB** vs the ladder's −0.96 (p=0.000), with room-tone context **−28.27** vs −14.51 (p=0.031), and near-field keep on that chain unchanged (24 clips, median −0.16 vs −0.19). Turn-taking SUPPRESS −17.10 at **94/6**, past the ladder's −16.14 at 87/13; Dawn WER **0.172** / deletion 0.088, the best of any version; moderate-reverb WER −0.156 (91% of the ladder's −0.171, up from 63% one step ago); in-domain +7.32. **Costs**: keep on the cross-chain QVF clips regresses, worst case `qvf_keep_in_touch_near1` −32.8 dB — a per-recording calibration failure, not damaged speech (harmonic contrast unchanged, level down 19.7 dB; the model's own proximity readout puts that clip at −0.59, between healthy near +1.35 and true far −3.00). The keep-side noise guard does not tilt (every s2f level pairs at p>0.05 against the ladder). NOT a deployment candidate for a cross-chain deployment; ep19 is archived beside it as the gentler operating point (worst cross-chain keep −12.6, synthetic far-only probes 2.6–5.9 dB deeper, ASR gates slightly behind). Verdict and the three probes that attribute the cost: [`../benchmarks/probes/curriculum_v1_VERDICT.md`](../benchmarks/probes/curriculum_v1_VERDICT.md) |

> **Rounds v13–v18 (2026-08/09) that did not become a shipped version.** Their configs stay under
> `config/exp/` and their verdicts under `benchmarks/probes/` and `benchmarks/field_test_vector/`:
> v13 inter-LSTM→Mamba (re-init debt on the last curriculum rung), v14 zero-init parallel Mamba
> branch (`v14_VERDICT.md`, no gain beyond a free `dry_blend` knob), v15 30 s-only rows (starves
> scene variety), v17 room-audio lead-in (`v17_round_design.md`; cold start is a missing room
> reference and is fixed on the deployment side, not by training), v18 loss floors
> (`v18_VERDICT.md`; the relative inactive-SDR mode caps the suppression incentive). Checkpoints
> for these live only in `exp/`.

> **On the reverberant-office numbers above.** Every `reverberant-office WER` figure in this
> table (v7 −0.024, v8 −0.024, v9 −0.050, v10 −0.004) comes from a 200-utterance set whose
> bootstrap interval is about ±0.03. Re-measured in 2026-08, neither v8 nor v10 is
> distinguishable from leaving the audio alone on that set, so the ordering it implies is not
> established. The primary WER gate is now `wer_set_moderate_test`, which resolves the same
> comparison cleanly. Detail and the paired numbers:
> [`../benchmarks/wer_sets/README.md`](../benchmarks/wer_sets/README.md).

`dpcrn_curriculum_v0.ckpt` — off the main line in a different sense: same architecture and
the same task, but the only version here that was **not** warm-started from another. It is
the reproducible single-run baseline (one config, one command) and the reference point for
what a schedule can and cannot replace; the versions above remain the deployment options.

`dpcrn_v6_gate.ckpt` — off the main line: `config/exp/train_dpcrn_gate.yaml` freezes v6 and
trains only a causal frame-level near/far VAD gate head (98,689 params). It reaches 0.90+
balanced accuracy on simulated data but the gate does not close on real recordings, so it is
an engineering reference for the gate path, not a deployable model. Its separator weights are
identical to v6; only 10 BatchNorm running-statistic buffers drifted during that run, so its
mask output is v6's up to those buffers.

**Choosing a version.** Take `dpcrn_v8.ckpt` with `dry_blend 0.9` — it remains the default.
`dpcrn_v9.ckpt` sits at a different operating point: pick it when downstream ASR quality on
the near speaker is the objective (its reverberant-office WER gain is double v8's and its
deletion is the lowest of any version), and accept ~3.5 dB shallower far-field suppression —
a far voice is more audible in the residual than under v8. `dpcrn_v7.ckpt` is the
alternative when the deployment sees reverberation well past the training domain (RT60 > 1 s):
it is neutral on the extreme-reverb WER monitor where v8 costs +0.020 (v9 is also neutral
there), and gives up about 2 dB
of real far-field suppression for it. `dpcrn_v6.ckpt` is the conservative fallback: no runtime
knob, and it largely passes far speech through. **`dpcrn_v10.ckpt` is not a deployment option**:
it is the only version that moves bot-idle cold-start suppression, but the full gate showed it
pays for that in WER on the deployment reverberation range: 0.024 worse than v8 on identical
utterances, with the bootstrap interval clear of zero. Take it as the cold-start reference, or as a warm-start point for a recipe that puts the
office gate back. v1–v5 are the training-history stages, kept so any stage can be re-judged or
re-warm-started; they are not deployment candidates.

**What none of them do.** Far speech recorded through a capture chain very unlike the training
corpora is still barely suppressed (about −1 dB on the cross-chain reference clips, where a
commercial reference reaches −44 dB). That gap is a property of the recording chain, not of
distance, and no version here closes it.

**Judging convention.** The scheduler (`CosineAnnealingWarmRestarts`, `T_0=20`) restarts every
20 epochs, so checkpoints are only comparable at the cosine troughs — ep19/ep39/ep59. Every
number above comes from a trough epoch.

## `streaming/` — per-frame ONNX exports

`dpcrn_v6.{onnx,json}` through `dpcrn_v10.{onnx,json}`, plus `dpcrn_v16_ep19` and
`dpcrn_curriculum_v0` and `dpcrn_curriculum_v1`, built with `../scripts/streaming_onnx.py export`. All carry a **30 ms (3-frame) algorithmic latency** from
the look-ahead, handled by future-buffering baked into the graph as extra state
(`puresound/streaming/dpcrn.py`), and all are verified against the offline model once aligned by
that latency. `verify` defaults to a **white-noise** probe, which is a stress signal rather than
a deployment one — v6 88–105 dB, v7 63 dB, v8 49.7 dB, v9 48.2 dB, v10 24.6 dB SI-SDR, a trend
that tracks how aggressively each version modulates its mask, not its streaming correctness.
Pass real speech with `--input_audio` and the same graphs are near bit-exact: on 30 s of the
field benchmark's 90D session, v8 124.4 dB, v9 120.2 dB, v10 116.1 dB. Judge a new export on the
speech number; use the noise number only to compare versions with each other. CPU RTF 0.43. Load with `puresound.streaming.StreamingDpcrnOrt` or the SDK's
manifest-driven `PureSoundStreamingRuntime` (`processor: stft_frame_ort`).

Any offline↔streaming comparison **must** align by the reported latency and trim the edges,
otherwise the delay reads as error; `streaming_onnx.py verify` does this:

```bash
uv run python egs/voice_isolate/scripts/streaming_onnx.py verify \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt \
    egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_v8.onnx \
    --manifest_path egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_v8.json --provider cpu
```

Streaming with v7 or v8 applies the blend on the output: mix the enhanced frame with the input
frame delayed by `streaming_delay_frames`. It costs no extra latency; both manifests carry the
value under `recommended_inference`.

## Re-export / re-train

```bash
# streaming ONNX from any checkpoint here
uv run python egs/voice_isolate/scripts/streaming_onnx.py export \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt /tmp/model.onnx

# run the default recipe, warm-starting from v7 as v8 did
# (from the recipe dir -- the config's metafile paths are relative to it)
cd egs/voice_isolate && uv run python main.py config/train_dpcrn.yaml --training \
    --pretrained_ckpt_path pretrained_ckpt/dpcrn_v7.ckpt
```

Checkpoints here are the judged troughs pulled out of the full training history under `../exp/`
(symlinked to a work volume, gitignored, keeps every epoch). Per-version measurements in
context: `../EXPERIMENT_LOG.md`.

## Filename history

Earlier logs and reports use the pre-versioning names:

| old name | version |
|---|---|
| `dpcrn_curriculum_core_ep39.ckpt` | `dpcrn_v1.ckpt` |
| `dpcrn_curriculum_expand_ep59.ckpt` | `dpcrn_v2.ckpt` |
| `dpcrn_antisup_w1_ep19.ckpt` | `dpcrn_v3.ckpt` |
| `dpcrn_antisup_w2_ep19.ckpt` | `dpcrn_v4.ckpt` |
| `dpcrn_antisup_w3_ep19.ckpt` | `dpcrn_v5.ckpt` |
| `dpcrn_wide_antisup_ep19.ckpt` | `dpcrn_v6.ckpt` |
| `dpcrn_gate_synth_ep7.ckpt` | `dpcrn_v6_gate.ckpt` |
| `dpcrn_realE2E_v2c_ep19.ckpt` | `dpcrn_v7.ckpt` |
| `dpcrn_realism_0729_ep19.ckpt` | `dpcrn_v8.ckpt` |
| `dpcrn_drrcontrast_ep19.ckpt` | `dpcrn_v9.ckpt` |
| `dpcrn_coldstart_ep39.ckpt` | `dpcrn_v10.ckpt` |
| `exp/dpcrn_v16_lengthmix/epoch=19*.ckpt` | `dpcrn_v16_ep19.ckpt` |

Training-run directories keep their original names (`exp/dpcrn_wide_antisup_0702`,
`exp/dpcrn_realE2E_v2c_0722`, `exp/dpcrn_realism_0729`, …).
