# voice_isolate — near-field foreground voice isolation

繁體中文版本：[`README.zh-TW.md`](README.zh-TW.md)

Single-channel, **enrollment-free, near-field (<1 m) foreground** voice isolation: keep the near
speaker, suppress far/competing speakers + noise. The only cue is the **near/far DRR
(direct-to-reverberant ratio) contrast**. Target effect = ai-coustics Voice Focus 2.0.

**Backbone: DPCRN** (complex ratio mask, 16 kHz, ~0.8 M params, 30 ms look-ahead). Picked after an
overfit + full-train comparison showed the original TS-Conformer (mapping head) could not separate
hard near/far cases while DPCRN/DPARN could — see "Pre-DPCRN history" below.

## The v1–v8 ladder's endpoint: `pretrained_ckpt/dpcrn_v8.ckpt` + `dry_blend 0.9`

> The model zoo's default is now `dpcrn_curriculum_v1.ckpt`, from the separate curriculum
> lineage; `dpcrn_v8` is registered beside it as the conservative cross-chain alternative.
> This section describes the ladder documented below, whose last rung v8 is. Both shipped
> versions and the archived ones: [`pretrained_ckpt/README.md`](pretrained_ckpt/README.md).

> **`dry_blend` is applied by the runtime, not by the graph.** The ONNX export
> records it in the manifest under `recommended_inference.dry_blend`, and
> `StreamingOrt` / the portable SDK apply it after the graph. Both manifests
> under `pretrained_ckpt/streaming/` carry `dry_blend: 0.9` there. An export
> without the section gets no relief -- the documented convention.
>
> Note what the blend costs: it keeps `1 - dry_blend` of the input whatever the
> model did, so 0.9 caps attenuation at exactly **-20 dB**
> (`Postprocessor.suppression_ceiling_db`). A far-field residual reported near
> -20 dB is measuring the blend, not the model. Re-export to change it:
>
> ```bash
> uv run python egs/voice_isolate/scripts/streaming_onnx.py export \
>     egs/voice_isolate/config/train_dpcrn.yaml \
>     egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt \
>     egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_v8.onnx --dry-blend 0.9
> ```

Trained with `config/train_dpcrn.yaml` (stage 8 below) and released **with the runtime blend as part
of the configuration** — `out = 0.9 * enhanced + 0.1 * input`, which bounds attenuation to −20 dB.
Stage 7 was the first version to suppress far-field speech in *real recordings* while keeping near
speech; stage 8 deepens that suppression by **2 dB** on identical recordings (>1 m median −15.91 vs
−13.72 dB) and removes a dip at 2–3 m that made stage 7's distance response non-monotone, while
keeping near-field speech flat and **lowering** ASR error on real recordings rather than raising it
(Dawn Chorus WER 0.180 vs 0.184 unprocessed, deletion 0.094). Streaming export verified
(`pretrained_ckpt/streaming/dpcrn_v8.{onnx,json}`, 30 ms latency).

One trade to know about: on reverberation far above the training domain (RT60 > 1 s) stage 8 raises
WER by 0.020 where stage 7 was neutral. `dpcrn_v7.ckpt` was the better pick for very reverberant
deployments; it is archived under `pretrained_ckpt/backup/` rather than shipped.

What made the difference was **real recordings on both sides of the decision** plus **turn-taking
supervision**, not more or better RIRs. Every simulated far-field rung tried before it (boundary
distances, measured RIRs in training, gate-only VAD head, joint separator+gate) learns the
RIR-convolution domain arbitrarily well and does not transfer to real recordings, and pushing
far-suppression harder inside that domain blows up real-acoustic deletion (reverberant-office WER
0.663 vs 0.529). Those rungs are closed.

`dpcrn_v6.ckpt` was the fallback (no runtime knob, most conservative on far-field suppression);
it is archived too. Per-version detail, results, and which versions are still shipped:
`pretrained_ckpt/README.md`.

## The pipeline (9 stages, each warm-started from the previous)

All stages share the **same DPCRN architecture** (`channels [2,32,64,128]`, `rnn_hidden 96`,
look-ahead `delay=[1,1,1]`) — only the RIR bank, augmentation, and loss weights change. Judge only at
`CosineAnnealingWarmRestarts` cosine troughs (ep19/ep39/ep59); mid-cycle epochs are LR-perturbed.

| # | stage | config | warm-start from | judged ckpt | headline result |
|---|---|---|---|---|---|
| 1 | curriculum core (RT60 0.20–0.45, DRR gap≥6dB) | internal | cold | `dpcrn_v1.ckpt` | in-domain SI-SDRi median **+6.98** (ep19 +5.47 → ep39 +6.98) |
| 2 | curriculum expand (RT60 ≤0.65, gap≥3dB) | internal | stage 1 ep39 | `dpcrn_v2.ckpt` | in-domain **+7.99** (ep19 +7.40→ep39 +7.75→ep59 +7.99); BUT real-RIR **+5.46**; first non-neutral real-WER win: BUT enh **0.777 < mix 0.796** |
| 3 | anti-suppression weight 1.0 (`OverSuppressionLoss`) | internal | stage 2 ep59 | `dpcrn_v3.ckpt` | in-domain **+8.21**; large-v3 BUT deletion **0.308→0.291** (direction confirmed, modest) |
| 4 | anti-suppression weight 2.0 | internal | stage 3 ep19 | `dpcrn_v4.ckpt` | in-domain **+8.32**; BUT deletion **→0.276**; safest across both domains |
| 5 | anti-suppression weight 3.0 | internal | stage 4 ep19 | `dpcrn_v5.ckpt` | in-domain **+8.45**; **best in deployment reverb** (moderate enh 0.372, best of all) but **worst in extreme OOD** (BUT enh 0.692, regressed vs w2's 0.676) — "domain split point" |
| 6 | wide-domain deployment (RIR 0.20–0.85 + media_voice/hpf realism) | `test/fixtures/recipes/train_dpcrn_wide_antisup.yaml` | stage 5 ep19 | `dpcrn_v6.ckpt` **(fallback)** | held-out unseen-room **+8.06** (best ever); deployment hard-gate passed (moderate enh 0.373 ≈ w3); BUT still not beaten (0.680, target was <0.676) — 4/5 judge gates passed |
| 7 | real recordings on both sides + turn-taking + distance aux head + channel consistency | internal | stage 6 ep19 | `dpcrn_v7.ckpt` | held-out real far-field **−9.45 dB, distance-graded**; keep flat (−0.11); **Dawn WER 0.174 < 0.184 raw**, deletion 0.088; reverberant-office WER −0.024 vs mix; in-domain +8.18 — needs `dry_blend 0.9` |
| 8 | measured-capture realism in synthesis (room-colored noise, absolute dBFS floor, geometry-driven SIR) | internal | stage 7 ep19 | `dpcrn_v8.ckpt` | real far-field **−15.91 dB** vs stage 7's −13.72 on identical files (leakage-free subset −17.74 vs −15.68); 2–3 m dip filled (−4.50 → −16.61) so grading is monotone; keep flat (0.00), worst case −5.45 → −1.06; **Dawn WER 0.180 < 0.184 raw**, deletion 0.094; reverberant-office WER −0.024 vs mix; in-domain +7.99 — costs +0.020 WER in extreme reverb; needs `dry_blend 0.9` |
| 9 | DRR contrast, then a cold-start curriculum (rows opening on real far-field solo with no near anchor) | internal, internal | stage 8 ep19 | `dpcrn_v9.ckpt`, `dpcrn_v10.ckpt` | **neither displaces stage 8.** v9 is the ASR-gate optimum: reverberant-office WER −0.050 vs mix (double v8's), Dawn 0.172 / deletion 0.086, extreme-reverb neutral — at 3.5 dB shallower real far-field suppression. v10 is the only version that moves bot-idle cold start (5/10 isolated far clips clear −6 dB vs v8's 1) but pays for it in WER on the deployment reverberation range (0.024 worse than v8, paired, CI clear of zero) and raises turn-taking KEEP violations 6 → 10. |

Before the first run, point the recipe's corpus and RIR-bank paths at your own data:
[`DATA_SETUP.md`](DATA_SETUP.md).

Run **from this directory** — the configs' metafile and work-folder paths are relative to it:
```bash
cd egs/voice_isolate

# default recipe: one curriculum run from scratch, no warm start needed
uv run python main.py config/train_dpcrn.yaml --training

# optional second step, warm-started from the first
uv run python main.py config/train_dpcrn_curriculum_v1.yaml --training \
    --pretrained_ckpt_path pretrained_ckpt/dpcrn_curriculum_v0.ckpt
```
`--ckpt_path <ckpt>` instead of `--pretrained_ckpt_path` = true resume (restores optimizer/scheduler/epoch).
Config details (shared design, eval-only variants, inference config): `config/README.md`.

## Fresh v20 experiment (not a continuation of an old R1a run)

The executable v20 baseline is [`train_dpcrn_v20_r1a.yaml`](../../test/fixtures/recipes/train_dpcrn_v20_r1a.yaml), kept as a test fixture.
It keeps the separator objective and backbone path, adds session data labels,
actual per-turn rendered distances, a training-only VAD/presence head, and an
explicit second device-chain view for a generic consistency term. Matched
distance and legacy source-pool pairing are disabled; the proximity baseline is
explicitly unbounded (`scale_free: false`). The nested paired view is excluded
from ordinary separation and VAD reductions.

Session-row collation and the paired view are covered by
`test/test_task/test_session_rows.py` and `test/test_task/test_session_paired_views.py`.
The existing six-second validation stays as a regression check.

## Why weight-3.0 isn't an outright winner (the "domain split point")

Escalating `OverSuppressionLoss` weight 1.0→2.0→3.0 monotonically reduced deletion (over-suppression)
and *raised* in-domain SI-SDRi at every step — but at weight 3.0 the two evaluation domains diverged:
**deployment reverb** (rt60 0.44, in-domain) kept improving, while **extreme-OOD reverb** (BUT real
RIR, rt60 1.15–1.84) got *worse* (substitution/insertion rose faster than deletion fell). SI-SDRi alone
would have said "keep pushing" — WER (real intelligibility) said stop. Stage 6 resolved this by
widening the training RIR domain (0.20→0.85) so the OOD boundary moved outward, rather than tuning the
loss weight further.

## Streaming deployment (ONNX, real-time)

`pretrained_ckpt/streaming/dpcrn_v8.{onnx,json}` (plus `dpcrn_curriculum_v1`) — per-frame
streaming exports built with `scripts/streaming_onnx.py`. The look-ahead (`delay=[1,1,1]`, 30 ms)
is handled by **future-buffering baked into the ONNX graph as extra state** (inter-LSTM warmup gate +
U-Net skip delay lines + noisy-spectrum delay) — no runtime code changes were needed; the existing
manifest-driven SDK runtime (`sdk/python/puresound_streaming`, `processor: stft_frame_ort`) loads it
directly. Verified against the offline model once aligned by the 30 ms latency (SI-SDR 88–105 dB for
v6, 63 dB for v7); CPU real-time factor 0.43. With v7, apply the release blend at the output — mix the
enhanced frame with the input frame delayed by `streaming_delay_frames`; it adds no latency. See
`pretrained_ckpt/README.md` and `puresound/streaming/dpcrn.py` for the mechanism.

## Pre-DPCRN history (compressed)

Before the pipeline above: TS-Conformer (mapping-head) runs plateaued near passthrough on hard
near/far cases (`mix_mode` curriculum helped but couldn't break through; an explicit far-decoder branch
(P1) didn't help either). An `overfit_check.py` capacity test isolated the cause to the backbone, not
the data/loss — TS-Conformer couldn't overfit the hardest batches (enh→target stuck at −2.2 dB) while
DPARN/DPCRN could (+4–5 dB). Switching to DPCRN (complex ratio mask) immediately cleared every
in-domain bucket. A subsequent cold-start-on-full-RIR-bank + `target_absent` combination caused a real-domain
over-suppression catastrophe (Dawn WER 0.626, SI-SDRi −9.53) — this is why the current pipeline is a
graded RIR curriculum with `target_absent: OFF`. Superseded, non-executable configs were removed;
their exact contents remain available from version-control history.

## Evaluation

Tools + usage: `scripts/README.md`.

- **Primary (synthetic, in-domain):** `scripts/eval_indomain.py --by-bucket` (or `check_training_run.sh`).
  SI-SDRi vs the early-reverb target; read the hard buckets (counter_level / 1N+0F / overlap), not the
  aggregate.
- **Real acoustics, deployment-reverb (rt60 0.44):** `config/eval/eval_but_real.yaml`'s sibling set built at
  a moderate RT60 — WER **0.723→0.487 (−32%)**, do-no-harm confirmed. This is the domain the product
  actually ships into.
- **Real acoustics, extreme OOD (BUT real-RIR, rt60 1.15–1.84):** `scripts/build_wer_set.py` (build
  once) → `scripts/eval_wer.py` (SI-SDRi + WER vs real LibriTTS transcripts), config
  `config/eval/eval_but_real.yaml`. Deliberately harder than training; tracks how far over-suppression is an
  OOD-reverb phenomenon (it is — see above).
- **Unseen-room generalization:** internal (seed-2026 disjoint room bank, same
  difficulty distribution as expand). seen→unseen drop is consistently small (≤0.4 dB) across every
  stage — the model generalizes on DRR/geometry, not memorized rooms.
- **Leakage probe (far-only/noise-only):** `config/eval/eval_targetabsent_probe.yaml` — forces every row
  target-absent; checks the model doesn't leak/hallucinate a near speaker. All pipeline checkpoints
  pass (power reduction ≤ −24.8 dB, false-near ≤3%) without ever training on this scenario.
- **Synthetic-vs-real domain-gap decomposition:** `scripts/eval_domain_gap.py` — on matched VOiCES
  (room, mic) triples that have both a real recording and a measured impulse response, splits the
  synthetic-to-real gap into an LTI-convolution ceiling (real vs measured-IR fit) and RIR-bank
  fidelity (measured vs synthetic-bank-IR fit). Produced `data_report/domain_gap_v7.jsonl`, the
  evidence that drove stage 8's measured-capture realism fixes (`dpcrn_v8`).
- **Dawn Chorus = reference / do-no-harm only.** It has **no near/far DRR contrast** and is 78%
  narrowband-GSM, so it is cue/bandwidth-mismatched to this task; a well-behaved model is ≈passthrough
  on it.

## Docs

| file | content |
|---|---|
| `config/README.md` | config layout: `train_*` / `eval_*` / `infer_dpcrn.yaml`, shared design. |
| `pretrained_ckpt/README.md` | checkpoint lineage table + streaming export usage. |
| `scripts/README.md` | all tooling (data prep, training-time validation, benchmarks, inference). |

Data pipeline (dynamic mixing, pre-generated RIR bank near/far contrast, overlap gating) is the
`puresound` library, driven by the `augmentation_*` config blocks.
