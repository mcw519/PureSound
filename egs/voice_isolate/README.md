# voice_isolate — near-field foreground voice isolation

Single-channel, **enrollment-free, near-field (<1 m) foreground** voice isolation: keep the near
speaker, suppress far/competing speakers + noise. The only cue is the **near/far DRR
(direct-to-reverberant ratio) contrast**. Target effect = ai-coustics Voice Focus 2.0.

**Backbone: DPCRN** (complex ratio mask, 16 kHz, ~1 M params, 30 ms look-ahead). Picked after an
overfit + full-train comparison showed the original TS-Conformer (mapping head) could not separate
hard near/far cases while DPCRN/DPARN could — see "Pre-DPCRN history" below.

## Current best checkpoint (2026-07-02): `pretrained_ckpt/dpcrn_wide_antisup_ep19.ckpt`

Wide-domain deployment recipe (widened RIR + realism augs), stage 6 of the pipeline below. Also the
first checkpoint **verified for real-time streaming deployment** — a bit-exact per-frame ONNX export
exists at `pretrained_ckpt/streaming/dpcrn_wide_antisup_ep19.{onnx,json}` (30 ms algorithmic latency,
CPU RTF 0.43). Training is currently **paused at ep19** pending a decision to true-resume to ep39 or
wait for the in-progress high-reverb RIR bank.

## The pipeline (6 stages, each warm-started from the previous)

All six stages share the **same DPCRN architecture** (`channels [2,32,64,128]`, `rnn_hidden 96`,
look-ahead `delay=[1,1,1]`) — only the RIR bank, augmentation, and loss weights change. Judge only at
`CosineAnnealingWarmRestarts` cosine troughs (ep19/ep39/ep59); mid-cycle epochs are LR-perturbed.

| # | stage | config | warm-start from | judged ckpt | headline result |
|---|---|---|---|---|---|
| 1 | curriculum core (RT60 0.20–0.45, DRR gap≥6dB) | `config/train_dpcrn_curriculum_core.yaml` | cold | `dpcrn_curriculum_core_ep39.ckpt` | in-domain SI-SDRi median **+6.98** (ep19 +5.47 → ep39 +6.98) |
| 2 | curriculum expand (RT60 ≤0.65, gap≥3dB) | `config/train_dpcrn_curriculum_expand.yaml` | stage 1 ep39 | `dpcrn_curriculum_expand_ep59.ckpt` | in-domain **+7.99** (ep19 +7.40→ep39 +7.75→ep59 +7.99); BUT real-RIR **+5.46**; first non-neutral real-WER win: BUT enh **0.777 < mix 0.796** |
| 3 | anti-suppression weight 1.0 (`OverSuppressionLoss`) | `config/train_dpcrn_antisup_w1.yaml` | stage 2 ep59 | `dpcrn_antisup_w1_ep19.ckpt` | in-domain **+8.21**; large-v3 BUT deletion **0.308→0.291** (direction confirmed, modest) |
| 4 | anti-suppression weight 2.0 | `config/train_dpcrn_antisup_w2.yaml` | stage 3 ep19 | `dpcrn_antisup_w2_ep19.ckpt` | in-domain **+8.32**; BUT deletion **→0.276**; safest across both domains |
| 5 | anti-suppression weight 3.0 | `config/train_dpcrn_antisup_w3.yaml` | stage 4 ep19 | `dpcrn_antisup_w3_ep19.ckpt` | in-domain **+8.45**; **best in deployment reverb** (moderate enh 0.372, best of all) but **worst in extreme OOD** (BUT enh 0.692, regressed vs w2's 0.676) — "domain split point" |
| 6 | wide-domain deployment (RIR 0.20–0.85 + media_voice/hpf realism) | `config/train_dpcrn_wide_antisup.yaml` | stage 5 ep19 | `dpcrn_wide_antisup_ep19.ckpt` **(current)** | held-out unseen-room **+8.06** (best ever); deployment hard-gate passed (moderate enh 0.373 ≈ w3); BUT still not beaten (0.680, target was <0.676) — 4/5 judge gates passed |

Run (from repo root):
```bash
uv run python egs/voice_isolate/main.py egs/voice_isolate/config/train_dpcrn_curriculum_core.yaml --training
# stage 2+: warm-start from the previous stage's judged checkpoint
uv run python egs/voice_isolate/main.py egs/voice_isolate/config/train_dpcrn_curriculum_expand.yaml --training \
    --pretrained_ckpt_path egs/voice_isolate/pretrained_ckpt/dpcrn_curriculum_core_ep39.ckpt
```
`--ckpt_path <ckpt>` instead of `--pretrained_ckpt_path` = true resume (restores optimizer/scheduler/epoch).
Config details (shared design, eval-only variants, inference config): `config/README.md`.

## Why weight-3.0 isn't an outright winner (the "domain split point")

Escalating `OverSuppressionLoss` weight 1.0→2.0→3.0 monotonically reduced deletion (over-suppression)
and *raised* in-domain SI-SDRi at every step — but at weight 3.0 the two evaluation domains diverged:
**deployment reverb** (rt60 0.44, in-domain) kept improving, while **extreme-OOD reverb** (BUT real
RIR, rt60 1.15–1.84) got *worse* (substitution/insertion rose faster than deletion fell). SI-SDRi alone
would have said "keep pushing" — WER (real intelligibility) said stop. Stage 6 resolved this by
widening the training RIR domain (0.20→0.85) so the OOD boundary moved outward, rather than tuning the
loss weight further.

## Streaming deployment (ONNX, real-time)

`pretrained_ckpt/streaming/dpcrn_wide_antisup_ep19.{onnx,json}` — per-frame streaming export of the
current best checkpoint, built with `scripts/streaming_onnx.py`. The look-ahead (`delay=[1,1,1]`, 30 ms)
is handled by **future-buffering baked into the ONNX graph as extra state** (inter-LSTM warmup gate +
U-Net skip delay lines + noisy-spectrum delay) — no runtime code changes were needed; the existing
manifest-driven SDK runtime (`sdk/python/puresound_streaming`, `processor: stft_frame_ort`) loads it
directly. Verified bit-exact vs the offline model once aligned by the 30 ms latency (SI-SDR 88–105 dB
end-to-end); CPU real-time factor 0.43. See `pretrained_ckpt/README.md` and
`puresound/streaming/dpcrn.py` for the mechanism.

## Pre-DPCRN history (compressed)

Before the pipeline above: TS-Conformer (mapping-head) runs plateaued near passthrough on hard
near/far cases (`mix_mode` curriculum helped but couldn't break through; an explicit far-decoder branch
(P1) didn't help either). An `overfit_sanity.py` capacity test isolated the cause to the backbone, not
the data/loss — TS-Conformer couldn't overfit the hardest batches (enh→target stuck at −2.2 dB) while
DPARN/DPCRN could (+4–5 dB). Switching to DPCRN (complex ratio mask) immediately cleared every
in-domain bucket. A subsequent cold-start-on-full-RIR-bank + `target_absent` combination caused a real-domain
over-suppression catastrophe (Dawn WER 0.626, SI-SDRi −9.53) — this is why the current pipeline is a
graded RIR curriculum with `target_absent: OFF`. Superseded configs for all of this are kept in
`config/backup/` for reproducibility.

## Evaluation

Tools + usage: `scripts/README.md`.

- **Primary (synthetic, in-domain):** `scripts/indomain_sisdri.py --by-bucket` (or `run_valid.sh`).
  SI-SDRi vs the early-reverb target; read the hard buckets (counter_level / 1N+0F / overlap), not the
  aggregate.
- **Real acoustics, deployment-reverb (rt60 0.44):** `config/eval_but_real.yaml`'s sibling set built at
  a moderate RT60 — WER **0.723→0.487 (−32%)**, do-no-harm confirmed. This is the domain the product
  actually ships into.
- **Real acoustics, extreme OOD (BUT real-RIR, rt60 1.15–1.84):** `scripts/build_but_wer_set.py` (build
  once) → `scripts/eval_but_wer.py` (SI-SDRi + WER vs real LibriTTS transcripts), config
  `config/eval_but_real.yaml`. Deliberately harder than training; tracks how far over-suppression is an
  OOD-reverb phenomenon (it is — see above).
- **Unseen-room generalization:** `config/eval_heldout.yaml` (seed-2026 disjoint room bank, same
  difficulty distribution as expand). seen→unseen drop is consistently small (≤0.4 dB) across every
  stage — the model generalizes on DRR/geometry, not memorized rooms.
- **Leakage probe (far-only/noise-only):** `config/eval_targetabsent_probe.yaml` — forces every row
  target-absent; checks the model doesn't leak/hallucinate a near speaker. All pipeline checkpoints
  pass (power reduction ≤ −24.8 dB, false-near ≤3%) without ever training on this scenario.
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
