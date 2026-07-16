# voice_isolate configs

Full pipeline narrative + results: `../README.md`.

## Training configs (`train_*.yaml`) — the 6-stage pipeline

Each stage warm-starts from the previous stage's judged checkpoint (`pretrained_ckpt/`, see
`../pretrained_ckpt/README.md`). All six share the **same DPCRN architecture** (complex ratio mask,
`channels [2,32,64,128]`, `rnn_hidden 96`, 30 ms look-ahead `delay=[1,1,1]`) — only the RIR bank,
augmentation, and loss weights change stage to stage.

| # | config | warm-start from | judged ckpt |
|---|---|---|---|
| 1 | `train_dpcrn_curriculum_core.yaml`   | cold | `dpcrn_curriculum_core_ep39.ckpt` |
| 2 | `train_dpcrn_curriculum_expand.yaml` | stage 1 ep39 | `dpcrn_curriculum_expand_ep59.ckpt` |
| 3 | `train_dpcrn_antisup_w1.yaml`        | stage 2 ep59 | `dpcrn_antisup_w1_ep19.ckpt` |
| 4 | `train_dpcrn_antisup_w2.yaml`        | stage 3 ep19 | `dpcrn_antisup_w2_ep19.ckpt` |
| 5 | `train_dpcrn_antisup_w3.yaml`        | stage 4 ep19 | `dpcrn_antisup_w3_ep19.ckpt` |
| 6 | `train_dpcrn_wide_antisup.yaml`      | stage 5 ep19 | `dpcrn_wide_antisup_ep19.ckpt` (current best, streaming-verified) |

`train_dpcrn_wide_causal.yaml` — untrained alternative (fully-causal, `delay=[0,0,0]`), kept only as a
documented fallback; not needed since streaming export solved look-ahead via future-buffering instead.

`train_dpcrn_gate.yaml` / `train_dpcrn_v2_sepgate.yaml` — the VAD gate recipes, kept for the future
real-data rung. Both warm-start from `dpcrn_wide_antisup_ep19.ckpt` and add a causal frame-level VAD
head (near-active vs inactive/far-only labels). `gate`: separator frozen, head-only (engineering
validation). `v2_sepgate`: separator+gate trained jointly on the obstacle-rich `hybrid_rir_16k_v2`
bank with turn-taking label-0 frames. **Both were judged negative on real end-to-end recordings**
(gate never closes on real clips; joint training also blows up real-acoustic deletion — see
`EXPERIMENT_LOG.md` 2026-07-10/16); synthetic scores are not evidence of real-recording transfer.

Run (from repo root):
```bash
uv run python egs/voice_isolate/main.py egs/voice_isolate/config/train_dpcrn_curriculum_core.yaml --training
uv run python egs/voice_isolate/main.py egs/voice_isolate/config/train_dpcrn_curriculum_expand.yaml --training \
    --pretrained_ckpt_path egs/voice_isolate/pretrained_ckpt/dpcrn_curriculum_core_ep39.ckpt
uv run python egs/voice_isolate/main.py egs/voice_isolate/config/train_dpcrn_gate.yaml --training \
    --pretrained_ckpt_path egs/voice_isolate/pretrained_ckpt/dpcrn_wide_antisup_ep19.ckpt
```
`--ckpt_path <ckpt>` instead of `--pretrained_ckpt_path` = true resume (restores optimizer/scheduler/epoch).

## Eval-only configs (`eval_*.yaml`)

Same recipe body as the matching training stage, with only the RIR bank / augmentation flag swapped —
reconstruct the exact augmentation pipeline so any checkpoint in `pretrained_ckpt/` can be benchmarked.
**Never used for training.**

| config | purpose | RIR / flag |
|---|---|---|
| `eval_but_real.yaml` | real BUT ReverbDB RIR benchmark (rt60 1.15–1.84, extreme OOD) | `but_real_rir_16k` |
| `eval_heldout.yaml` | unseen-room generalization (same distribution as expand, disjoint rooms) | `hybrid_rir_16k_levels_test/expand` |
| `eval_targetabsent_probe.yaml` | far-only/noise-only leakage probe | `augmentation_target_absent` forced ON |

**Frozen benchmark fixtures** (byte-frozen — do not edit; they define the eval distributions every
historical judgment in `EXPERIMENT_LOG.md` used, via `../run_full_benchmark.sh`):

| config | benchmark station | bank |
|---|---|---|
| `eval_indomain_phase1.yaml` | 2 (in-domain SI-SDRi + solo-leakage + turn-taking buckets) | `hybrid_rir_16k_phase1` (wide+boundary merge) |
| `eval_targetabsent_probe_high.yaml` | 4 (unseen high-reverb far-only probe) | `hybrid_rir_16k_high_levels/all` |
| `eval_targetabsent_probe_boundary.yaml` | 5 (unseen boundary-distance far-only probe) | `hybrid_rir_16k_boundary_heldout_levels/all` |

Tools: `../scripts/README.md`.

## Inference config (`infer_dpcrn.yaml`)

**One shared config for all 6 pipeline checkpoints** — the `model:` block never changed across the
whole pipeline, so there is nothing stage-specific to configure at inference time; pick the checkpoint
via `--ckpt` / `--checkpoint_path` (or the `scripts/demo.py` dropdown, which scans `trainer.work_folder:
pretrained_ckpt` automatically). Used by `scripts/demo.py` and `scripts/streaming_onnx.py`.

## Shared design (all `train_*`/`eval_*`/`infer_*` configs)

- **Backbone DPCRN** (complex ratio mask). TS-Conformer was dropped (couldn't separate hard near/far
  cases).
- **30 ms look-ahead**: `backbone.delay=[1,1,1]` (3 frames), inter-RNN unidirectional → bounded
  look-ahead. Streaming ONNX export handles this via future-buffering (see `../scripts/streaming_onnx.py`).
- **Wider net**: `channels [2,32,64,128]`, `rnn_hidden 96`.
- **Dual-SSL ASR loss** (stages 1–2): two `ASRFeatureLoss` (HuBERT + WavLM, cosine) + `SDRLoss` (SD-SDR)
  + `MultiResolutionSTFTLoss` + `ResidualReferenceLoss`. **Anti-suppression** (stages 3–6): adds
  `OverSuppressionLoss` (pure-magnitude, one-sided; weight escalates 1.0→2.0→3.0 across stages 3–5).
- **Early target** (`target_rir_type: early`), **hard SIR [-10,10]** + `mix_mode`.
- **`target_absent: OFF`** — forced-silent rows drive real-domain over-suppression.
- Fixed near-field: no distance query, no FiLM, no `gate_silence`.
- **Scheduler `CosineAnnealingWarmRestarts T_0=20`** — restarts ~ep20/40/…; **compare only at the
  cosine troughs (ep19/ep39/ep59)**. Optimizer trains the backbone only (encoder/features frozen).

## `backup/`

Superseded configs kept only for reproducibility (dead-end runs: conformer/mixmode/P1, pre-DPCRN
query/FiLM/VAD variants, the failed all-RIR cold-start, ASR-loss ablations, the closed
boundary/realfar rungs, the never-run `dpcrn_curriculum_stress.yaml`, and the never-wired
`eval_targetabsent_probe_wide.yaml`). Not part of the active pipeline. Note: the corresponding
library code for the conformer/distance-query axis was removed from `puresound/` in the 2026-07-16
refactor, so these configs document history rather than runnable recipes.
