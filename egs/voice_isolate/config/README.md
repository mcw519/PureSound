# voice_isolate configs

繁體中文版本：[`README.zh-TW.md`](README.zh-TW.md)

Two files here are the **defaults**: the tuned settings to use as-is. Everything under
`exp/` is the recipe history — earlier pipeline stages, ablations and eval-only fixtures.

| config | use |
|---|---|
| `train_dpcrn.yaml` | default training recipe (produced the released `dpcrn_v8` checkpoint) |
| `infer_dpcrn.yaml` | default inference config; loads any checkpoint in `../pretrained_ckpt/` |

```bash
# train (from repo root), warm-starting from the previous release
uv run python egs/voice_isolate/main.py egs/voice_isolate/config/train_dpcrn.yaml --training \
    --pretrained_ckpt_path egs/voice_isolate/pretrained_ckpt/backup/dpcrn_v7.ckpt

# inference / demo
uv run python egs/voice_isolate/scripts/demo.py \
    --config_path egs/voice_isolate/config/infer_dpcrn.yaml
```

`--ckpt_path <ckpt>` instead of `--pretrained_ckpt_path` is a true resume (restores
optimizer/scheduler/epoch). The released inference setting includes `dry_blend 0.9` — see
`../pretrained_ckpt/README.md`.

## Shared design (all configs here)

- **Backbone DPCRN**, complex ratio mask, `channels [2,32,64,128]`, `rnn_hidden 96`, ~0.8 M
  params, 16 kHz native.
- **30 ms look-ahead**: `backbone.delay=[1,1,1]` (3 frames), inter-RNN unidirectional, so the
  look-ahead is bounded. The streaming ONNX export handles it with future-buffering baked into
  the graph (`../scripts/streaming_onnx.py`).
- **Early target** (`target_rir_type: early`): the target is the de-reverberated near speech, so
  passthrough cannot match it and separation stays a real objective.
- **Hard SIR** `[-10, 10]` plus `mix_mode`: the foreground may be up to 10 dB *quieter* than the
  interferer, so loudness alone cannot solve the task.
- **Measured-capture realism ON** (default recipe only): noise convolved with the speech's own
  room, an absolute dBFS microphone floor, and `mix_mode distance_level` drawing SIR from the
  scene geometry. Synthesis without these is implausibly clean next to real recordings, which
  teaches "clean means suppressible" instead of "far means suppressible".
- **Synthetic `target_absent`: OFF.** Forced-silent rows teach "emit silence when unsure", which
  mis-fires outside the training domain. Absolute-suppress supervision comes from real-recording
  rows instead (`augmentation_realfar.lone_far_prob`, turn-taking).
- **Anti-deletion losses**: `OverSuppressionLoss` + two `ASRFeatureLoss` terms (HuBERT + WavLM,
  cosine) + `SDRLoss` + `MultiResolutionSTFTLoss` + `ResidualReferenceLoss`.
- **Scheduler `CosineAnnealingWarmRestarts T_0=20`** — restarts every 20 epochs, so compare
  checkpoints only at the cosine troughs (ep19/ep39/ep59). The optimizer trains the backbone;
  encoder/features stay frozen.

## `exp/`

Recipe history, kept so any stage can be reproduced, re-judged or re-warm-started. Not needed
to train or run the default model.

**Pipeline stages** (each warm-starts from the previous stage's checkpoint; the version table in
`../pretrained_ckpt/README.md` maps recipe → checkpoint → result):
`train_dpcrn_curriculum_core.yaml` → `train_dpcrn_curriculum_expand.yaml` →
`train_dpcrn_antisup_w1.yaml` → `_w2` → `_w3` → `train_dpcrn_wide_antisup.yaml`, then the
real-recording rounds `train_dpcrn_realE2E.yaml` → `_v2` → `_v2b` → `_v2c` (which produced
`dpcrn_v7`), and then the realism round promoted to `../train_dpcrn.yaml` (`dpcrn_v8`).

**Alternatives and side branches:**

| config | what it is |
|---|---|
| `train_dpcrn_realE2E_v2c.yaml` | the recipe that produced `dpcrn_v7`; superseded as the default by the realism settings, kept reproducible |
| `train_dpcrn_wide_causal.yaml` | fully-causal variant (`delay=[0,0,0]`), zero look-ahead; untrained, kept as a documented fallback since future-buffering solved streaming without it |
| `train_dpcrn_gate.yaml` | separator frozen, trains only the causal frame-level VAD gate head |
| `train_dpcrn_v2_sepgate.yaml` | separator + gate head trained jointly on an obstacle-rich RIR bank |

The gate recipes learn the gate well on simulated data, but the gate does not close on real
recordings and pushing far-suppression through joint training raises deletion on real audio;
treat simulated gate scores as engineering signal only.

**Eval-only configs** (`eval_*.yaml`) rebuild the exact augmentation pipeline with the RIR bank
or a single flag swapped, so any checkpoint can be benchmarked. Never used for training.

| config | purpose |
|---|---|
| `eval_but_real.yaml` | measured-RIR benchmark, RT60 1.15–1.84 (far beyond the training domain) |
| `eval_heldout.yaml` | unseen-room generalization, same distribution as the expand stage |
| `eval_targetabsent_probe.yaml` | far-only / noise-only leakage probe (`augmentation_target_absent` forced ON) |
| `eval_indomain_phase1.yaml` | in-domain SI-SDRi + solo-leakage + turn-taking buckets |
| `eval_targetabsent_probe_high.yaml` | far-only probe on unseen high-reverb rooms |
| `eval_targetabsent_probe_boundary.yaml` | far-only probe on unseen boundary distances |

The last three are **byte-frozen fixtures**: they define the distributions every recorded
judgment used, via `../run_full_benchmark.sh`. Do not edit them. Tools: `../scripts/README.md`.

`exp/backup/` — superseded configs kept only for reproducibility (dead-end runs, pre-DPCRN
query/FiLM/VAD variants, ASR-loss ablations, closed rounds). The library code for the
conformer/distance-query axis has since been removed from `puresound/`, so those files document
history rather than runnable recipes.
