# voice_isolate configs

繁體中文版本：[`README.zh-TW.md`](README.zh-TW.md)

Everything here is meant to be run as-is.

| config | use |
|---|---|
| `train_dpcrn.yaml` | **default training recipe.** One curriculum run from scratch — no warm-start checkpoint needed. |
| `train_dpcrn_curriculum_v1.yaml` | the lineage's second step; warm-starts from the `train_dpcrn.yaml` run's ep99 checkpoint (`dpcrn_curriculum_v0`, not distributed) and adds session rows, a paired capture view, and per-frame proximity/presence heads. Produced the released `dpcrn_curriculum_v1`. |
| `train_dpcrn_curriculum_v2_base.yaml` | `train_dpcrn.yaml` on the wider model with 1.5x the rows per epoch, 80 epochs from scratch; its ep79 is the warm start of the next row. |
| `train_dpcrn_curriculum_v2.yaml` | `train_dpcrn_curriculum_v1.yaml` on the wider model with 1.5x the rows per epoch, warm-started from the base run's ep79. Produced the released `dpcrn_curriculum_v2`. |
| `infer_dpcrn.yaml` | default inference config; loads `dpcrn_v8` and `dpcrn_curriculum_v1` (every checkpoint in `../pretrained_ckpt/` except `dpcrn_curriculum_v2`) |
| `infer_dpcrn_wide.yaml` | inference config for the wider model: loads `dpcrn_curriculum_v2` |
| `infer_dpcrn_heads.yaml` | same as `infer_dpcrn.yaml`, for a checkpoint that exports VAD side-heads |
| `eval/` | the evaluation configs `../run_full_benchmark.sh` drives |

Before the first run, point the corpus and RIR-bank paths at your own data:
[`../DATA_SETUP.md`](../DATA_SETUP.md).

```bash
# the training configs' data and work-folder paths are relative to the recipe directory
cd egs/voice_isolate

# train from scratch
uv run python main.py config/train_dpcrn.yaml --training

# then, optionally, the second curriculum step, from that run's ep99 checkpoint
uv run python main.py config/train_dpcrn_curriculum_v1.yaml --training \
    --pretrained_ckpt_path exp/dpcrn_curriculum/lightning_logs/version_0/checkpoints/epoch=99-*.ckpt

# inference / demo
uv run python scripts/demo.py --config_path config/infer_dpcrn.yaml
```

`--ckpt_path <ckpt>` instead of `--pretrained_ckpt_path` is a true resume (restores
optimizer/scheduler/epoch). The released inference setting includes `dry_blend 0.9` — see
`../pretrained_ckpt/README.md`.

## Shared design (all configs here)

- **Backbone DPCRN**, complex ratio mask, 16 kHz native: `channels [2,32,64,128]`,
  `rnn_hidden 96`, ~0.8 M params; the two `curriculum_v2` recipes and `infer_dpcrn_wide.yaml`
  widen it to `channels [2,48,96,128]`, `rnn_hidden 128`, ~1.2 M params.
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

## `eval/`

Eval-only configs rebuild the exact augmentation pipeline with the RIR bank or a single flag
swapped, so any checkpoint can be benchmarked. Never used for training.

| config | purpose |
|---|---|
| `eval_but_real.yaml` | measured-RIR benchmark, RT60 1.15–1.84 (far beyond the training domain); the WER stages build their model from its `model:` block |
| `eval_targetabsent_probe.yaml` | far-only / noise-only leakage probe (`augmentation_target_absent` forced ON) |
| `eval_indomain_phase1.yaml` | in-domain SI-SDRi + solo-leakage + turn-taking buckets |
| `eval_targetabsent_probe_high.yaml` | far-only probe on unseen high-reverb rooms |
| `eval_targetabsent_probe_boundary.yaml` | far-only probe on unseen boundary distances |

The last three are **byte-frozen fixtures**: they define the distributions every recorded
judgment used, via `../run_full_benchmark.sh`. Do not edit them. Tools: `../scripts/README.md`.
