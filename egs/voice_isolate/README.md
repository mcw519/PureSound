# voice_isolate — near-field foreground voice isolation

Traditional Chinese: [`README.zh-TW.md`](README.zh-TW.md)

Single-channel, **enrollment-free, near-field** voice isolation: keep the speaker
within about 1 m of the microphone, and suppress farther speakers and noise. There
is no enrollment and no second microphone, so the cue the model has to learn is
the difference distance makes to a voice -- above all the near/far
direct-to-reverberant ratio contrast.

Every released model is a DPCRN (complex ratio mask, 16 kHz) with a 30 ms
(3-frame) look-ahead, streamed through a per-frame ONNX export: `dpcrn_v8` and
`dpcrn_curriculum_v1` at `channels [2,32,64,128]`, `rnn_hidden 96`, and the wider
`dpcrn_curriculum_v2` at `channels [2,48,96,128]`, `rnn_hidden 128`.

`main.py` is this recipe's training entry point (`VoiceIsolationDataset`, with
the real-recording row types no other recipe has). Everything downstream of the
dataset is shared with [`egs/noise_suppression`](../noise_suppression/README.md)
through `puresound/system/runner.py`; that README documents the command-line
flags, resume versus warm start (including `--pretrained_allow_reshaped`), DDP,
precision and VAD labelling, all of which apply here unchanged.

## Released models

| Checkpoint | Zoo id / role | Use it when |
|---|---|---|
| **`pretrained_ckpt/dpcrn_curriculum_v1.ckpt`** | `voice-isolate-dpcrn-curriculum-v1`, **default** | the capture path is the one the model was tuned for; it suppresses a distant talker furthest, including from a cold start |
| `pretrained_ckpt/dpcrn_curriculum_v2.ckpt` | `voice-isolate-dpcrn-curriculum-v2`, candidate | you can afford 1.4x the CPU of v1: a wider model on the same recipe, with lower WER on the reverberant sets and deeper cold-start suppression; near-talker keep on unfamiliar capture hardware is no better than v1's |
| `pretrained_ckpt/dpcrn_v8.ckpt` | `voice-isolate-dpcrn-v8`, candidate | the capture hardware is unknown or unlike the training corpora: it suppresses less, but does not attenuate the near talker there |

All three ship with a **runtime dry blend of 0.9** (`out = 0.9 * enhanced + 0.1 *
input`), which bounds attenuation at -20 dB and trades a little residual
interferer for far fewer deleted words on unfamiliar capture chains. The streaming
manifests record it under `recommended_inference`, and the runtimes apply it.

```bash
uv run puresound infer voice-isolate-dpcrn-curriculum-v1 \
    --input audio=in.wav --output audio=out.wav --provider auto

# Gradio demo over every checkpoint and export (run from the recipe directory;
# pass config/infer_dpcrn_wide.yaml to load dpcrn_curriculum_v2)
cd egs/voice_isolate && uv run python scripts/demo.py --config_path config/infer_dpcrn.yaml
```

Per-version judgements and how to choose:
[`pretrained_ckpt/README.md`](pretrained_ckpt/README.md).

## How they were trained

Two lineages, one architecture at two widths.

| Lineage | Recipe(s) | Warm start | Produces |
|---|---|---|---|
| curriculum | `config/train_dpcrn.yaml`, 120 epochs | from scratch | `dpcrn_curriculum_v0` (ep99; not distributed) |
| | `config/train_dpcrn_curriculum_v1.yaml`, 40 epochs | `dpcrn_curriculum_v0` | **`dpcrn_curriculum_v1`** |
| curriculum, wide | `config/train_dpcrn_curriculum_v2_base.yaml`, 80 epochs, then `config/train_dpcrn_curriculum_v2.yaml`, 40 epochs | from scratch, then the base run's ep79 | `dpcrn_curriculum_v2` |
| ladder | `dpcrn_v8` came from an unpublished multi-stage ladder of warm-started runs | each from the previous stage | **`dpcrn_v8`** |

`config/train_dpcrn.yaml` writes what the ladder changed *between* runs as
`curriculum` schedules that move *during* one run: the room pool widens, the
anti-suppression loss ramps in, capture realism and the real-recording rows arrive
once the synthetic decision is learned, and the distance loss arrives with them.
`config/train_dpcrn_curriculum_v1.yaml` keeps that data recipe at its end values
and adds one axis on a schedule: multi-turn session rows, a paired capture view
for a consistency term, and per-frame proximity and presence heads. The two
`curriculum_v2` recipes are those two steps on a wider model (channels
`[2, 48, 96, 128]`, `rnn_hidden` 128) with 1.5x the rows per epoch. The
recipes' headers explain each choice.

## Train

Before the first run, point the recipe's corpus and RIR-bank paths at your own
data: [`DATA_SETUP.md`](DATA_SETUP.md). Run **from this directory** -- the configs'
metafile and work-folder paths are relative to it:

```bash
cd egs/voice_isolate

# check the data the recipe will train on, and that the pipeline can learn at all
uv run python scripts/check_training_data.py config/train_dpcrn.yaml --n 64 --dump 8
uv run python scripts/overfit_check.py config/train_dpcrn.yaml --steps 800 --device cuda

# the default recipe: one curriculum run from scratch
uv run python main.py config/train_dpcrn.yaml --training

# the second step, warm-started from your first run's ep99 checkpoint
# (dpcrn_curriculum_v0 was taken there; it is not distributed)
uv run python main.py config/train_dpcrn_curriculum_v1.yaml --training \
    --pretrained_ckpt_path exp/dpcrn_curriculum/lightning_logs/version_0/checkpoints/epoch=99-*.ckpt

# the wide lineage: the same two steps on the wider model
uv run python main.py config/train_dpcrn_curriculum_v2_base.yaml --training --set_seed 1234
uv run python main.py config/train_dpcrn_curriculum_v2.yaml --training --set_seed 1234 \
    --pretrained_ckpt_path exp/dpcrn_curriculum_v2_base/lightning_logs/version_0/checkpoints/epoch=79-step=60000.ckpt
```

Use `--pretrained_ckpt_path` for the second step, not `--ckpt_path`: the model
gains heads, so the checkpoint loads non-strict and the optimiser starts fresh.
Resume an *interruption* of either run with `--ckpt_path` on its own latest
checkpoint.

The scheduler (`CosineAnnealingWarmRestarts`, `T_0=20`) restarts every 20
epochs, so compare checkpoints only at the cosine troughs -- epochs 19, 39, 59,
... -- and score a block of the last few before believing a difference between
runs. Both recipes set `trainer.find_unused_parameters: true` for their
auxiliary heads, and `precision: bf16-mixed`.

Shared design of the configs (early target, hard SIR, measured-capture realism,
no synthetic target-absent rows, the anti-deletion losses):
[`config/README.md`](config/README.md).

## Benchmark

```bash
cd egs/voice_isolate
bash run_full_benchmark.sh <ckpt> <tag> [device] [dry_blend] [presence_readout]
```

Pass `dry_blend 0.9` to score a checkpoint the way it is deployed (the default
1.0 turns the blend off). Training must be stopped first; the stages use the GPU.
`BLOCK_CKPTS="a.ckpt b.ckpt ..."` adds the block protocol over a run's last
checkpoints; `CFG_*` overrides point the stages at recipes matching a
non-default backbone (`scripts/make_arch_eval_configs.py` writes them).

| # | Stage | Role |
|---|---|---|
| 0 | preflight: every stage's recipe loads the checkpoint whole | abort |
| 1 | real-recording field scorecard, including cross-chain reference clips (private recordings) | gate |
| 1b | field block protocol over `BLOCK_CKPTS` (optional; private recordings) | how a field difference is believed |
| 2 | in-domain SI-SDRi by bucket, with solo leakage | synthetic, same synthesis chain only |
| 3–5 | synthetic far-only probes: seen distances, unseen high reverb, unseen boundary distances | synthetic leakage checks |
| 6 | Dawn Chorus WER | deletion guardrail |
| 7a | moderate-reverb WER | primary WER gate |
| 7b | BUT-OFFICE measured-RIR WER | monitor: too few utterances to separate a model from no processing |
| 8 | BUT high-reverb WER | do-no-harm monitor, far outside the training domain |
| 9 | real-RIR turn-taking: keep near, suppress far solo | keep / suppress scorecard |

Stages 1 and 1b read a field set of private recordings
(`data_report/field_cases/test_vector_cases`, which also holds the cross-chain
reference clips). It is not distributed, so skip those stages: without the set,
stage 1 is reported as failed and every other stage still runs.

Stages 2–5 synthesise their audio at evaluation time, so they compare only
against records made on the same synthesis chain (the summary prints the
commit); the others read fixed audio from disk. The evaluation sets live under
`data_report/`, which is not tracked. Tools, set construction and the scripts
each stage calls: [`scripts/README.md`](scripts/README.md) and
[`scripts/WER_SETS.md`](scripts/WER_SETS.md).

## Streaming deployment

`pretrained_ckpt/streaming/` holds the per-frame ONNX exports of all three released
checkpoints, built with `scripts/streaming_onnx.py` (export `dpcrn_curriculum_v2`
with `config/infer_dpcrn_wide.yaml`). The 30 ms look-ahead is
handled by future-buffering inside the graph as extra state, and the
manifest-driven runtimes -- `puresound.streaming.StreamingOrt` and the portable SDK
-- load it directly and apply the recorded dry blend after the graph:

```bash
uv run python scripts/streaming_onnx.py export \
    config/infer_dpcrn.yaml pretrained_ckpt/dpcrn_curriculum_v1.ckpt /path/to/model.onnx
uv run python scripts/streaming_onnx.py verify \
    config/infer_dpcrn.yaml pretrained_ckpt/dpcrn_curriculum_v1.ckpt \
    pretrained_ckpt/streaming/dpcrn_curriculum_v1.onnx --input_audio speech.wav
```

`export` records `--dry-blend 0.9` by default, the released setting. Details --
look-ahead state, the onset guard, auxiliary-head outputs, the one-hop lag rule:
[`docs/usage/streaming/dpcrn_onnx.md`](../../docs/usage/streaming/dpcrn_onnx.md).

## Docs

| File | Content |
|---|---|
| [`DATA_SETUP.md`](DATA_SETUP.md) | from public corpora to the paths the default recipe reads |
| [`config/README.md`](config/README.md) | the configs, and the design they share |
| [`pretrained_ckpt/README.md`](pretrained_ckpt/README.md) | the version judgement table, how to choose, streaming exports |
| [`scripts/README.md`](scripts/README.md) | data preparation, training-time checks, benchmarks and inference tools |
| [`docs/architecture/task/voice_isolation.md`](../../docs/architecture/task/voice_isolation.md) | `VoiceIsolationDataset`, the real-recording row types and the labels it emits |
