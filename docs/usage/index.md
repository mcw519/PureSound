# Using PureSound

Traditional Chinese: [index.zh-TW.md](index.zh-TW.md)

The workflow from a fresh clone to a deployed model, in the order you meet it.
Each step links to the page that covers it.

## 1. Install

Python 3.12 or newer, with [uv](https://docs.astral.sh/uv/). Pick one ONNX
Runtime backend (the two cannot be installed together):

```bash
uv sync --locked --group dev --extra cpu     # CPU (CoreML on macOS)
uv sync --locked --group dev --extra cuda    # NVIDIA CUDA
```

Optional extras: `asr` (local Whisper for the WER stages and the web word check;
not combinable with `cuda`), `hybrid-rir` / `hybrid-rir-gpu` (the RIR generator's
optional backends). Run repository tools with `uv run` or `.venv/bin/python`, not
another interpreter on `PATH`. More in the [project README](../../README.md).

## 2. Prepare data

| What | Page |
|---|---|
| speech metafiles, noise folders, pooling corpora, cleaned targets, and the recipe knobs that mix them | [data_preparation.md](data_preparation.md) |
| the voice-isolation recipe's full data chain (chapter corpus, real-recording pools, RIR-bank views, measured RIRs) | [egs/voice_isolate/DATA_SETUP.md](../../egs/voice_isolate/DATA_SETUP.md) |
| generating and packaging synthetic RIR banks | [egs/rir_generation/README.md](../../egs/rir_generation/README.md) |

## 3. Configure a recipe

A recipe is one YAML file validated before anything is built: task, dataset,
augmentation blocks, sampler, model, losses, optimiser, scheduler, and an
optional `curriculum` block that moves knobs by epoch.
[configuration.md](configuration.md) covers the schema and its rules;
[recipes.md](recipes.md) how `model` and `loss_func` become objects.

## 4. Train

Each task has its own entry point over one shared driver
(`puresound/system/runner.py`):

| Recipe | Entry point | Released lineage |
|---|---|---|
| [noise suppression](../../egs/noise_suppression/README.md) | `egs/noise_suppression/main.py` | four recipes run in order, each warm-started from the last |
| [voice isolation](../../egs/voice_isolate/README.md) | `egs/voice_isolate/main.py` | one scheduled curriculum run, then a warm-started second step |

```bash
uv run python egs/noise_suppression/main.py <recipe.yaml> --dump_training_samples   # listen first
uv run python egs/noise_suppression/main.py <recipe.yaml> --training
```

**Curriculum ladders** come in two forms. A *ladder of runs* changes the recipe
between runs and warm-starts each from the previous one's last checkpoint; a
*`curriculum` block* writes the same changes as epoch schedules inside one run.
Both are used by the released models -- see each recipe's README.

**Resume or warm start.** `--ckpt_path` resumes the same run under the same
config (optimiser, scheduler and epoch restored). `--pretrained_ckpt_path`
warm-starts a new run from another checkpoint's weights: names are matched
non-strictly (new parameters stay at init, stale ones are ignored, both logged)
and shapes strictly -- a changed shape is refused unless you pass
`--pretrained_allow_reshaped`, meant for a deliberate STFT window change, which
rebuilds those parameters at init. The flags are documented in
[egs/noise_suppression/README.md](../../egs/noise_suppression/README.md#resume-warm-start-reload).

Compare checkpoints at the scheduler's cosine troughs, and as a block of a run's
last checkpoints rather than one.

## 5. Evaluate: the gate

A checkpoint is released only through its recipe's gate. Every stage scores the
candidate against doing nothing and reads the paired difference; gate stages can
fail a release, monitors only report.

```bash
bash egs/noise_suppression/run_full_gate.sh baseline
bash egs/noise_suppression/run_full_gate.sh <tag> <ckpt> <recipe>
```

| Page | Covers |
|---|---|
| [evaluation.md](evaluation.md) | building the sets, running the gate, its variables, records and verdicts |
| [egs/noise_suppression/benchmarks/stages.md](../../egs/noise_suppression/benchmarks/stages.md) | what each stage can and cannot decide |
| [egs/voice_isolate/README.md](../../egs/voice_isolate/README.md#benchmark) | the voice-isolation benchmark, `run_full_benchmark.sh`; its field-recording stages use private recordings that are not distributed with the repository |

## 6. Export and deploy

Checkpoints deploy as per-frame ONNX graphs with a JSON manifest; the runtime owns
STFT, state, and the post-graph stages the manifest records (dry blend, onset
guard).

```bash
uv run python egs/voice_isolate/scripts/streaming_onnx.py export <infer.yaml> <ckpt> model.onnx
uv run python egs/voice_isolate/scripts/streaming_onnx.py verify <infer.yaml> <ckpt> model.onnx \
    --input_audio speech.wav
```

| Page | Covers |
|---|---|
| [streaming/index.md](streaming/index.md) | the streaming pages |
| [streaming/dpcrn_onnx.md](streaming/dpcrn_onnx.md) | export, verify, look-ahead, the runtime, auxiliary heads |
| [sdk/python](../../sdk/python/README.md) | the portable runtime: NumPy and ONNX Runtime only |

## 7. Web playground

`uv run puresound web` serves a local UI at <http://127.0.0.1:7860>: run models on
files, recordings or a live microphone, compare them with scores and word checks,
and inspect the model zoo. See [web.md](web.md), including the HTTP API.

## 8. Model zoo

`model_zoo/catalog.yaml` registers the released models; each entry points at an
ONNX export and its manifest under a recipe's `pretrained_ckpt/streaming/`.

```bash
uv run puresound models list
uv run puresound models validate
uv run puresound infer <model-id> --input audio=in.wav --output audio=out.wav --provider auto
```

Why each version was released, and how to choose between them, is the
judgement table in each recipe's `pretrained_ckpt/README.md`
([noise suppression](../../egs/noise_suppression/pretrained_ckpt/README.md),
[voice isolation](../../egs/voice_isolate/pretrained_ckpt/README.md)). How run
directories and catalog ids are named, and which artifacts are released:
[repository_layout.md](../repository_layout.md).

Explore moving sources in the [acoustic world](world.md), or run models in
the browser with Playground's *This device* ([Web SDK](../../sdk/web/README.md)).
