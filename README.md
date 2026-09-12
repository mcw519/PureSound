# PureSound

PureSound is a PyTorch toolkit for speech enhancement, voice isolation, speaker
verification, and room impulse response (RIR) generation.

Traditional Chinese: [README.zh-TW.md](README.zh-TW.md)

## What is included

- Training recipes for noise suppression, voice isolation, and speaker embeddings
- Released ONNX models for voice isolation and speaker verification
- A command-line interface and local web UI for inference
- A small Python SDK for embedding ONNX inference in another project
- Audio DSP, augmentation, metrics, and RIR generation tools
- Config-driven PyTorch Lightning training

The target-speaker-extraction recipe is kept for compatibility but is no longer
actively developed.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for the recommended setup

## Install

Clone the repository, then choose one ONNX Runtime backend:

```bash
git clone <project-url>
cd PureSound

# CPU ONNX Runtime, or CoreML on macOS
uv sync --locked --group dev --extra cpu

# NVIDIA CUDA ONNX Runtime
uv sync --locked --group dev --extra cuda
```

Do not install the `cpu` and `cuda` extras together. The `asr` extra uses the
CPU ONNX Runtime package and cannot be combined with `cuda`.

To install with pip instead:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## Run inference

List the released models and available execution providers:

```bash
uv run puresound models list
uv run puresound providers
```

### Voice isolation

Input audio is converted to mono and resampled to 16 kHz when needed.

```bash
uv run puresound infer voice-isolate-dpcrn-v8 \
  --input audio=input.wav \
  --output audio=output.wav \
  --provider auto
```

Two voice-isolation models are included:

- `voice-isolate-dpcrn-v8`: safer when the recording device is unknown
- `voice-isolate-dpcrn-curriculum-v1`: stronger far-voice suppression on its
  target capture setup

See [the checkpoint notes](egs/voice_isolate/pretrained_ckpt/README.md) for the
trade-offs.

### Speaker verification

```bash
uv run puresound infer speaker-verification-ps-spk-v1-1 \
  --input enrollment=enrollment.wav \
  --input test=test.wav \
  --output enrollment_embedding=enrollment.npy \
  --output test_embedding=test.npy \
  --provider auto
```

The command prints the cosine similarity and pass/fail verdict.

### Web UI

```bash
uv run puresound web
```

Open <http://127.0.0.1:7860>. The UI provides voice isolation, speaker
verification, model inspection, and audio measurements. It has no built-in
authentication; do not expose it to an untrusted network.

API details are in [docs/web.md](docs/web.md).

## Train a model

Runnable recipes live under `egs/`. Update the dataset and output paths in a
recipe before training.

| Task | Start here |
| --- | --- |
| Noise suppression | [egs/noise_suppression/README.md](egs/noise_suppression/README.md) |
| Voice isolation | [egs/voice_isolate/README.md](egs/voice_isolate/README.md) |
| Speaker embedding | [egs/speaker_embedding/README.md](egs/speaker_embedding/README.md) |
| Target speaker extraction | [egs/target_speaker_extraction/README.md](egs/target_speaker_extraction/README.md) |

For example, to train the default voice-isolation recipe from scratch:

```bash
uv run python egs/voice_isolate/main.py \
  egs/voice_isolate/config/train_dpcrn.yaml \
  --training
```

## Generate RIR training data

The public RIR pipeline creates deterministic room banks with per-item quality
checks:

```bash
PYTHONPATH=. .venv/bin/python egs/rir_generation/generate_m6_bank.py \
  --output-dir /path/to/rir-bank \
  --backend path-events-m4 \
  --n-rooms 1000 \
  --rir-per-room 4 \
  --num-workers 8 \
  --seed 1337 \
  --sample-rate 16000
```

See [egs/rir_generation/README.md](egs/rir_generation/README.md) for backends,
output formats, and optional dependencies.

## Portable ONNX runtime

[`sdk/python`](sdk/python) contains the standalone inference runtime. It only
needs NumPy, ONNX Runtime, an ONNX model, and its JSON manifest. Use it when the
full training package is not required.

## Development

Run the standard checks:

```bash
uv run python test/run_repo_checks.py --suite standard
```

Run the full suite before a release:

```bash
uv run python test/run_repo_checks.py
```

Build the package:

```bash
./build_puresound.sh
```

## Repository layout

```text
puresound/   Python package
egs/         Training, evaluation, and RIR recipes
docs/        Technical documentation
model_zoo/   Released-model catalog
sdk/python/  Standalone ONNX runtime
test/        Tests and public test fixtures
```

## Documentation

- [Documentation index](docs/index.md)
- [Configuration](docs/configuration.md)
- [Audio and RIR](docs/audio/index.md)
- [Data augmentation](docs/augmentation/index.md)
- [Streaming inference](docs/streaming/index.md)
- [Neural networks](docs/nnet/index.md)
- [Training systems](docs/system/index.md)

Most documentation is available in English and Traditional Chinese.
