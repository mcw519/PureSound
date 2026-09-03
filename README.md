# PureSound

PureSound is a speech processing toolkit based on PyTorch and PyTorch Lightning.
It provides reusable audio utilities, model components, and training recipes for:

- Noise Suppression (NS)
- Near-field Voice Isolation
- Speaker Embedding / Speaker Verification (SV)
- Target Speaker Extraction (TSE, legacy)
- Room Impulse Response (RIR) generation for training-data augmentation

繁體中文版本：[`README.zh-TW.md`](README.zh-TW.md)

## Highlights

- Modular design: `audio`, `dataset`, `nnet`, `system`, and `task`
- Config-driven training and inference (YAML)
- A model library that keeps every backbone config-reachable (DPCRN, DPARN, DPRNN, SkiM, Conv-TasNet, TF-GridNet, ECAPA-TDNN), even when only some are used by an active recipe
- Streaming **DPCRN** ONNX Runtime deployment for real-time voice-isolate inference (the released deployment path; DPARN streaming is kept as a legacy alternative)
- A standalone `sdk/python` runtime for embedding streaming inference in another project without the full training package
- A public RIR generation pipeline (`egs/rir_generation`) for near/far augmentation training data
- Built-in objective and subjective metrics (for example: PESQ, STOI, SDR-related tools)

## Requirements

- Python 3.10+
- A working PyTorch environment compatible with your platform

## Installation

### Option 1: Use uv (recommended for development)

```bash
git clone <project-url>
cd PureSound
uv sync --locked --group dev
```

The repository pins `torch`, `torchaudio`, and `torchcodec` to the PyTorch CUDA 12.4 wheel index in `pyproject.toml` and `uv.lock`, so the same `uv` setup can be reused on another machine without re-resolving to a different CUDA build.

### Option 2: Use pip

```bash
git clone <project-url>
cd PureSound
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`requirements.txt` mirrors the runtime dependencies and includes the PyTorch CUDA 12.4 wheel index for `pip`.

## Quick Validation

Run the test suite:

```bash
uv run pytest
```

Or with pip environment:

```bash
pytest
```

## Web Inference Workspace

The local Model Zoo and unified ONNX runtime are also available through a
small dependency-light browser workspace. It uses the same named-input
processors as the CLI and Gradio compatibility demos:

```bash
puresound web
# open http://127.0.0.1:7860
```

Use `puresound web --ip 0.0.0.0 --port 8080` to control the bind address and
port when serving on a trusted network.

See [`docs/web.md`](docs/web.md) for the API contract and upload format.

## Quick Start Recipes

The `egs` folder contains runnable examples.

### 1) Noise Suppression

```bash
cd egs/noise_suppression

# Prepare training/validation manifests
uv run python prepare_metafile.py --help

# Train
uv run python main.py --training True config/dpcrn.yaml

# Inference
uv run python main.py --inference True --ckpt_path /path/to/model.ckpt config/dpcrn.yaml
```

More detail (the shared `dataset.task` switch between noise-suppression and voice-isolation training, DDP/precision flags, VAD labeling): `egs/noise_suppression/README.md`.

### 2) Speaker Embedding / Verification

```bash
cd egs/speaker_embedding

# Prepare metadata
uv run python prepare_metafile.py --help

# Train
uv run python main.py --training True conf/PS-spk-v1.yaml

# Inference (extract embeddings)
uv run python main.py --inference True --ckpt_path /path/to/model.ckpt conf/PS-spk-v1.yaml
```

More speaker embedding details and pretrained checkpoints are documented in:

- `egs/speaker_embedding/README.md`

### 3) Target Speaker Extraction (legacy)

Frozen legacy recipe: no new features, no rewrites. Kept working for reference only.

```bash
cd egs/target_speaker_extraction

# Prepare metadata
uv run python prepare_metafile.py --help

# Train
uv run python main.py --training True config/default_config.yaml

# Inference
uv run python main.py --inference True --ckpt_path /path/to/model.ckpt config/default_config.yaml
```

### 4) Voice Isolate Streaming ONNX

Train or fine-tune the 16 kHz DPCRN voice-isolate recipe, then export a
feature-frame ONNX model. Example using the `egs/voice_isolate` recipe and its
current default checkpoint:

```bash
uv run python egs/voice_isolate/scripts/streaming_onnx.py export \
  egs/voice_isolate/config/infer_dpcrn.yaml \
  egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt \
  /path/to/model.onnx
```

Run streaming ONNX inference:

```bash
uv run python egs/voice_isolate/scripts/streaming_onnx.py infer \
  /path/to/model.onnx \
  input.wav \
  output.wav \
  --provider auto
```

`dpcrn_v8` ships with its runtime dry/wet blend baked into the exported graph
(`out = 0.9 * enhanced + 0.1 * input`), so no extra post-processing is needed at
inference time. Full pipeline history, per-version results, and deployment
notes: `egs/voice_isolate/README.md`. The (legacy) DPARN streaming path is
documented separately in `docs/streaming/dparn_onnx.md`.

For deployment in another project without the full PureSound training package,
install or copy the portable runtime in `sdk/python`. It only requires NumPy,
ONNX Runtime, `model.onnx`, and `model.json`.

### 5) RIR Generation (training-data augmentation)

Generate the recommended M6 training bank (deterministic generation + per-item
QC + release packaging in one command):

```bash
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/generate_m6_bank.py \
  --output-dir egs/rir_generation/exp/rir_realism/m6/training_pilot \
  --backend path-events-m4 \
  --n-rooms 1000 \
  --rir-per-room 4 \
  --num-workers 8 \
  --seed 1337 \
  --sample-rate 16000
```

Always run with `.venv/bin/python` (or an environment with `pyroomacoustics`/
`rir_generator` installed and a matching numpy ABI) — a mismatched interpreter
produces collection errors that look like code bugs but are not. Full usage
and the algorithm-to-code reference: `egs/rir_generation/README.md` and
`docs/audio/index.md`.

## Repository Structure

```text
PureSound/
├── puresound/                 # Core library
│   ├── audio/                 # Audio I/O, DSP, augmentation, RIR generation (audio/rir/)
│   ├── dataset/                # Dataset base classes and parsers
│   ├── nnet/                   # Model architectures and building blocks
│   ├── streaming/               # Streaming inference and ONNX Runtime utilities
│   ├── system/                 # Lightning training systems
│   ├── task/                   # Task-specific dataset logic
│   ├── third_party/             # Vendored third-party research code (pytARD)
│   ├── metrics.py               # Evaluation metrics
│   ├── recipes.py               # Model/loss initialization helpers
│   └── utils.py                 # General utilities
├── egs/                        # End-to-end recipes and configs
├── docs/                       # API and module documentation
├── sdk/                        # Portable inference SDKs for external projects
└── test/                       # Unit tests
```

## Documentation

- Main docs entry: `docs/index.md`
- Audio modules (incl. RIR generation): `docs/audio/index.md`
- Neural network modules: `docs/nnet/index.md`
- System modules: `docs/system/index.md`
- Streaming runtimes: `docs/streaming/index.md`

Every document under `docs/`, `egs/`, and the package-level `README.md`s ships
in both English (`name.md`) and Traditional Chinese (`name.zh-TW.md`).

## Build Package

```bash
./build_puresound.sh
```

This script runs:

- `uv sync --group dev`
- `uv sync --locked --group dev`
- `uv build`

## Notes

- Some recipe scripts use boolean CLI flags in the form `--training True` or `--inference True`.
- Please adjust dataset paths and output folders in each YAML config before training.

## Troubleshooting

### 1) `ModuleNotFoundError: No module named 'puresound'`

Cause:
- The package is not installed in your active environment.

Fix:

```bash
uv sync --locked --group dev
# or
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

### 2) `OSError` or backend errors from `torchaudio`

Cause:
- PyTorch / torchaudio binary mismatch, or missing runtime codec/backend support.

Fix:

```bash
python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__)"
```

Make sure `torch` and `torchaudio` are installed from compatible channels/versions.
For the repository-managed `uv` flow, they are pinned to the PyTorch CUDA 12.4 wheel index and should be installed with the checked-in lock file.

### 3) CUDA not available (`torch.cuda.is_available() == False`)

Cause:
- CPU-only environment, unsupported CUDA runtime, or mismatched PyTorch build.

Fix:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

If CUDA is required, reinstall a CUDA-enabled PyTorch build matching your system.
This repository's reusable `uv` setup targets CUDA 12.4.

### 4) Recipe starts but cannot find data files

Cause:
- Dataset paths in YAML are not updated for local machine.

Fix:
- Edit each recipe config in `egs/*/config` or `egs/speaker_embedding/conf`.
- Run `prepare_metafile.py` first to generate manifests.

### 5) RIR generation: collection errors or validator failures that look like code bugs

Cause:
- Running `egs/rir_generation` scripts with a Python interpreter that lacks
  `pyroomacoustics`/`rir_generator`, or whose numpy ABI doesn't match.

Fix:
- Always invoke RIR generation scripts with `.venv/bin/python` (or an
  equivalent environment with those packages installed).

## Minimal Demo

### Demo A: Basic import smoke test

```bash
python - <<'PY'
from puresound.audio.io import AudioIO
from puresound.metrics import Metrics

print("PureSound import OK")
print("AudioIO:", AudioIO)
print("Metrics:", Metrics)
PY
```

### Demo B: Save and reload a 1-second silent waveform

```bash
python - <<'PY'
import os
import torch
from puresound.audio.io import AudioIO

sr = 16000
wav = torch.zeros(sr)
out_dir = "./tmp_demo"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "silence.wav")

AudioIO.save(wav=wav, f_path=out_path, sr=sr)
rwav, rsr = AudioIO.open(out_path)
print("Saved:", out_path)
print("Loaded shape:", tuple(rwav.shape), "sr:", rsr)
PY
```

### Demo C: Run speaker verification web demo (requires ONNX model)

```bash
cd egs/speaker_embedding
uv run python demo.py --address 0.0.0.0 --port 7860 pretrained/PS-spk-v1.onnx
```
