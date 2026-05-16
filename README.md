# PureSound

PureSound is a speech processing toolkit based on PyTorch and PyTorch Lightning.
PureSound 是一個以 PyTorch 與 PyTorch Lightning 為核心的語音處理工具包。

It provides reusable audio utilities, model components, and training recipes for:
它提供可重用的音訊工具、模型元件與訓練流程，涵蓋：

- Noise Suppression (NS)
- Speaker Embedding / Speaker Verification (SV)
- Target Speaker Extraction (TSE)

## Highlights

- Modular design: `audio`, `dataset`, `nnet`, `system`, and `task`
- Config-driven training and inference (YAML)
- Multiple backbone models (for example: DPCRN, DPRNN, DPARn, ECAPA-TDNN, TF-GridNet)
- Built-in objective and subjective metrics (for example: PESQ, STOI, SDR related tools)

## Requirements / 系統需求

- Python 3.10+
- A working PyTorch environment compatible with your platform

## Installation / 安裝

### Option 1: Use uv (recommended for development)

### 方案 1：使用 uv（建議開發使用）

```bash
git clone <project-url>
cd PureSound
uv sync --locked --group dev
```

The repository pins `torch`, `torchaudio`, and `torchcodec` to the PyTorch CUDA 12.4 wheel index in `pyproject.toml` and `uv.lock`, so the same `uv` setup can be reused on another machine without re-resolving to a different CUDA build.

### Option 2: Use pip

### 方案 2：使用 pip

```bash
git clone <project-url>
cd PureSound
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`requirements.txt` mirrors the runtime dependencies and includes the PyTorch CUDA 12.4 wheel index for `pip`.

## Quick Validation / 快速驗證

Run the test suite:

```bash
uv run pytest
```

Or with pip environment:

```bash
pytest
```

## Quick Start Recipes / 快速開始（Recipes）

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

### 3) Target Speaker Extraction

```bash
cd egs/target_speaker_extraction

# Prepare metadata
uv run python prepare_metafile.py --help

# Train
uv run python main.py --training True config/default_config.yaml

# Inference
uv run python main.py --inference True --ckpt_path /path/to/model.ckpt config/default_config.yaml
```

## Repository Structure / 專案結構

```text
PureSound/
├── puresound/                 # Core library
│   ├── audio/                 # Audio I/O, DSP, augmentation
│   ├── dataset/               # Dataset base classes and parsers
│   ├── nnet/                  # Model architectures and building blocks
│   ├── system/                # Lightning training systems
│   ├── task/                  # Task-specific dataset logic
│   ├── metrics.py             # Evaluation metrics
│   ├── recipes.py             # Model/loss initialization helpers
│   └── utils.py               # General utilities
├── egs/                       # End-to-end recipes and configs
├── docs/                      # API and module documentation
└── test/                      # Unit tests
```

## Documentation / 文件

- Main docs entry: `docs/index.md`
- Audio modules: `docs/audio/index.md`
- Neural network modules: `docs/nnet/index.md`
- System modules: `docs/system/index.md`

## Build Package / 打包

```bash
./build_puresound.sh
```

This script runs:

- `uv sync --group dev`
- `uv sync --locked --group dev`
- `uv build`

## Notes / 備註

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
