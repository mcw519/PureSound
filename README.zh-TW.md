# PureSound

PureSound 是一個以 PyTorch 與 PyTorch Lightning 為核心的語音處理工具包。
它提供可重用的音訊工具、模型元件與訓練流程，涵蓋：

- 噪音抑制（Noise Suppression, NS）
- 近場人聲分離（Near-field Voice Isolation）
- 語者 Embedding／語者驗證（Speaker Verification, SV）
- 目標語者萃取（Target Speaker Extraction, TSE，legacy）
- 用於訓練資料增強的房間脈衝響應（Room Impulse Response, RIR）生成

English version: [`README.md`](README.md)

## 特色

- 模組化設計：`audio`、`dataset`、`nnet`、`system`、`task`
- 設定檔驅動（YAML）的訓練與推論流程
- 一個讓每個 backbone 都保持可透過設定檔存取的模型庫（DPCRN、DPARN、DPRNN、SkiM、Conv-TasNet、TF-GridNet、ECAPA-TDNN），即使目前只有部分被現役 recipe 使用
- **DPCRN** 串流 ONNX Runtime 部署，用於即時 voice-isolate 推論（現行發布路徑；DPARN 串流則保留作為 legacy 替代方案）
- 獨立的 `sdk/python` runtime，可在不安裝完整訓練套件的情況下，將串流推論嵌入其他專案
- 公開的 RIR 生成流程（`egs/rir_generation`），用於近／遠場增強訓練資料
- 內建的客觀與主觀評估指標（例如：PESQ、STOI、SDR 相關工具）

## 系統需求

- Python 3.12+
- 一套與你的平台相容、可正常運作的 PyTorch 環境

## 安裝

### 方案 1：使用 uv（建議開發使用）

```bash
git clone <project-url>
cd PureSound
uv sync --locked --group dev --extra cpu
```

Linux 使用固定的 PyTorch CUDA 12.4 wheels；macOS 使用標準平台 wheels，因此可安裝並使用 PyTorch MPS。

ONNX Runtime 的 CPU 與 GPU distributions 提供相同 Python package，因此改用
互斥 extras；請只選一種 runtime：

```bash
uv sync --locked --group dev --extra cpu   # CPU 或 macOS CoreML
uv sync --locked --group dev --extra cuda  # NVIDIA CUDA 12.x
```

請勿同時啟用兩個 extras。`sdk/python` 的可攜式 SDK 也採相同的 `cpu`／`cuda`
選擇方式。執行推論前可確認目前使用的 wheel：

CPU／CoreML 環境若要使用 `faster-whisper`，可再加上 `--extra asr`。它的 package
metadata 會要求 CPU ONNX Runtime distribution，因此不能和 `--extra cuda` 併用。

```bash
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
# 或
puresound providers
```

輸出必須包含 `CUDAExecutionProvider` 才代表 ONNX Runtime 能使用 CUDA。只有
CUDA 版 PyTorch 並不會自動替 ONNX Runtime 加入 CUDA；NVIDIA driver、CUDA 與
cuDNN 動態函式庫也必須能被目前的 process 載入。

在 macOS 上安裝一般的 `onnxruntime` wheel。官方 macOS wheel 可提供
`CoreMLExecutionProvider`，在 operator 支援時使用 Apple GPU／Neural Engine。
ONNX Runtime 沒有原生的 `MPSExecutionProvider`；PureSound 接受
`--provider mps` 作為 CoreML 的 alias（明確寫法是 `--provider coreml`）。
`auto` 會依序選擇 CUDA、CoreML、CPU。只有 PyTorch 的 advanced path 才會使用
PyTorch 自己的 MPS backend，且需 `torch.backends.mps.is_available()` 為 true。

### 方案 2：使用 pip

```bash
git clone <project-url>
cd PureSound
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`requirements.txt` 鏡射了執行期相依套件，並為 `pip` 附上 PyTorch CUDA 12.4 wheel 索引。

## 快速驗證

執行測試套件：

```bash
uv run pytest
```

或在 pip 環境下：

```bash
pytest
```

## Web 推論工作區

本地 Model Zoo 與統一 ONNX runtime 也提供一個不需額外前端框架的瀏覽器
工作區。它與 CLI 及 Gradio 相容 demo 使用相同的 named-input processor：

```bash
puresound web
# 開啟 http://127.0.0.1:7860
```

在可信任網路提供服務時，可用
`puresound web --ip 0.0.0.0 --port 8080` 控制綁定的 IP 與 port。

API 與上傳格式請參考 [`docs/web.zh-TW.md`](docs/web.zh-TW.md)。

## 快速開始（Recipes）

`egs` 資料夾內含可直接執行的範例。

### 1）噪音抑制（Noise Suppression）

```bash
cd egs/noise_suppression

# 準備訓練／驗證用的 manifest
uv run python prepare_metafile.py --help

# 訓練
uv run python main.py --training True config/dpcrn.yaml

# 推論
uv run python main.py --inference True --ckpt_path /path/to/model.ckpt config/dpcrn.yaml
```

更多細節（noise-suppression 與 voice-isolation 訓練共用的 `dataset.task` 切換開關、DDP／精度旗標、VAD 標註機制）：見 `egs/noise_suppression/README.md`。

### 2）語者 Embedding／驗證

```bash
cd egs/speaker_embedding

# 準備 metadata
uv run python prepare_metafile.py --help

# 訓練
uv run python main.py --training True conf/PS-spk-v1.yaml

# 推論（萃取 embedding）
uv run python main.py --inference True --ckpt_path /path/to/model.ckpt conf/PS-spk-v1.yaml
```

更多語者 embedding 細節與預訓練 checkpoint，請見：

- `egs/speaker_embedding/README.md`

### 3）目標語者萃取（Target Speaker Extraction，legacy）

已凍結的 legacy recipe：不再新增功能，也不再重構，僅維持可運作以供參考。

```bash
cd egs/target_speaker_extraction

# 準備 metadata
uv run python prepare_metafile.py --help

# 訓練
uv run python main.py --training True config/default_config.yaml

# 推論
uv run python main.py --inference True --ckpt_path /path/to/model.ckpt config/default_config.yaml
```

### 4）Voice Isolate 串流 ONNX

訓練或微調 16 kHz 的 DPCRN voice-isolate recipe，再匯出逐幀（feature-frame）ONNX 模型。以下範例使用 `egs/voice_isolate` recipe 及其目前的預設 checkpoint：

```bash
uv run python egs/voice_isolate/scripts/streaming_onnx.py export \
  egs/voice_isolate/config/infer_dpcrn.yaml \
  egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt \
  /path/to/model.onnx
```

執行串流 ONNX 推論：

```bash
uv run python egs/voice_isolate/scripts/streaming_onnx.py infer \
  /path/to/model.onnx \
  input.wav \
  output.wav \
  --provider auto
```

`dpcrn_v8` 已將 runtime 的 dry/wet 混合比例烘焙進匯出的計算圖中
（`out = 0.9 * enhanced + 0.1 * input`），推論時不需要額外的後處理。完整
的 pipeline 歷史、各版本結果與部署注意事項：見 `egs/voice_isolate/README.md`。
（legacy 的）DPARN 串流路徑則另外記錄在 `docs/streaming/dparn_onnx.md`。

若要在不安裝完整 PureSound 訓練套件的情況下部署到其他專案，可安裝或複製
`sdk/python` 內的可攜式 runtime。它只需要 NumPy、ONNX Runtime、`model.onnx`
與 `model.json`。

### 5）RIR 生成（訓練資料增強）

生成建議使用的 M6 訓練用 RIR bank（一個指令完成確定性生成＋逐項 QC＋release 打包）：

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

務必使用 `.venv/bin/python` 執行（或任何已安裝 `pyroomacoustics`／
`rir_generator` 且 numpy ABI 相符的環境）——直譯器不對會產生外觀像程式
bug、但其實不是的蒐集錯誤（collection errors）。完整用法與演算法對應
程式碼的參考文件：見 `egs/rir_generation/README.md` 與 `docs/audio/index.md`。

## 專案結構

```text
PureSound/
├── puresound/                 # 核心函式庫
│   ├── audio/                 # 音訊 I/O、DSP、增強、RIR 生成（audio/rir/）
│   ├── dataset/                # Dataset 基底類別與 parser
│   ├── nnet/                   # 模型架構與建構模組
│   ├── streaming/               # 串流推論與 ONNX Runtime 工具
│   ├── system/                 # Lightning 訓練系統
│   ├── task/                   # 任務專屬的 dataset 邏輯
│   ├── third_party/             # vendored 的第三方研究程式碼（pytARD）
│   ├── metrics.py               # 評估指標
│   ├── recipes.py               # 模型／損失函數初始化輔助工具
│   └── utils.py                 # 通用工具函式
├── egs/                        # 端到端 recipe 與設定檔
├── docs/                       # API 與模組文件
├── sdk/                        # 供外部專案使用的可攜式推論 SDK
└── test/                       # 單元測試
```

## 文件

- 文件主入口：`docs/index.md`
- 音訊模組（含 RIR 生成）：`docs/audio/index.md`
- 神經網路模組：`docs/nnet/index.md`
- 系統模組：`docs/system/index.md`
- 串流 runtime：`docs/streaming/index.md`

`docs/`、`egs/` 底下的每一份文件，以及各套件層級的 `README.md`，都同時提供
英文版（`name.md`）與繁體中文版（`name.zh-TW.md`）。

## 打包

```bash
./build_puresound.sh
```

此腳本會依序執行：

- `uv sync --locked --group dev --extra cpu`
- `uv build`

## 備註

- 部分 recipe 腳本使用 `--training True` 或 `--inference True` 這種布林值 CLI 旗標。
- 訓練前請先調整每份 YAML 設定檔中的資料集路徑與輸出資料夾。

## 疑難排解

### 1）`ModuleNotFoundError: No module named 'puresound'`

原因：
- 目前使用的環境中尚未安裝此套件。

解法：

```bash
uv sync --locked --group dev --extra cpu
# 或
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

### 2）`torchaudio` 拋出 `OSError` 或後端錯誤

原因：
- PyTorch／torchaudio 的二進位版本不相容，或缺少執行期所需的 codec／後端支援。

解法：

```bash
python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__)"
```

請確認 `torch` 與 `torchaudio` 是從相容的頻道／版本安裝的。若使用本專案
管理的 `uv` 流程，它們已釘選在 PyTorch CUDA 12.4 wheel 索引上，應搭配已
checked-in 的 lock file 安裝。

### 3）CUDA 無法使用（`torch.cuda.is_available() == False`）

原因：
- 純 CPU 環境、不支援的 CUDA runtime，或 PyTorch build 版本不匹配。

解法：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

若需要 CUDA，請重新安裝與你系統相符、支援 CUDA 的 PyTorch build。本專案
可重複使用的 `uv` 設定是以 CUDA 12.4 為目標。

ONNX 推論要另外檢查 ONNX Runtime：

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

若沒有 `CUDAExecutionProvider`，通常代表仍安裝 CPU wheel，或 GPU wheel 無法
載入 CUDA／cuDNN 相依函式庫。請依安裝章節替換 wheel，並重新啟動 web service。
API 與 UI 會回報實際選到的 provider；當 provider 不可用時，即使明確要求 CUDA
也會依相容性規則 fallback 到 CPU。

在 macOS 上請檢查 `CoreMLExecutionProvider`，並選擇 `coreml` 或 `mps` alias；
ONNX Runtime 沒有 `MPSExecutionProvider`。

### 4）Recipe 啟動後找不到資料檔案

原因：
- YAML 中的資料集路徑尚未依本機環境更新。

解法：
- 編輯 `egs/*/config` 或 `egs/speaker_embedding/conf` 底下對應的 recipe 設定檔。
- 先執行 `prepare_metafile.py` 以生成 manifest。

### 5）RIR 生成：外觀像程式 bug 的蒐集錯誤或驗證失敗

原因：
- 用缺少 `pyroomacoustics`／`rir_generator`，或 numpy ABI 不相符的
  Python 直譯器執行 `egs/rir_generation` 底下的腳本。

解法：
- 一律使用 `.venv/bin/python`（或任何已安裝上述套件的等效環境）執行
  RIR 生成腳本。

## 最小示範

### 示範 A：基本 import 煙霧測試

```bash
python - <<'PY'
from puresound.audio.io import AudioIO
from puresound.metrics import Metrics

print("PureSound import OK")
print("AudioIO:", AudioIO)
print("Metrics:", Metrics)
PY
```

### 示範 B：儲存並重新讀取一段 1 秒的靜音波形

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

### 示範 C：執行語者驗證網頁 Demo（需要 ONNX 模型）

```bash
cd egs/speaker_embedding
uv run python demo.py --address 0.0.0.0 --port 7860 pretrained/PS-spk-v1.onnx
```
