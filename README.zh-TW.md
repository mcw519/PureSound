# PureSound

PureSound 是一套以 PyTorch 開發的語音工具，涵蓋語音增強、人聲分離、語者驗證與
房間脈衝響應（RIR）生成。

English: [README.md](README.md)

## 包含內容

- 噪音抑制、人聲分離與語者 embedding 的訓練流程
- 可直接使用的 voice isolation 與 speaker verification ONNX 模型
- 推論 CLI 與本機 Web UI
- 可獨立放進其他專案的 Python ONNX SDK
- 音訊 DSP、資料增強、評估指標與 RIR 生成工具
- 由設定檔驅動的 PyTorch Lightning 訓練

目標語者萃取（TSE）流程只維持相容，不再主動開發。

## 系統需求

- Python 3.12 以上
- 建議使用 [uv](https://docs.astral.sh/uv/) 管理環境

## 安裝

Clone 專案後，選擇一種 ONNX Runtime backend：

```bash
git clone <project-url>
cd PureSound

# CPU 版 ONNX Runtime；macOS 也使用這組安裝 CoreML runtime
uv sync --locked --group dev --extra cpu

# NVIDIA CUDA 版 ONNX Runtime
uv sync --locked --group dev --extra cuda
```

`cpu` 與 `cuda` 不可同時安裝。`asr` extra 依賴 CPU 版 ONNX Runtime，因此也不能
和 `cuda` 一起使用。

若不使用 uv：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## 執行推論

先查看可用模型與執行 backend：

```bash
uv run puresound models list
uv run puresound providers
```

### 人聲分離

輸入會自動轉成單聲道，並在需要時重採樣為 16 kHz。

```bash
uv run puresound infer voice-isolate-dpcrn-v8 \
  --input audio=input.wav \
  --output audio=output.wav \
  --provider auto
```

專案提供兩個 voice-isolation 模型：

- `voice-isolate-dpcrn-v8`：錄音裝置未知時較安全
- `voice-isolate-dpcrn-curriculum-v1`：在目標收音設備上有較強的遠場人聲抑制

兩者差異請看 [checkpoint 說明](egs/voice_isolate/pretrained_ckpt/README.zh-TW.md)。

### 語者驗證

```bash
uv run puresound infer speaker-verification-ps-spk-v1-1 \
  --input enrollment=enrollment.wav \
  --input test=test.wav \
  --output enrollment_embedding=enrollment.npy \
  --output test_embedding=test.npy \
  --provider auto
```

指令會輸出 cosine similarity 與判定結果。

### Web UI

```bash
uv run puresound web
```

開啟 <http://127.0.0.1:7860>。Web UI 包含人聲分離、語者驗證、模型資訊與音訊量測。
服務沒有內建驗證機制，請勿直接開放到不可信任的網路。

API 說明請看 [docs/web.zh-TW.md](docs/web.zh-TW.md)。

## 訓練模型

可執行的流程都放在 `egs/`。開始訓練前，先修改 recipe 裡的資料集與輸出路徑。

| 任務 | 文件 |
| --- | --- |
| 噪音抑制 | [egs/noise_suppression/README.zh-TW.md](egs/noise_suppression/README.zh-TW.md) |
| 人聲分離 | [egs/voice_isolate/README.zh-TW.md](egs/voice_isolate/README.zh-TW.md) |
| 語者 embedding | [egs/speaker_embedding/README.zh-TW.md](egs/speaker_embedding/README.zh-TW.md) |
| 目標語者萃取 | [egs/target_speaker_extraction/README.zh-TW.md](egs/target_speaker_extraction/README.zh-TW.md) |

從頭訓練預設 voice-isolation recipe：

```bash
uv run python egs/voice_isolate/main.py \
  egs/voice_isolate/config/train_dpcrn.yaml \
  --training
```

## 生成 RIR 訓練資料

公開的 RIR 流程可生成結果固定、逐項檢查的 room bank：

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

Backend、輸出格式與選用套件請看
[egs/rir_generation/README.zh-TW.md](egs/rir_generation/README.zh-TW.md)。

## 獨立 ONNX runtime

[`sdk/python`](sdk/python) 是可獨立使用的推論 runtime，只需要 NumPy、ONNX Runtime、
ONNX 模型與對應的 JSON manifest。外部專案不需要安裝完整訓練套件。

## 開發

執行標準檢查：

```bash
uv run python test/run_repo_checks.py --suite standard
```

Release 前執行完整測試：

```bash
uv run python test/run_repo_checks.py
```

建立 Python package：

```bash
./build_puresound.sh
```

## 目錄結構

```text
puresound/   Python package
egs/         訓練、評估與 RIR 流程
docs/        技術文件
model_zoo/   發布模型目錄
sdk/python/  獨立 ONNX runtime
test/        測試與公開 test fixtures
```

## 文件

- [文件首頁](docs/index.zh-TW.md)
- [設定檔](docs/configuration.md)
- [音訊與 RIR](docs/audio/index.zh-TW.md)
- [資料增強](docs/augmentation/index.zh-TW.md)
- [串流推論](docs/streaming/index.zh-TW.md)
- [神經網路](docs/nnet/index.zh-TW.md)
- [訓練系統](docs/system/index.zh-TW.md)

大多數文件都有英文與繁體中文版本。
