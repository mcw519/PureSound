# PureSound

[授權：Apache-2.0](LICENSE)

English: [README.md](README.md)

PureSound 是一套以 PyTorch 開發的單聲道語音增強工具：包含即時合成訓練資料的訓練
recipe、評測閘門、串流 ONNX 匯出，以及可從指令列、本機 Web UI 或小型獨立 Python
runtime 執行的 model zoo。

## 已發布模型

列在 [`model_zoo/catalog.yaml`](model_zoo/catalog.yaml)。每個模型都吃 16 kHz
單聲道音訊；輸入會在需要時自動轉換。

| 任務 | model id | role | 用途 |
| --- | --- | --- | --- |
| 人聲分離 | `voice-isolate-dpcrn-curriculum-v1` | default | 保留靠近麥克風的說話者，壓制遠處的說話者與噪音；在調校時用的收音硬體上效果最強 |
| 人聲分離 | `voice-isolate-dpcrn-curriculum-v2` | candidate | 加寬的 curriculum 模型：WER 比 v1 低、遠處講話者壓得更深，CPU 成本是 v1 的 1.4 倍 |
| 人聲分離 | `voice-isolate-dpcrn-v8` | candidate | 收音硬體未知時的保守選擇 |
| 降噪 | `noise-suppression-dpcrn-mamba-v3` | candidate | 較寬的串流模型，PESQ 與 STOI 較高；刪字護欄仍未定論 |
| 降噪 | `noise-suppression-dpcrn-mamba-v2` | default | 串流降噪，平衡音質與 CPU 成本 |
| 降噪 | `noise-suppression-dpcrn-mamba-v1` | candidate | 同一個模型、少了最後的音質微調；在乾淨朗讀語音上刪字較少 |
| 語者驗證 | `speaker-verification-ps-spk-v1-1` | default | 192 維語者 embedding |
| 語者驗證 | `speaker-verification-ps-spk-v1` | alternative | 較早的 embedding 版本 |

每個版本依據什麼判定，請看各 recipe 的 checkpoint 說明：
[人聲分離](egs/voice_isolate/pretrained_ckpt/README.zh-TW.md)、
[降噪](egs/noise_suppression/pretrained_ckpt/README.zh-TW.md)、
[語者 embedding](egs/speaker_embedding/README.zh-TW.md)。

## 安裝

需要 Python 3.12 以上；建議使用 [uv](https://docs.astral.sh/uv/)。選擇一種 ONNX
Runtime backend：

```bash
git clone <project-url>
cd PureSound

# CPU 版 ONNX Runtime（macOS 上為 CoreML）
uv sync --locked --group dev --extra cpu

# NVIDIA CUDA 版 ONNX Runtime
uv sync --locked --group dev --extra cuda
```

`cpu` 與 `cuda` extra 不能同時安裝，`asr` extra（WER stage 用的本機 Whisper）也
不能與 `cuda` 一起使用。

改用 pip：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## 快速開始

列出已發布模型與這台機器可用的 ONNX Runtime provider：

```bash
uv run puresound models list
uv run puresound providers
```

對一個檔案執行模型：

```bash
uv run puresound infer voice-isolate-dpcrn-curriculum-v1 \
  --input audio=input.wav --output audio=output.wav --provider auto

uv run puresound infer speaker-verification-ps-spk-v1-1 \
  --input enrollment=enrollment.wav --input test=test.wav
```

語者驗證會輸出 cosine similarity 與判定結果。

啟動本機 Web UI（<http://127.0.0.1:7860>）：

```bash
uv run puresound web
```

它沒有驗證機制，請勿開放到不可信任的網路。

若要在其他專案中嵌入已發布模型而不安裝 PureSound，請用
[`sdk/python`](sdk/python/README.zh-TW.md) 的獨立 runtime：只需要 NumPy、ONNX
Runtime、ONNX 檔與它的 JSON manifest。

## 訓練模型

`egs/` 下的每個 recipe 都要在自己的目錄裡執行，因為 recipe 裡的資料與輸出路徑
是相對於該目錄。請先把 recipe 裡的 metafile 與資料夾路徑指向你的資料。

```bash
cd egs/voice_isolate
uv run python main.py config/train_dpcrn.yaml --training
```

| 任務 | recipe |
| --- | --- |
| 降噪 | [egs/noise_suppression](egs/noise_suppression/README.zh-TW.md) |
| 人聲分離 | [egs/voice_isolate](egs/voice_isolate/README.zh-TW.md) |
| 語者 embedding | [egs/speaker_embedding](egs/speaker_embedding/README.zh-TW.md) |
| 目標語者萃取（凍結的 legacy） | [egs/target_speaker_extraction](egs/target_speaker_extraction/README.zh-TW.md) |
| RIR bank 生成 | [egs/rir_generation](egs/rir_generation/README.zh-TW.md) |

## 文件

入口是 [docs/index.zh-TW.md](docs/index.zh-TW.md)，分成三部分：

- [架構](docs/architecture/index.zh-TW.md)——各 package 如何組合，以及資料如何從
  語料流到部署
- [演算法](docs/index.zh-TW.md#演算法)——每個模型、loss、合成階段與指標計算什麼
- [使用方式](docs/index.zh-TW.md#使用方式)——準備資料、訓練、評測、匯出與部署

新檔案該放哪：[docs/repository_layout.zh-TW.md](docs/repository_layout.zh-TW.md)。

## 開發

```bash
uv run python test/run_repo_checks.py --suite standard   # 日常檢查
uv run python test/run_repo_checks.py                    # 完整測試，release 前執行
./build_puresound.sh                                     # 建立 package
```

```text
puresound/   函式庫
egs/         recipe driver、設定檔、已發布 checkpoint
docs/        文件
model_zoo/   已發布模型目錄
sdk/python/  獨立串流 runtime
test/        測試與公開 test fixtures
```

## 授權

PureSound 自有程式碼、SDK、設定檔、文件，以及 Milo Wu 或 PureSound 貢獻者自行
訓練的模型均採用 [Apache-2.0](LICENSE)，包含 model zoo 發布的 checkpoint 與
ONNX 匯出。其他人可以依授權條款使用、修改、商用與再散佈這些模型。
適用範圍與排除項目請見 [NOTICE](NOTICE)。第三方內容（包含第三方模型）保留
各自的授權。資料集與音訊錄音除非另有明確授權，均不包含在此次 Apache-2.0
授權中；使用前請查閱各項目的授權與來源資訊。
