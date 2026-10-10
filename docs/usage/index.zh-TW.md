# 使用 PureSound

English: [index.md](index.md)

從剛 clone 下來到部署一個模型的工作流程，依你會遇到的順序排列。每一步都連到說明它的頁面。

## 1. 安裝

Python 3.12 以上，搭配 [uv](https://docs.astral.sh/uv/)。選一個 ONNX Runtime 後端（兩者
不能同時安裝）：

```bash
uv sync --locked --group dev --extra cpu     # CPU（macOS 上為 CoreML）
uv sync --locked --group dev --extra cuda    # NVIDIA CUDA
```

選用的 extra：`asr`（WER 關卡與 web 逐字檢查用的本機 Whisper；不能與 `cuda` 併用）、
`hybrid-rir` / `hybrid-rir-gpu`（RIR 產生器的選用後端）。用 `uv run` 或 `.venv/bin/python`
執行 repo 的工具，不要用 `PATH` 上的其他直譯器。更多見[專案 README](../../README.zh-TW.md)。

## 2. 準備資料

| 內容 | 頁面 |
|---|---|
| 語音 metafile、噪音資料夾、合併語料、清理後的目標，以及混合它們的 recipe 旋鈕 | [data_preparation.zh-TW.md](data_preparation.zh-TW.md) |
| 人聲隔離 recipe 完整的資料鏈（章節語料、真實錄音池、RIR bank view、實測 RIR） | [egs/voice_isolate/DATA_SETUP.zh-TW.md](../../egs/voice_isolate/DATA_SETUP.zh-TW.md) |
| 產生與封裝合成 RIR bank | [egs/rir_generation/README.zh-TW.md](../../egs/rir_generation/README.zh-TW.md) |

## 3. 設定 recipe

recipe 是一份在建構任何東西之前就驗證完的 YAML：任務、dataset、增強區塊、sampler、模型、
loss、優化器、scheduler，以及依 epoch 移動旋鈕的選用 `curriculum` 區塊。
[configuration.zh-TW.md](configuration.zh-TW.md) 說明 schema 與它的規則；
[recipes.zh-TW.md](recipes.zh-TW.md) 說明 `model` 與 `loss_func` 怎麼變成物件。

## 4. 訓練

每個任務都有自己的進入點，底下共用一個驅動程式（`puresound/system/runner.py`）：

| Recipe | 進入點 | 已發布的血統 |
|---|---|---|
| [噪音抑制](../../egs/noise_suppression/README.zh-TW.md) | `egs/noise_suppression/main.py` | 四份 recipe 依序執行，每一份從前一份 warm-start |
| [人聲隔離](../../egs/voice_isolate/README.zh-TW.md) | `egs/voice_isolate/main.py` | 一次排程好的 curriculum run，再加一個 warm-start 的第二步 |

```bash
uv run python egs/noise_suppression/main.py <recipe.yaml> --dump_training_samples   # 先聽
uv run python egs/noise_suppression/main.py <recipe.yaml> --training
```

**課程階梯**有兩種形式。*由多個 run 組成的階梯*在 run 之間改 recipe，每一個都從前一個的
最後一顆 checkpoint warm-start；*`curriculum` 區塊*則把同樣的改動寫成一次 run 之內的 epoch
排程。已發布的模型兩種都用了——見各 recipe 的 README。

**續跑或 warm start。** `--ckpt_path` 在同一份 config 下續跑同一個 run（還原優化器、
scheduler 與 epoch）。`--pretrained_ckpt_path` 從另一顆 checkpoint 的權重 warm-start 一個新
的 run：名稱以寬鬆方式對應（新參數維持初始化、過時的參數被忽略，兩者都會記進 log），形狀則
嚴格——形狀改變會被拒絕，除非傳 `--pretrained_allow_reshaped`；它是給刻意改變 STFT 窗長用
的，會把那些參數在初始化時重建。旗標說明見
[egs/noise_suppression/README.zh-TW.md](../../egs/noise_suppression/README.zh-TW.md#續跑warm-start重新載入)。

比較 checkpoint 要在 scheduler 的 cosine 谷底，而且要把一個 run 末尾的一組 checkpoint 當成
一個 block 來比，而不是只比一顆。

## 5. 評測：閘門

一顆 checkpoint 只能透過它 recipe 的閘門發布。每一關都拿候選與「什麼都不做」比，讀配對差；
gate 關卡可以否決發版，monitor 只報告。

```bash
bash egs/noise_suppression/run_full_gate.sh baseline
bash egs/noise_suppression/run_full_gate.sh <tag> <ckpt> <recipe>
```

| 頁面 | 涵蓋 |
|---|---|
| [evaluation.zh-TW.md](evaluation.zh-TW.md) | 建測試集、執行閘門、它的變數、紀錄與判定 |
| [egs/noise_suppression/benchmarks/stages.zh-TW.md](../../egs/noise_suppression/benchmarks/stages.zh-TW.md) | 每一關能決定什麼、不能決定什麼 |
| [egs/voice_isolate/README.zh-TW.md](../../egs/voice_isolate/README.zh-TW.md#benchmark) | 人聲隔離的 benchmark `run_full_benchmark.sh`；其中田野錄音的關卡使用不隨儲存庫散布的私有錄音 |

## 6. 匯出與部署

checkpoint 以逐幀 ONNX graph 加 JSON manifest 部署；runtime 負責 STFT、狀態，以及 manifest
記錄的 graph 後處理階段（dry blend、onset guard）。

```bash
uv run python egs/voice_isolate/scripts/streaming_onnx.py export <infer.yaml> <ckpt> model.onnx
uv run python egs/voice_isolate/scripts/streaming_onnx.py verify <infer.yaml> <ckpt> model.onnx \
    --input_audio speech.wav
```

| 頁面 | 涵蓋 |
|---|---|
| [streaming/index.zh-TW.md](streaming/index.zh-TW.md) | 串流相關頁面 |
| [streaming/dpcrn_onnx.zh-TW.md](streaming/dpcrn_onnx.zh-TW.md) | 匯出、驗證、look-ahead、runtime、輔助頭 |
| [sdk/python](../../sdk/python/README.zh-TW.md) | 可攜式 runtime：只需 NumPy 與 ONNX Runtime |

## 7. Web playground

`uv run puresound web` 在 <http://127.0.0.1:7860> 提供本機 UI：對檔案、錄音或即時麥克風跑
模型，用分數與逐字檢查比較它們，並瀏覽 model zoo。見 [web.zh-TW.md](web.zh-TW.md)，包括
HTTP API。

## 8. Model zoo

`model_zoo/catalog.yaml` 登錄已發布的模型；每一筆都指向某個 recipe
`pretrained_ckpt/streaming/` 底下的 ONNX 匯出與其 manifest。

```bash
uv run puresound models list
uv run puresound models validate
uv run puresound infer <model-id> --input audio=in.wav --output audio=out.wav --provider auto
```

每個版本為什麼發布、要怎麼選，寫在各 recipe `pretrained_ckpt/README.md` 的判定表
（[噪音抑制](../../egs/noise_suppression/pretrained_ckpt/README.zh-TW.md)、
[人聲隔離](../../egs/voice_isolate/pretrained_ckpt/README.zh-TW.md)）。run 目錄與 catalog id
怎麼命名、哪些產物會發布：[repository_layout.zh-TW.md](../repository_layout.zh-TW.md)。

可在[聲學世界](world.zh-TW.md)探索移動聲源，或在試聽台選 *這台裝置*，直接在
瀏覽器中執行模型（[Web SDK](../../sdk/web/README.zh-TW.md)）。
