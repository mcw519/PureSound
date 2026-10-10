# 測試

English: [README.md](README.md)

先建立 repo 管理的環境：

```bash
uv sync --locked --group dev
```

## 結構

`test/` 對應函式庫結構：`test/<package>/test_<module>.py`。整棵樹的測試檔名都不重複
（這些目錄不是 Python package，pytest 以檔名 import 每個測試檔）。

| 目錄 | 涵蓋範圍 |
| --- | --- |
| `audio/` | RIR 以外的 `puresound.audio`：I/O、DSP、增強旋鈕 |
| `config/` | recipe schema 與載入（`puresound.config`） |
| `dataset/` | dataset 框架與所有語料準備工具（`puresound.dataset`、`puresound.dataset.corpus`） |
| `evaluation/` | 評測閘門：工具、統計、紀錄、指標 |
| `inference/` | model zoo、runtime、recipe registry、CLI |
| `nnet/` | backbone、積木、head 與 loss（`puresound.nnet`） |
| `rir/` | RIR 產生堆疊（`puresound.audio.rir`）與 `egs/rir_generation` 的工具 |
| `streaming/` | ONNX 匯出、串流一致性與獨立的 SDK runtime |
| `system/` | Lightning module、訓練驅動、後處理、presence gate、onset guard |
| `task/` | 合成資料集、裝置鏈、噪音階段、sampler、session rows |
| `web/` | web playground 伺服器 |

`fixtures/` 放多個測試共用的小型 recipe 與資料；`test_case/` 放它們讀取的音檔。

## 執行

測試分三層，在 pytest-xdist 下執行，每個 worker 一個運算執行緒：

| 層 | 範圍 |
| --- | --- |
| `--suite quick` | 單元測試目錄（`rir/` 與 `web/` 以外），不含 `slow` 測試 |
| `--suite standard` | 標為 `slow` 以外的全部 |
| `--suite full`（預設） | 全部 |

```bash
uv run python test/run_repo_checks.py                    # full，commit 前跑
uv run python test/run_repo_checks.py --suite quick      # 開發中
```

需要 debugger 或可讀的即時輸出時加 `--jobs 0` 關掉 xdist，或用 `--jobs N` 指定 worker
數。執行緒釘選請保留：不然 torch 會在每個 worker 開一個核心一條 intra-op 執行緒，
worker 之間互相搶滿整台機器。

超過幾秒的測試標為 `slow`：bank 的建置/QC/發布流程、離線對 ONNX 的串流等效性、長
session 合成。部署路徑（DPCRN、DPARN、串流 runtime）保留在 `standard` 的快速測試。

## 撰寫測試

一個測試釘住呼叫端依賴的一個行為：輸出契約、已出貨的路徑、必須大聲失敗的錯誤。
同一個行為的多個情境寫成一個參數化測試。見
[docs/repository_layout.zh-TW.md](../docs/repository_layout.zh-TW.md) 的「註解、文件與測試」一節。

要聽 recipe 實際訓練的資料，用 `egs/voice_isolate/scripts/check_training_data.py --dump`
（真實 recipe pipeline）或 `egs/rir_generation/tools/audition/simulate_room_scene.py`
（即時房間模擬）。
