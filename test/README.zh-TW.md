# 測試工具

English version: `README.md`

先建置 repository 所管理的環境：

```bash
uv sync --locked --group dev
```

測試套件分成三個層級。三個層級都在 pytest-xdist 下平行執行，且每個 worker
只綁定一條計算執行緒——同一套測試若序列執行會超過一小時，時間主要花在 RIR
物理與 M5/M6 evidence 測試上。

| 層級 | 範圍 | 實際耗時 |
|---|---|---|
| `--suite quick` | 只跑單元測試（`test/*/` 子套件） | ~20 秒 |
| `--suite standard` | 全部，但排除標記為 `slow` 的測試 | ~35 秒 |
| `--suite full` *(預設)* | 全部 | ~2 分鐘 |

```bash
uv run python test/run_repo_checks.py                    # full，commit 前跑這個
uv run python test/run_repo_checks.py --suite quick      # 疊代開發時
```

上述耗時是在 24 核心機器上量測的。之所以把 `full` 設為預設，是因為它只比
`standard` 多花約一分鐘；而 `standard` 在核心數較少的機器上才會顯出價值，
因為那時兩者的差距會拉大。

需要使用 debugger 或想看到即時輸出時，加上 `--jobs 0` 可關閉 xdist；也可以
用 `--jobs N` 指定 worker 數量。除非你清楚它的用途，否則不要動每個 worker
的執行緒綁定：torch 預設會在**每一個** worker 裡，依核心數各開一條 intra-op
執行緒，由此產生的超額訂閱（oversubscription）曾讓那 63 個 `slow` 測試用掉
339 分鐘的 CPU，只為了完成 3 分鐘的工作量。

668 個測試中有 63 個帶有 `slow` 標記：M5/M6 evidence-chain validators（每一個
都會實際建立、QC 並發布一個 bank）、兩個 offline 對 ONNX 的 streaming 等價性
檢查（各約 2.5 分鐘），以及 library 狀態 backbone（SkiM、DPRNN、TF-GridNet）
的 forward smoke tests。現役部署路徑——DPCRN 與 DPARN——則刻意保留在
`standard` 層級中。

整個 pytest 套件共有 87 個測試檔案。其中多數——64 個檔案，約 74%——
直接放在 `test/` 根目錄下，沒有歸入任何具名子目錄；這些檔案幾乎全部
是 RIR/impedance/room-acoustics 子系統的測試（`test_rir_*`、
`test_impedance_*`、`test_m5_*`、`test_m6_*`，以及 `egs/rir_generation`
子系統相關的 FDN/path-event/calibration 涵蓋範圍）。這就是這個套件
目前實際的樣貌，並不是有待補齊的缺口：RIR 子系統的測試量本來就比
其餘部分加起來還多，規模自然超過任何一個具名子目錄。

其餘 23 個檔案依領域分類，歸入五個子目錄：

- `test/test_audio`：audio I/O、DSP、augmentation 與 simulation
- `test/test_metrics`：DNSMOS 等 evaluation metrics
- `test/test_losses`：loss functions
- `test/test_utils`：recipe smoke tests、CLI helpers 與 data adapters
- `test/test_system`：系統層級整合測試——channel-consistency
  regularization、DPCRN gate/VAD 訓練、optimizer 的 param-group
  管線（plumbing），以及 SISO `compute_loss` 路由（routing）

若要檢視某個 recipe 實際訓練時用的是什麼資料，可以使用
`egs/voice_isolate/scripts/check_training_data.py --dump`（真實 recipe
pipeline），或 `egs/rir_generation/tools/audition/simulate_room_scene.py`
（on-the-fly 房間模擬器）。
