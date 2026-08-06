# 測試工具

English version: `README.md`

先建置 repository 所管理的環境：

```bash
uv sync --locked --group dev
```

在疊代開發時，執行聚焦檢查：

```bash
uv run python test/run_repo_checks.py
```

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

執行完整的 pytest 套件：

```bash
uv run python test/run_repo_checks.py --suite full --skip-ruff
```

若要檢視某個 recipe 實際訓練時用的是什麼資料，可以使用
`egs/voice_isolate/scripts/check_training_data.py --dump`（真實 recipe
pipeline），或 `egs/rir_generation/tools/audition/simulate_room_scene.py`
（on-the-fly 房間模擬器）。
