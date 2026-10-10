# RIR 套件佈局 — `puresound.audio.rir`

English version: [rir_package_layout.md](rir_package_layout.md)

RIR 套件如何分層、每項工作該 import 哪個模組，以及讓各層保持分離的規則。每個
階段計算什麼見[實作指南](rir_realism_algorithm.zh-TW.md)。

## 分層

相依只能往下；模組可以 import 同層或任何較低層。

| 層級 | Layer | 內容 |
|---|---|---|
| 4 | `api` | 便利性的 re-export |
| 3 | `render`、`calibration`、`bank` | Backend 與組裝；量測房間校準；bank 儲存、manifest、QC、release、loader |
| 2 | `scene`、`path_events`、`metrics` | Scene schema 與抽樣；相干路徑；聲學指標 |
| 1 | `physics` | 傳播、阻抗、波動求解器 |
| 0 | `contracts` | 資料慣例、`HybridRIRConfig`、`RIRArray`、`BackendCapabilities` |

`contracts` 只 import 標準函式庫與 NumPy：不 import Torch、renderer、bank 模組，
也不存取檔案系統。必須在沒有 Torch、Pyroomacoustics 或 CuPy 時仍可 import 的模組
包括 scene、metrics 與 path-event 層、`physics.propagation`、FDN 與耦合模組，以及
bank 的 schema、QC、release、evaluation 與 production 模組——讀 manifest 不應需要
渲染堆疊。`bank/__init__.py` 基於同樣理由不 import 任何東西。
`test/rir/test_rir_import_boundaries.py` 強制這兩條規則（其 `LAYER_RANK` 表與
輕量模組清單），違反時讓 build 失敗。

訓練只 import `bank.loader`。改動其他任何模組，在 bank 重建並重新發佈之前都不會
影響任何訓練。

## 入口

| 工作 | Import |
|---|---|
| 渲染 hybrid RIR | `from puresound.audio.rir.render.hybrid import generate_hybrid_rir` |
| 設定與資料契約，不載入渲染堆疊 | `from puresound.audio.rir.contracts import HybridRIRConfig, RIRArray, validate_rir_metadata` |
| Scene | `from puresound.audio.rir.scene.schema import RoomSceneV2`；抽樣在 `scene.sampling` |
| Path event | `from puresound.audio.rir.path_events import PathEvent, render_path_events` |
| Backend | `render.low_frequency`、`render.high_frequency` |
| Crossover 工具 | `render.crossover`（`hybrid_crossover_with_metadata`、`align_high_band_direct`、`clip_rir_before_physical_arrival`） |
| 空間輸出 | `render.spatial.render_room_scene_spatial_rir` |
| 指標 | `from puresound.audio.rir.metrics import analyze_rir` |
| Bank manifest（不需 Torch） | `bank.schema` |
| 訓練 loader | `bank.loader`（`PreGeneratedRoomBank`、`PreGeneratedReleaseBank`、`UnionRoomBank`） |
| 寫出一個 item | `bank.storage.write_hybrid_rir_dataset_item` |

`puresound.audio.rir.api` 為了方便 re-export 一組常用名稱。它不是穩定性邊界，而且
會載入整個渲染堆疊（含 Torch）；函式庫程式從所屬的 layer import。
