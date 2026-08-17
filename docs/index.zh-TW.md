# PureSound Documentation

English version: [index.md](index.md)

PureSound 是一套模組化的音訊處理與深度學習框架，用於語音增強、聲紋辨識
（speaker verification）與目標語者抽取（target speaker extraction）等任務。

## Package Version

`0.1`

## Module Overview

| Module | Description |
|--------|-------------|
| [puresound.audio](audio/index.md) | 音訊 I/O、DSP、augmentation，以及頻譜相關工具 |
| [puresound.dataset](dataset/index.md) | Dataset 的 base classes 與 parser |
| [puresound.nnet](nnet/index.md) | 神經網路架構與 building blocks |
| [puresound.system](system/index.md) | PyTorch Lightning 訓練系統 |
| [puresound.task](task/index.md) | Task-specific 的 dataset 實作 |
| [puresound.streaming](streaming/index.md) | Streaming 推論與 ONNX Runtime 部署 |
| [puresound.metrics](metrics.md) | 音訊品質評估 metrics |
| [puresound.utils](utils.md) | 通用工具函式 |
| `puresound.logging_setup` | 函式庫執行期輸出的去向，以及如何接手控制 |
| [`puresound.config`](configuration.md) | Pydantic task recipes、capability models、migration 與驗證 |
| [puresound.recipes](recipes.md) | 高階的模型初始化 recipes |

## Architecture Overview

```
puresound/
├── audio/          # Audio processing primitives
├── dataset/        # Dataset base classes
├── nnet/           # Neural network models
│   ├── lobe/       # Reusable network building blocks
│   └── loss/       # Loss functions
├── system/         # Training system (PyTorch Lightning)
├── streaming/      # Streaming inference runtimes
├── task/           # Task-specific datasets (NS, near-field voice isolation, SV, TSE)
├── third_party/    # Vendored research code (e.g. pytARD for low-frequency RIR simulation)
├── logging_setup.py # 函式庫 logging 契約（純 stdlib；由 __init__ 匯入）
├── config/         # Typed recipe/capability models 與共用 loader
├── metrics.py      # Evaluation metrics
├── utils.py        # Utilities
└── recipes.py      # Model construction recipes
```

## Key Design Patterns

- **Modular Architecture**：音訊處理、神經網路建模，與訓練系統之間有清楚的分工。
- **PyTorch Lightning Integration**：所有訓練系統都繼承自 `BaseLightningModule`，統一訓練迴圈的寫法。
- **Configuration-Driven**：用 YAML config 驅動模型建構，讓實驗可重現。
- **Multi-Task Support**：noise suppression（NS）、speaker verification（SV）、target speaker extraction（TSE）共用同一組 base classes——另外還有 `puresound.task.voice_isolation`，是目前開發最活躍的 recipe。Voice isolation 是建構在共用的 NS synthesis skeleton 之上、自成一格的任務（真實錄音列、`mix_mode`、turn-taking、distance/DRR 輔助標籤），不只是通用 NS 的一個變體——詳見 [task/index.md](task/index.md)。
- **Flexible Masking**：支援 complex、real、polar、deep-filter、Wiener、MVDR 等多種 mask。
- **Composable Augmentation**：透過 `AudioEffectAugmentor` 做可插拔式的音訊增強。

## Config 驗證

每一份 recipe 都會在任何 dataset / model / trainer 存在之前，先被解析成 typed
Pydantic model——見 [configuration.md](configuration.md)。`puresound.config.load_recipe`
是唯一入口，沒有繞過它的 dict 路徑。

未知的 key 是錯誤，不是預設值。促成這件事的是兩種失效：打錯字（`porb: 0.5`）以前會讓
該區塊以機率 0 執行而不吭聲；以及刪掉機制不會刪掉它的旋鈕——`augmentation_query_distance`
那 10 個旋鈕在程式碼消失後還活在 33 份 config 裡。

task 這個辨別子決定用哪個 model，所以一個區塊只存在於真的會用它的 task：
`augmentation_realfar` 只是 `voice_isolation` 的欄位；`augmentation_speed` 對增強類
任務是 `speed_range`，對 speaker embedding 是 `speed_change`。整包交給建構子的區塊
（RIR bank loader、room simulator、VAD labeler）只會轉發 recipe 真的寫過的 key，讓那些
元件自己的預設值仍然生效。

這些 model 上的欄位預設值是**行為預設值**：dataset 現在以屬性存取讀它們，所以這裡的
預設值就是 pipeline 實際使用的值。

## 函式庫輸出

執行期間函式庫講的每一句話——語料統計、augmentor 初始化、checkpoint 載入報告、
警告——都走 `puresound` 這個 logger 的 `logging`，不是 `print`。唯一的例外是
`on_test_epoch_end`：它印的是 `--scoring` 的**結果**，不是關於過程的訊息。

`import puresound` 會預設掛上一個 stdout handler，所以從來不設定 logging 的腳本
輸出照舊。要接手控制：

| 目的 | 做法 |
|---|---|
| 完全靜音 | 環境變數 `PURESOUND_LOG_AUTOCONFIG=0` |
| 保留警告、去掉進度訊息 | `logging.getLogger("puresound").setLevel(logging.WARNING)` |
| 自己決定輸出去哪 | `puresound.logging_setup.configure_library_logging(level=..., stream=..., force=True)` |

記錄會被過濾成只有 rank 0 輸出，所以多卡訓練時語料統計只會印一次，而不是每張卡
各印一次。真的需要每個 rank 都講的訊息可以用 `extra={"all_ranks": True}` 豁免。
rank 是從 launcher 的環境變數（`RANK` / `LOCAL_RANK` / `SLURM_PROCID`）讀的，不是
`torch.distributed`——因為這些輸出多半發生在建 dataset 的時候，那時 Lightning 還
沒初始化 process group。
