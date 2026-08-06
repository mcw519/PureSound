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
