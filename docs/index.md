# PureSound Documentation

繁體中文版本：[index.zh-TW.md](index.zh-TW.md)

PureSound is a modular audio processing and deep learning framework for speech enhancement, speaker verification, and target speaker extraction tasks.

## Package Version

`0.1`

## Module Overview

| Module | Description |
|--------|-------------|
| [puresound.audio](audio/index.md) | Audio I/O, DSP, augmentation, and spectrum utilities |
| [puresound.dataset](dataset/index.md) | Dataset base classes and parsers |
| [puresound.nnet](nnet/index.md) | Neural network architectures and building blocks |
| [puresound.system](system/index.md) | PyTorch Lightning training systems |
| [puresound.task](task/index.md) | Task-specific dataset implementations |
| [puresound.streaming](streaming/index.md) | Streaming inference and ONNX Runtime deployment |
| [puresound.metrics](metrics.md) | Audio quality evaluation metrics |
| [puresound.utils](utils.md) | General utility functions |
| [puresound.recipes](recipes.md) | High-level model initialization recipes |

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

- **Modular Architecture**: Clear separation between audio processing, neural network modeling, and training systems.
- **PyTorch Lightning Integration**: All training systems extend `BaseLightningModule` for standardized training loops.
- **Configuration-Driven**: YAML-based configs drive model construction for reproducible experiments.
- **Multi-Task Support**: Shared base classes for noise suppression (NS), speaker verification (SV), and target speaker extraction (TSE) — plus `puresound.task.voice_isolation`, the most actively developed recipe today. Voice isolation is its own task built on the shared NS synthesis skeleton (real-recording rows, `mix_mode`, turn-taking, auxiliary distance/DRR labels), not just a variant of generic NS — see [task/index.md](task/index.md).
- **Flexible Masking**: Support for complex, real, polar, deep-filter, Wiener, and MVDR masks.
- **Composable Augmentation**: Pluggable audio augmentation via `AudioEffectAugmentor`.
