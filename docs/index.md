# PureSound Documentation

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
├── task/           # Task-specific datasets (NS, SV, TSE)
├── metrics.py      # Evaluation metrics
├── utils.py        # Utilities
└── recipes.py      # Model construction recipes
```

## Key Design Patterns

- **Modular Architecture**: Clear separation between audio processing, neural network modeling, and training systems.
- **PyTorch Lightning Integration**: All training systems extend `BaseLightningModule` for standardized training loops.
- **Configuration-Driven**: YAML-based configs drive model construction for reproducible experiments.
- **Multi-Task Support**: Shared base classes for noise suppression (NS), speaker verification (SV), and target speaker extraction (TSE).
- **Flexible Masking**: Support for complex, real, polar, deep-filter, Wiener, and MVDR masks.
- **Composable Augmentation**: Pluggable audio augmentation via `AudioEffectAugmentor`.
