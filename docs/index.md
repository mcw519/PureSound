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
| [puresound.web](web.md) | Dependency-light Model Zoo browser workspace and HTTP API |
| [puresound.metrics](metrics.md) | Audio quality evaluation metrics |
| [puresound.utils](utils.md) | General utility functions |
| `puresound.logging_setup` | Where the library's runtime output goes, and how to take it over |
| [`puresound.config`](configuration.md) | Pydantic task recipes, capability models, migration, and validation |
| [puresound.recipes](recipes.md) | High-level model initialization recipes |

There is also a cross-module [Data Augmentation DSP Handbook](augmentation/index.md),
organising the pipeline's signal processing techniques along two axes — algorithm
and engineering: derivations and assumptions, config mapping, RNG determinism
contracts, ordering constraints, and pitfalls.

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
├── logging_setup.py # Library logging contract (stdlib only; imported by __init__)
├── config/         # Typed recipe/capability models and the shared loader
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

## Configuration Validation

Every recipe is parsed into a typed Pydantic model before a dataset, model or
trainer exists — see [configuration.md](configuration.md). `puresound.config.load_recipe`
is the only entry point; there is no dict path around it.

An unknown key is an error, not a default. Two failures motivated this: a typo
(`porb: 0.5`) used to run its block at probability zero without complaint, and
removing a mechanism did not remove its knobs — the ten
`augmentation_query_distance` knobs outlived their code in 33 configs.

The task discriminator picks the model, so a block only exists for the tasks
that consume it: `augmentation_realfar` is a `voice_isolation` field and nothing
else, and `augmentation_speed` means `speed_range` to the enhancement tasks and
`speed_change` to speaker embedding. Blocks handed straight to a constructor
(the RIR bank loader, the room simulator, the VAD labeler) forward only the keys
the recipe actually wrote, so the component's own defaults still apply.

Field defaults on those models are behavioural: the datasets read them by
attribute, so a default here is the value the pipeline uses.

## Library Output

Everything the library says at runtime — corpus statistics, augmentor setup,
checkpoint-load reports, warnings — goes through `logging` on the `puresound`
logger, not `print`. The single exception is `on_test_epoch_end`, which prints
the metric scores because those are the result of `--scoring`, not a note about
it.

Importing `puresound` attaches one stdout handler by default, so scripts that
never configure logging keep the output they had. To take control:

| Goal | How |
|---|---|
| Silence the library entirely | `PURESOUND_LOG_AUTOCONFIG=0` in the environment |
| Keep warnings, drop progress chatter | `logging.getLogger("puresound").setLevel(logging.WARNING)` |
| Route it yourself | `puresound.logging_setup.configure_library_logging(level=..., stream=..., force=True)` |

Records are filtered to rank zero, so a multi-GPU run prints the corpus
statistics once rather than once per rank. A record that genuinely belongs on
every rank opts out with `extra={"all_ranks": True}`. Rank is read from the
launcher's environment (`RANK` / `LOCAL_RANK` / `SLURM_PROCID`) rather than
`torch.distributed`, because most of this output is emitted while datasets are
built — before Lightning initializes the process group.
