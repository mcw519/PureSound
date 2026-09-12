# PureSound documentation

Traditional Chinese: [index.zh-TW.md](index.zh-TW.md)

Use this page to find the right document. Installation and first-run commands
are in the [project README](../README.md).

## Start here

| Goal | Document |
| --- | --- |
| Run a released model | [Model Zoo and Web UI](web.md) |
| Train from YAML | [Recipe configuration](configuration.md) |
| Understand audio and RIR APIs | [Audio](audio/index.md) |
| Understand data synthesis | [Data augmentation](augmentation/index.md) |
| Choose a model or loss | [Neural networks](nnet/index.md) |
| Work with datasets | [Datasets](dataset/index.md) |
| Understand training modules | [Training systems](system/index.md) |
| Deploy streaming ONNX | [Streaming](streaming/index.md) |
| Implement a task dataset | [Tasks](task/index.md) |
| Use evaluation metrics | [Metrics](metrics.md) |
| Use shared helpers | [Utilities](utils.md) |

## Package layout

| Package | Responsibility |
| --- | --- |
| `puresound.audio` | Audio I/O, DSP, augmentation, and RIR processing |
| `puresound.config` | Typed YAML recipe loading and validation |
| `puresound.dataset` | Manifest parsing and dataset bases |
| `puresound.nnet` | Models, layers, maskers, and losses |
| `puresound.streaming` | ONNX export and streaming runtime |
| `puresound.system` | PyTorch Lightning modules |
| `puresound.task` | Task-specific datasets and samplers |
| `puresound.inference` | Model Zoo and unified inference runtime |

Runnable recipes live in `egs/`. The standalone ONNX runtime lives in
`sdk/python/`.

## Conventions

- Active APIs and release workflows are documented here.
- Generated reports and private experiment records are not part of the public
  documentation.
- English files use `.md`; Traditional Chinese files use `.zh-TW.md`.
- Constructor signatures in source code are authoritative when a reference page
  and implementation differ.
