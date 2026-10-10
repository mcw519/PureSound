# PureSound documentation

Traditional Chinese: [index.zh-TW.md](index.zh-TW.md)

Installation and a first run are in the [project README](../README.md). The
documentation is in three parts; every page has a Traditional Chinese twin
(`.zh-TW.md`) unless noted.

## Architecture

How the packages fit together and how data flows from corpus to deployment.

| page | covers |
| --- | --- |
| [Architecture overview](architecture/index.md) | package map; corpus → synthesis → training → gate → export → deployment; the config system and the model zoo |
| [Datasets](architecture/dataset/index.md) | metafile parser, `DynamicBaseDataset` |
| [Tasks](architecture/task/index.md) | task datasets, speaker sampler, synthesis stages |
| [Training systems](architecture/system/index.md) | Lightning modules, training driver, optimizer factory |
| [Utilities](architecture/utils.md) | `puresound.utils` helpers |
| [Repository layout](repository_layout.md) | where a file belongs, run and release naming, the documentation standard |

## Algorithms

What each model, loss, synthesis stage and metric computes, and why.

| page | covers |
| --- | --- |
| [Models](algorithms/models/index.md) | backbones, building blocks, features, maskers |
| [Losses](algorithms/losses/index.md) | the loss library |
| [Audio](algorithms/audio/index.md) | audio I/O, DSP, room acoustics and RIR generation |
| [Data augmentation](algorithms/augmentation/index.md) | how a training mixture is built |
| [Metrics](algorithms/metrics.md) | objective metrics |

## Usage

How to prepare data, train, evaluate, export and deploy.

| page | covers |
| --- | --- |
| [Data preparation](usage/data_preparation.md) | corpora to metafiles |
| [Recipe configuration](usage/configuration.md) | recipe fields |
| [Recipes](usage/recipes.md) | building models and losses from a recipe |
| [Evaluation](usage/evaluation.md) | the benchmark gate |
| [Streaming](usage/streaming/index.md) | ONNX export and the streaming runtime |
| [Web UI and inference API](usage/web.md) | the model zoo from the command line and the browser |
