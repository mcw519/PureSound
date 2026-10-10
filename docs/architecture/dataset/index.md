# puresound.dataset

繁體中文版本：[index.zh-TW.md](index.zh-TW.md)

Metafile parsing, corpus preparation and the base classes the task datasets
build on. `puresound/dataset/__init__.py` is empty: there is no package-level
re-export, so every class is imported from its own submodule (e.g.
`from puresound.dataset.dynamic_base import DynamicBaseDataset`). The
`dataset.corpus` sub-package does re-export its record, scan and resample helpers.

## Sub-modules

| Module | Status | Description |
|--------|--------|-------------|
| [dataset.corpus](../../usage/data_preparation.md) | active | corpus preparation: scan, resample, split, write metafiles |
| [dataset.parser](parser.md) | active | CSV metafile parser |
| [dataset.dynamic_base](dynamic_base.md) | active | dynamic-augmentation base dataset (the task datasets build on it) |
| [dataset.kaldi_base](kaldi_base.md) | legacy | Kaldi-format (scp) dataset: the frozen SV/TSE recipes train on it, and the runner's `--scoring` / `--inference` stages read a recipe's test folder through it |
