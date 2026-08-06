# puresound.dataset

繁體中文版本：[index.zh-TW.md](index.zh-TW.md)

Dataset sub-package providing base classes and parsers for audio datasets.
`puresound/dataset/__init__.py` is empty — there is no package-level
re-export, so every class is imported from its own submodule (e.g.
`from puresound.dataset.dynamic_base import DynamicBaseDataset`), as the
examples in each sub-module doc below show.

## Sub-modules

| Module | Status | Description |
|--------|--------|-------------|
| [dataset.parser](parser.md) | active | CSV metafile parser |
| [dataset.dynamic_base](dynamic_base.md) | active | dynamic-augmentation base dataset (the task datasets build on it) |
| [dataset.kaldi_base](kaldi_base.md) | legacy | Kaldi-format (scp) dataset, used by the frozen SV/TSE recipes |
