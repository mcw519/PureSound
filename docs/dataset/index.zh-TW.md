# puresound.dataset

English version: [index.md](index.md)

Dataset 子套件，提供音訊資料集的 base classes 與 parser。
`puresound/dataset/__init__.py` 是空檔案——沒有 package 層級的 re-export，
所以每個 class 都要從各自的子模組 import（例如
`from puresound.dataset.dynamic_base import DynamicBaseDataset`），下方各子模組
文件裡的範例也是這樣寫的。

## Sub-modules

| Module | Status | Description |
|--------|--------|-------------|
| [dataset.parser](parser.zh-TW.md) | active | CSV metafile 解析器 |
| [dataset.dynamic_base](dynamic_base.zh-TW.md) | active | 動態增強的 base dataset（task datasets 都建構於其上） |
| [dataset.kaldi_base](kaldi_base.zh-TW.md) | legacy | Kaldi 格式（scp）dataset，供已凍結的 SV/TSE recipes 使用 |
