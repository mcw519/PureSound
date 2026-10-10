# puresound.dataset

English version: [index.md](index.md)

Metafile 解析、語料準備，以及 task dataset 所建構其上的 base classes。
`puresound/dataset/__init__.py` 是空檔案：沒有 package 層級的 re-export，每個
class 都從各自的子模組 import（例如
`from puresound.dataset.dynamic_base import DynamicBaseDataset`）。
`dataset.corpus` 子套件則會 re-export 它的 record、scan 與 resample helper。

## Sub-modules

| Module | Status | Description |
|--------|--------|-------------|
| [dataset.corpus](../../usage/data_preparation.zh-TW.md) | active | 語料準備：掃描、重取樣、切分、寫 metafile |
| [dataset.parser](parser.zh-TW.md) | active | CSV metafile 解析器 |
| [dataset.dynamic_base](dynamic_base.zh-TW.md) | active | 動態增強的 base dataset（task datasets 都建構於其上） |
| [dataset.kaldi_base](kaldi_base.zh-TW.md) | legacy | Kaldi 格式（scp）dataset：已凍結的 SV/TSE recipe 以它訓練，runner 的 `--scoring` / `--inference` stage 也透過它讀取 recipe 的 test folder |
