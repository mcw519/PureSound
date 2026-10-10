# puresound.task

English version: [`index.md`](index.md)

各任務的 datasets、它們的 samplers，以及它們組合使用的合成元件。
`puresound/task/__init__.py` 是空檔案；請從各子模組 import。

## Sub-modules

| Module | Status | Description |
|--------|--------|-------------|
| [task.ns](ns.zh-TW.md) | active | noise-suppression dataset 與共用的合成骨架（row-type hooks、collate） |
| [task.voice_isolation](voice_isolation.zh-TW.md) | active | 近場 voice isolation：真實錄音列、session 列、`mix_mode`、scalar labels |
| [task.sampler](sampler.zh-TW.md) | active | speaker batch sampler：seeded validation、length schedule、epoch keys、來源權重 |
| task.device_chain | active | 套在完成混音上的收音與傳輸鏈（SRC、IIR、HPF、volume、compressor、A/D、codec、packet loss）；見 [Device chain](../../algorithms/augmentation/device_chain.zh-TW.md) |
| task.noise_stage | active | 非語音來源：依 SNR 的錄製噪音、白噪音、絕對位準的收音底噪；見 [Scene construction](../../algorithms/augmentation/scene_construction.zh-TW.md) |
| task.overlap_gating | active | 誰在何時說話：逐 frame Bernoulli 重疊或對話式 turn-taking；見 [Scene construction](../../algorithms/augmentation/scene_construction.zh-TW.md) |
| task.paired_views | active | 對同一個混音再抽一次裝置鏈，以及保留來源對應的 collation |
| task.session_rows | active | 以 script 排出的多輪 session 列，帶逐 frame 與逐 turn 的身分 labels |
| task.trace | active | 為 web 管線檢視器逐階段被動記錄合成的一列；見 [task.ns](ns.zh-TW.md#追蹤一列) |
| [task.sv](sv.zh-TW.md) | legacy | speaker verification/embedding dataset |
| [task.tse](tse.zh-TW.md) | legacy | target speaker extraction dataset |

這些合成元件共用三條契約：各級順序是承重的（每一級都從共用的 RNG stream 抽樣）、
未啟用的級不抽任何東西，以及元件在 `rebind_augmentation_blocks()` 中每個 dataset
組一次，讓 curriculum 改動區塊時也能傳到它。

Legacy 模組維持可運作但已凍結：不新增功能、不重寫。
