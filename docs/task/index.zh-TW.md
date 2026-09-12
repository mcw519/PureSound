# puresound.task

English version: [`index.md`](index.md)

Task-specific 的 dataset 實作。

## Sub-modules

| Module | Status | Description |
|--------|--------|-------------|
| [task.ns](ns.zh-TW.md) | active | 通用 noise-suppression dataset + 共用的 synthesis skeleton |
| [task.voice_isolation](voice_isolation.zh-TW.md) | active | 近場 voice isolation（real-recording rows、mix_mode、turn-taking、輔助 labels） |
| [task.sampler](sampler.zh-TW.md) | active | N-way K-shot speaker sampler（用於 deterministic validation 的 seeded 版本） |
| [task.sv](sv.zh-TW.md) | legacy | speaker verification/embedding dataset |
| [task.tse](tse.zh-TW.md) | legacy | target speaker extraction dataset |

Legacy 模組維持可運作但已凍結：不新增功能、不重寫。
