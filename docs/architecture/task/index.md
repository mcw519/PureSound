# puresound.task

繁體中文版本：[`index.zh-TW.md`](index.zh-TW.md)

Task-specific datasets, their samplers, and the synthesis components they
compose. `puresound/task/__init__.py` is empty; import from each submodule.

## Sub-modules

| Module | Status | Description |
|--------|--------|-------------|
| [task.ns](ns.md) | active | noise-suppression dataset and the shared synthesis skeleton (row-type hooks, collate) |
| [task.voice_isolation](voice_isolation.md) | active | near-field voice isolation: real-recording rows, session rows, `mix_mode`, scalar labels |
| [task.sampler](sampler.md) | active | speaker batch sampler: seeded validation, length schedule, epoch keys, source weights |
| task.device_chain | active | capture and transmission chain (SRC, IIR, HPF, volume, compressor, A/D, codec, packet loss) applied to a finished mixture; see [Device chain](../../algorithms/augmentation/device_chain.md) |
| task.noise_stage | active | non-speech sources: recorded noise at an SNR, white noise, absolute capture floor; see [Scene construction](../../algorithms/augmentation/scene_construction.md) |
| task.overlap_gating | active | who talks when: per-frame Bernoulli overlap or conversational turn-taking; see [Scene construction](../../algorithms/augmentation/scene_construction.md) |
| task.paired_views | active | second device-chain draw of the same mixture, and its source-preserving collation |
| task.session_rows | active | scripted multi-turn session rows with per-frame and per-turn identity labels |
| task.trace | active | passive per-stage recorder of one synthesised row for the web pipeline inspector; see [task.ns](ns.md#tracing-a-row) |
| [task.sv](sv.md) | legacy | speaker verification/embedding dataset |
| [task.tse](tse.md) | legacy | target speaker extraction dataset |

The synthesis components share three contracts: stage order is load-bearing
(every stage draws from the shared RNG stream), a disabled stage draws nothing,
and a component is composed once per dataset in `rebind_augmentation_blocks()`
so a curriculum that changes a block reaches it.

Legacy modules are kept working but frozen: no new features, no rewrites.
