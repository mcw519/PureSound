# puresound.task

Task-specific dataset implementations.

## Sub-modules

| Module | Status | Description |
|--------|--------|-------------|
| [task.ns](ns.md) | active | generic noise-suppression dataset + the shared synthesis skeleton |
| [task.voice_isolation](voice_isolation.md) | active | near-field voice isolation (real-recording rows, mix_mode, turn-taking, aux labels) |
| [task.sampler](sampler.md) | active | N-way K-shot speaker sampler (seeded variant for deterministic validation) |
| [task.sv](sv.md) | legacy | speaker verification/embedding dataset |
| [task.tse](tse.md) | legacy | target speaker extraction dataset |

Legacy modules are kept working but frozen: no new features, no rewrites.
