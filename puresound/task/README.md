# puresound.task

Task-specific dataset implementations, all built on `puresound.dataset`'s
dynamic-augmentation base (`dynamic_base.py`).

| module | status | use |
|---|---|---|
| `ns.py` | active | generic noise-suppression dataset + the shared synthesis skeleton every task dataset specializes |
| `voice_isolation.py` | active | near-field voice isolation — real-recording rows, `mix_mode`, turn-taking, auxiliary distance/DRR/VAD labels; the voice-isolate recipe's dataset |
| `sampler.py` | active | N-way K-shot speaker sampler, incl. a seeded variant for deterministic validation |
| `sv.py` | legacy | speaker verification / embedding dataset |
| `tse.py` | legacy | target speaker extraction dataset |

Legacy modules are kept working but frozen: no new features, no rewrites.

Full API reference: `docs/task/`.
