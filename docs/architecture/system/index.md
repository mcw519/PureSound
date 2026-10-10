# puresound.system

繁體中文版本：[`index.zh-TW.md`](index.zh-TW.md)

PyTorch Lightning training systems, the shared training driver, and the inference-only stages that run after a model's mask.

## Sub-modules

| Module | Status | Description |
|--------|--------|-------------|
| [system.base](base.md) | active | base Lightning module: loss registry and provider-based loss dispatch (`invoke_loss`, `reduce_losses`), optimizer/scheduler plumbing, LR warmup, GPU-batched VAD labeling, checkpoint reload |
| [system.siso](siso.md) | active | single-input enhancement trainer (`EncDecMaskBase`) + embedding classifier (`EncPredClassBase`, legacy) |
| [system.optim](optim.md) | active | optimizer and LR-scheduler factory over the typed `optimizer` / `scheduler` recipe blocks |
| [system.logger](logger.md) | active | per-epoch metric accumulator (`Logging`) |
| `system.runner` | active | the driver every recipe's `main.py` hands off to: CLI (`build_arg_parser`), run seeding (`seed_run`), train/validation dataloaders (`build_dataloaders`, whose workers start in `seed_worker`), Lightning trainer and DDP strategy, warm start (`load_warm_start`), and the `--training` / `--scoring` / `--inference` / `--dump_training_samples` stages (`run_stages`) |
| `system.curriculum` | active | `CurriculumCallback`: applies a recipe curriculum's per-epoch loss weights on the module and logs every scheduled value; the dataset half of a curriculum travels with each item from the sampler ([configuration](../../usage/configuration.md)) |
| `system.sampling` | active | carries a `CoverageSampler` walk across runs: records it in every checkpoint and places it for `--ckpt_path` / `--pretrained_ckpt_path` ([task.sampler](../task/sampler.md#class-coveragesampler)) |
| `system.metric_gan` | active | MetricGAN training term for `EncDecMaskBase(metric_gan=...)`: `MetricGanConfig`, the asynchronous PESQ replay buffer (`PesqReplay`), discriminator and generator losses |
| `system.paired_views` | active | auxiliary-view consistency for `EncDecMaskBase(paired_view_consistency=...)`: `PairedViewConsistencyConfig` and `paired_view_loss`, one extra rank-synchronized forward shared by every loss that declares `paired_output` |
| `system.postprocess` | active | `Postprocessor`: inference-only over-suppression relief (`dry_blend`, `spec_floor`), its suppression ceiling, and its manifest entry |
| `system.presence_gate` | active | `PresenceGate`: inference-only near-presence gain applied after the blend, from a bottleneck readout or a trained presence head's logits |
| `system.onset_guard` | active | `OnsetGuard`: inference-only onset protection that returns the dry input until a talker has been heard for a sustained stretch; an offline torch face (`apply`) and a hop-by-hop numpy face (`streaming_state` / `step`) for the ONNX runtime, bit-identical by construction |
| [system.miso](miso.md) | legacy | conditional (speaker-aware) enhancement trainer for TSE |

The three inference-only stages are not learned and are not part of an exported graph. `EncDecMaskBase.forward` applies them in a fixed order — `Postprocessor`, then `PresenceGate`, then `OnsetGuard` — and a deployment running the graph alone is running a different system. The streaming export therefore records the `Postprocessor` and `OnsetGuard` settings in its manifest, and the ONNX runtime applies them ([streaming](../../usage/streaming/index.md)); `PresenceGate` has a manifest form (`as_manifest`) but runs only in the PyTorch `forward`, and the streaming export does not carry it.

## Seeding and DataLoader workers

`runner.seed_run(seed)` calls `seed_everything(seed, workers=True)`. Lightning re-applies the same seed on every DDP rank, so without `workers=True` worker *k* of every rank starts from the same random state: the ranks draw different speakers (the sampler offsets its stream by rank) but the same SNRs, noise clips, rooms and mix modes for them, and a two-GPU run sees one GPU's worth of augmentation. Items that carry their own seed (`CoverageSampler`, the seeded validation sampler) reseed every RNG at the top of `__getitem__` and draw the same either way.

Both training loaders start their workers in `runner.seed_worker`. It runs Lightning's per-rank worker seeding when `seed_everything(..., workers=True)` asked for it -- Lightning adds its own initialiser only to a loader that has none -- and then pins every thread pool in the worker to one thread (`puresound.utils.pin_thread_pools`): torch's intra- and inter-op pools, numba, and the BLAS/OpenMP pools numpy and scipy use, which a worker otherwise inherits at one thread per core.

## Top-level Exports

```python
from puresound.system import (
    EncDecCondMaskBase,   # MISO: conditional (speaker-aware) enhancement  [legacy]
    EncDecMaskBase,       # SISO: mask/mapping enhancement                  [active]
    EncPredClassBase,     # SISO: classification (speaker embedding)        [legacy]
)
```

Every other name is imported from its own submodule (e.g. `from puresound.system import runner`, `from puresound.system.postprocess import Postprocessor`).
