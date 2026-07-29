# puresound.system

Trainers, all built on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

| class | file | status | use |
|---|---|---|---|
| `EncDecMaskBase` | siso.py | active | mask/mapping enhancement (the voice-isolate/NS trainer) |
| `EncPredClassBase` | siso.py | legacy | speaker-embedding classification |
| `EncDecCondMaskBase` | miso.py | legacy | speaker-conditioned enhancement (TSE) |

`base.py` carries the shared plumbing (loss registry, optimizer/scheduler
registration, warmup); `optim.py` builds optimizers/schedulers from the recipe
config. Full API reference: `docs/system/`.
