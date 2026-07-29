# puresound.system

PyTorch Lightning training systems.

## Sub-modules

| Module | Status | Description |
|--------|--------|-------------|
| [system.base](base.md) | active | base Lightning module (loss registry, optimizer/scheduler plumbing, warmup) |
| [system.siso](siso.md) | active | single-input enhancement trainer (`EncDecMaskBase`) + embedding classifier (`EncPredClassBase`, legacy) |
| [system.optim](optim.md) | active | optimizer and LR-scheduler factory |
| [system.logger](logger.md) | active | training metric accumulation logger |
| [system.miso](miso.md) | legacy | conditional (speaker-aware) enhancement trainer for TSE |

## Top-level Exports

```python
from puresound.system import (
    EncDecCondMaskBase,   # MISO: conditional (speaker-aware) enhancement  [legacy]
    EncDecMaskBase,       # SISO: mask/mapping enhancement                  [active]
    EncPredClassBase,     # SISO: classification (speaker embedding)        [legacy]
)
```
