# puresound.system

PyTorch Lightning training systems for speech enhancement and speaker verification.

## Sub-modules

| Module | Description |
|--------|-------------|
| [system.base](base.md) | Base PyTorch Lightning module |
| [system.siso](siso.md) | Single-Input Single-Output training system |
| [system.miso](miso.md) | Multi-Input Single-Output (speaker-conditional) training system |
| [system.optim](optim.md) | Optimizer and learning rate scheduler factory |
| [system.logger](logger.md) | Training metric accumulation logger |

## Top-level Exports

```python
from puresound.system import (
    EncDecCondMaskBase,   # MISO: conditional (speaker-aware) enhancement
    EncDecMaskBase,       # SISO: standard enhancement
    EncPredClassBase,     # SISO: classification (speaker embedding)
)
```
