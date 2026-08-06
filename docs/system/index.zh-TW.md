# puresound.system

English version: [`index.md`](index.md)

PyTorch Lightning 訓練系統。

## Sub-modules

| Module | Status | Description |
|--------|--------|-------------|
| [system.base](base.zh-TW.md) | active | base Lightning module（loss registry、optimizer/scheduler 管線、warmup、GPU-batched VAD labeling） |
| [system.siso](siso.zh-TW.md) | active | 單輸入 enhancement trainer（`EncDecMaskBase`）+ embedding classifier（`EncPredClassBase`，legacy） |
| [system.optim](optim.zh-TW.md) | active | optimizer 與 LR-scheduler 工廠 |
| [system.logger](logger.zh-TW.md) | active | 訓練 metric 累積 logger |
| [system.miso](miso.zh-TW.md) | legacy | TSE 用的 conditional（speaker-aware）enhancement trainer |

## Top-level Exports

```python
from puresound.system import (
    EncDecCondMaskBase,   # MISO: conditional (speaker-aware) enhancement  [legacy]
    EncDecMaskBase,       # SISO: mask/mapping enhancement                  [active]
    EncPredClassBase,     # SISO: classification (speaker embedding)        [legacy]
)
```
