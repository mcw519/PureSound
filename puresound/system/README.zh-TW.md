# puresound.system

English version: [`README.md`](README.md)

Trainer，全部建立在 [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) 之上。

| 類別 | 檔案 | 狀態 | 用途 |
|---|---|---|---|
| `EncDecMaskBase` | siso.py | active | mask/mapping 增強（enhancement）（voice-isolate/NS 的 trainer） |
| `EncPredClassBase` | siso.py | legacy | 語者 embedding 分類 |
| `EncDecCondMaskBase` | miso.py | legacy | 語者條件式（speaker-conditioned）增強（TSE） |

`base.py` 承載共用的基礎設施（loss registry、optimizer/scheduler 註冊、
warmup）；`optim.py` 依 recipe config 建構 optimizer/scheduler。完整 API
參考文件：`docs/system/`。
