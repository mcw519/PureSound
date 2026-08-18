# 目標語者萃取（Target Speaker Extraction，TSE）

> **狀態：已凍結的 legacy。** 不再新增功能，也不再重構——僅維持可運作以供參考。

English version: [`README.md`](README.md)

目標語者萃取採用 **enrollment-based** 的方式：給定一段含噪混音，以及該目標語者另一段乾淨的
enrollment 語音，模型會以從 enrollment 音檔萃取出的語者 embedding 為條件（MISO trainer
`EncDecCondMaskBase`），從混音中萃取出該語者的聲音。這與本專案其他地方的近場
`egs/voice_isolate` recipe 所用的線索不同——後者是 enrollment-free，只靠近／遠場的
DRR（direct-to-reverberant ratio，直混比）對比來判斷。

## 訓練／推論

```bash
# 準備 metadata
python prepare_metafile.py --help

# 訓練
python main.py --training config/default_config.yaml

# 推論
python main.py --inference --ckpt_path ckpt_path config/default_config.yaml
```

`--training` 搭配 `--ckpt_path` 是續訓（恢復 optimizer／scheduler／epoch）；只想用某個
checkpoint 的權重重新熱身訓練，則改用 `--pretrained_ckpt_path`。

## 完整參考文件

本 README 只涵蓋這個 recipe 的最小指令集。資料集（混音組成、enrollment 取樣、augmentation）與
訓練系統（MISO 架構、各元件的 learning rate 縮放）的完整說明，請見：

- [`docs/task/tse.md`](../../docs/task/tse.md) — `TargetSpeakerExtractDataset`
- [`docs/system/miso.md`](../../docs/system/miso.md) — `EncDecCondMaskBase`
