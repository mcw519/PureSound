# puresound.task

任務專屬的 dataset 實作，全部建立在 `puresound.dataset` 的動態增強（dynamic-augmentation）基底
（`dynamic_base.py`）之上。

| 模組 | 狀態 | 用途 |
|---|---|---|
| `ns.py` | active | 通用的 noise-suppression dataset，也是其他任務 dataset 共用並特化的合成骨架 |
| `voice_isolation.py` | active | 近場人聲分離（voice isolation）—— 真實錄音列（real-recording rows）、`mix_mode`、輪流說話（turn-taking）、距離／DRR／VAD 輔助標籤；voice-isolate 專案使用的 dataset |
| `sampler.py` | active | N-way K-shot 語者取樣器，含用於驗證階段可重現結果的 seeded 變體 |
| `sv.py` | legacy | 語者驗證／embedding dataset |
| `tse.py` | legacy | 目標語者萃取（target speaker extraction）dataset |

Legacy 模組維持可運作但已凍結：不再新增功能，也不再重構。

完整 API 參考文件：`docs/task/`。
