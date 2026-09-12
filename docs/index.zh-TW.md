# PureSound 文件

English: [index.md](index.md)

這一頁只負責導覽。安裝與第一個可執行範例請看[專案 README](../README.zh-TW.md)。

## 從這裡開始

| 目的 | 文件 |
| --- | --- |
| 執行已發布模型 | [Model Zoo 與 Web UI](web.zh-TW.md) |
| 使用 YAML 訓練 | [Recipe 設定](configuration.md) |
| 查詢音訊與 RIR API | [Audio](audio/index.zh-TW.md) |
| 理解訓練資料合成 | [資料增強](augmentation/index.zh-TW.md) |
| 選擇模型或 loss | [神經網路](nnet/index.zh-TW.md) |
| 使用 dataset | [Datasets](dataset/index.zh-TW.md) |
| 理解訓練模組 | [訓練系統](system/index.zh-TW.md) |
| 部署串流 ONNX | [Streaming](streaming/index.zh-TW.md) |
| 實作 task dataset | [Tasks](task/index.zh-TW.md) |
| 使用評估指標 | [Metrics](metrics.zh-TW.md) |
| 使用共用工具 | [Utilities](utils.zh-TW.md) |

## Package 分工

| Package | 用途 |
| --- | --- |
| `puresound.audio` | 音訊 I/O、DSP、資料增強與 RIR |
| `puresound.config` | YAML recipe 載入與型別驗證 |
| `puresound.dataset` | Manifest parser 與 dataset 基底 |
| `puresound.nnet` | 模型、layers、maskers 與 losses |
| `puresound.streaming` | ONNX export 與串流 runtime |
| `puresound.system` | PyTorch Lightning modules |
| `puresound.task` | 各任務的 datasets 與 samplers |
| `puresound.inference` | Model Zoo 與統一推論 runtime |

可執行 recipe 放在 `egs/`，獨立 ONNX runtime 放在 `sdk/python/`。

## 文件原則

- 這裡只放仍有效的 API 與 release 流程。
- 生成報告與私人實驗紀錄不屬於公開文件。
- 英文檔名使用 `.md`，繁中版本使用 `.zh-TW.md`。
- 文件與實作不一致時，以原始碼的 constructor signature 為準。
