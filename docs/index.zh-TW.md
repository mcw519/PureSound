# PureSound 文件

English: [index.md](index.md)

安裝與第一次執行請看[專案 README](../README.zh-TW.md)。文件分成三部分；除非另外
註明，每一頁都有英文版與繁體中文版（`.zh-TW.md`）。

## 架構

各 package 如何組合，以及資料如何從語料一路流到部署。

| 頁面 | 涵蓋 |
| --- | --- |
| [架構總覽](architecture/index.zh-TW.md) | package 分工；語料 → 合成 → 訓練 → 閘門 → 匯出 → 部署；設定系統與 model zoo |
| [Datasets](architecture/dataset/index.zh-TW.md) | metafile parser、`DynamicBaseDataset` |
| [Tasks](architecture/task/index.zh-TW.md) | 任務 dataset、speaker sampler、合成階段 |
| [訓練系統](architecture/system/index.zh-TW.md) | Lightning module、訓練 driver、optimizer 工廠 |
| [Utilities](architecture/utils.zh-TW.md) | `puresound.utils` 工具函式 |
| [儲存庫佈局](repository_layout.zh-TW.md) | 檔案該放哪、run 與版本命名、文件規範 |

## 演算法

每個模型、loss、合成階段與指標計算什麼，以及為什麼。

| 頁面 | 涵蓋 |
| --- | --- |
| [模型](algorithms/models/index.zh-TW.md) | backbone、組成區塊、特徵、masker |
| [Loss](algorithms/losses/index.zh-TW.md) | loss 函式庫 |
| [Audio](algorithms/audio/index.zh-TW.md) | 音訊 I/O、DSP、房間聲學與 RIR 生成 |
| [資料增強](algorithms/augmentation/index.zh-TW.md) | 一筆訓練混音如何組成 |
| [指標](algorithms/metrics.zh-TW.md) | 客觀指標 |

## 使用方式

如何準備資料、訓練、評測、匯出與部署。

| 頁面 | 涵蓋 |
| --- | --- |
| [資料準備](usage/data_preparation.zh-TW.md) | 從語料到 metafile |
| [Recipe 設定](usage/configuration.zh-TW.md) | recipe 欄位 |
| [Recipes](usage/recipes.zh-TW.md) | 從 recipe 建立模型與 loss |
| [評測](usage/evaluation.zh-TW.md) | 評測閘門 |
| [Streaming](usage/streaming/index.zh-TW.md) | ONNX 匯出與串流 runtime |
| [Web UI 與推論 API](usage/web.zh-TW.md) | 從指令列與瀏覽器使用 model zoo |
