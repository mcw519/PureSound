# 串流部署

English: [index.md](index.md)

`puresound.streaming` 把 checkpoint 匯出成逐幀的 ONNX graph 加一份 JSON manifest，並即時
執行它。Python（或可攜式 SDK）負責音訊緩衝、STFT/iSTFT、狀態張量，以及 manifest 記錄的
graph 後處理階段；ONNX Runtime 一次跑一個特徵幀。

| 頁面 | 狀態 | 涵蓋 |
|--------|--------|-------------|
| [DPCRN 串流 ONNX](dpcrn_onnx.zh-TW.md) | 現役 | 匯出、驗證與執行；look-ahead 緩衝；runtime 套用的 dry blend 與 onset guard；輔助頭輸出。每一顆已發布 checkpoint 的部署路徑 |
| [DPARN 串流 ONNX](dparn_onnx.zh-TW.md) | legacy | 固定前端 DPARN 的特徵幀匯出與 runtime |
| [可攜式 SDK](../../../sdk/python/README.zh-TW.md) | 現役 | `puresound_streaming`：只靠 NumPy 與 ONNX Runtime 的同一個 runtime，給不帶訓練堆疊的部署用 |

已發布的匯出放在各自 checkpoint 旁邊，位於 `egs/voice_isolate/pretrained_ckpt/streaming/`
與 `egs/noise_suppression/pretrained_ckpt/streaming/`；model zoo（`model_zoo/catalog.yaml`、
`puresound infer`、[web UI](../web.zh-TW.md)）執行的就是它們。
