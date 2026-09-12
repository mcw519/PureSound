# puresound.streaming

English version: [index.md](index.md)

給低延遲部署用的 streaming 推論工具。Python 端負責音訊緩衝、STFT/iSTFT 與
ONNX Runtime 狀態管理；ORT 每次只跑一個 feature frame。

## Modules

| Module | Status | Description |
|--------|--------|-------------|
| [streaming/dpcrn_onnx](dpcrn_onnx.md) | active | DPCRN 逐 frame 的 ONNX export 與 ORT streaming（已發佈的 voice-isolate 部署路徑；透過 future-buffering 支援有界的 look-ahead） |
| [streaming/dparn_onnx](dparn_onnx.md) | legacy | DPARN 逐 feature frame 的 ONNX export 與 ORT streaming |
