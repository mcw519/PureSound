# puresound.streaming

繁體中文版本：[index.zh-TW.md](index.zh-TW.md)

Streaming inference utilities for low-latency deployment. Python owns audio
buffering, STFT/iSTFT and ONNX Runtime state; ORT runs one feature frame at a
time.

## Modules

| Module | Status | Description |
|--------|--------|-------------|
| [streaming/dpcrn_onnx](dpcrn_onnx.md) | active | DPCRN per-frame ONNX export and ORT streaming (the released voice-isolate deployment path; supports bounded look-ahead via future-buffering) |
| [streaming/dparn_onnx](dparn_onnx.md) | legacy | DPARN feature-frame ONNX export and ORT streaming |
