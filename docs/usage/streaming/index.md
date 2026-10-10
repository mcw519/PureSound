# Streaming deployment

Traditional Chinese: [index.zh-TW.md](index.zh-TW.md)

`puresound.streaming` exports a checkpoint as a per-frame ONNX graph plus a JSON
manifest, and runs it in real time. Python (or the portable SDK) owns audio
buffering, STFT/iSTFT, the state tensors and the post-graph stages the manifest
records; ONNX Runtime runs one feature frame at a time.

| Page | Status | Covers |
|--------|--------|-------------|
| [DPCRN streaming ONNX](dpcrn_onnx.md) | active | export, verify and run; look-ahead buffering; the dry blend and onset guard the runtime applies; auxiliary-head outputs. The path every released checkpoint deploys through |
| [DPARN streaming ONNX](dparn_onnx.md) | legacy | feature-frame export and runtime for a fixed-front-end DPARN |
| [Portable SDK](../../../sdk/python/README.md) | active | `puresound_streaming`: the same runtime with only NumPy and ONNX Runtime, for deployments without the training stack |

The released exports sit beside their checkpoints, under
`egs/voice_isolate/pretrained_ckpt/streaming/` and
`egs/noise_suppression/pretrained_ckpt/streaming/`; the model zoo
(`model_zoo/catalog.yaml`, `puresound infer`, the [web UI](../web.md)) runs them.
