# PureSound Streaming SDK

Portable Python runtime for exported PureSound streaming ONNX models.

This package is meant to be copied or installed into another project without
installing the full PureSound training repository. It only needs:

- `numpy`
- `onnxruntime`
- an exported `model.onnx`
- the matching `model.json` manifest

## Install From This Folder

```bash
python -m pip install sdk/python
```

Or copy `puresound_streaming/` into your project.

## Usage

```python
import numpy as np

from puresound_streaming import PureSoundStreamingRuntime

runtime = PureSoundStreamingRuntime("model.onnx", "model.json", provider="auto")

audio = np.zeros(16000, dtype=np.float32)
enhanced = runtime.process_samples(audio)
enhanced = np.concatenate([enhanced, runtime.flush()])
```

For realtime systems that use PCM int16:

```python
out_i16 = runtime.process_int16(input_i16)
tail_i16 = runtime.flush_int16()
```

## LiveKit Agent Shape

Use this SDK inside a small LiveKit `FrameProcessor` adapter:

```python
class PureSoundFrameProcessor(rtc.FrameProcessor[rtc.AudioFrame]):
    def __init__(self, model_path: str, manifest_path: str):
        super().__init__()
        self.runtime = PureSoundStreamingRuntime(model_path, manifest_path)

    def _process(self, frame):
        pcm = np.frombuffer(frame.data, dtype=np.int16)
        enhanced = self.runtime.process_int16(pcm)
        # Build and return a new rtc.AudioFrame with the enhanced PCM.
```

The SDK intentionally does not import LiveKit. Each host project can adapt the
returned int16 or float32 samples to its own audio frame type.

## Supported Profiles

The current SDK supports the `stft_frame_ort` processor profile used by the
DPARN voice-isolate export. Future PureSound models can use the same SDK by
exporting a compatible manifest, or by adding another processor profile.

The public runtime intentionally stays model-neutral:

```python
from puresound_streaming import PureSoundStreamingRuntime
```

Model-specific behavior belongs in processor classes registered by manifest
`processor` name, not in model-specific runtime aliases.
