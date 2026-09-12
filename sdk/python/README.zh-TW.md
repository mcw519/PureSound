# PureSound Streaming SDK

English version: [`README.md`](README.md)

可攜式（portable）Python runtime，用來執行已匯出的 PureSound streaming ONNX 模型。

這個套件的設計目的是可以直接複製或安裝進另一個專案，不需要安裝完整的 PureSound
訓練 repo。它只需要：

- `numpy`
- `onnxruntime`
- 一個已匯出的 `model.onnx`
- 對應的 `model.json` manifest

ONNX Runtime 請只選一個 extra：

```bash
python -m pip install "sdk/python[cpu]"   # CPU 或 macOS CoreML
python -m pip install "sdk/python[cuda]"  # NVIDIA CUDA 12.x
```

Runtime 接受 `auto`、`cpu`、`cuda`、`coreml` 與 `mps`。`auto` 依序偏好 CUDA、
CoreML、CPU；因為 ONNX Runtime 沒有原生 MPS execution provider，`mps` 是 CoreML
的 alias。可用 `onnxruntime.get_available_providers()` 確認目前實際 provider。

## 從這個資料夾安裝

```bash
python -m pip install "sdk/python[cpu]"
```

或直接把 `puresound_streaming/` 複製進你的專案。

## 使用方式

```python
import numpy as np

from puresound_streaming import PureSoundStreamingRuntime

runtime = PureSoundStreamingRuntime("model.onnx", "model.json", provider="auto")

audio = np.zeros(16000, dtype=np.float32)
enhanced = runtime.process_samples(audio)
enhanced = np.concatenate([enhanced, runtime.flush()])
```

對於使用 PCM int16 的即時（realtime）系統：

```python
out_i16 = runtime.process_int16(input_i16)
tail_i16 = runtime.flush_int16()
```

## 增強一個 WAV 檔案

`examples/enhance_wav_file.py` 是一個可直接執行的離線範例（讀取檔案、透過
runtime 進行 streaming 處理、寫出增強後的結果）。除了 SDK 本身的相依套件外，
它還需要 `soundfile`：

```bash
python examples/enhance_wav_file.py model.onnx input.wav output.wav
python examples/enhance_wav_file.py model.onnx input.wav output.wav \
    --manifest model.json --provider cuda
```

輸入必須是 16 kHz 單聲道；這個 SDK 不會做 resample。其輸出已驗證與訓練 repo
自身的 ORT 推論路徑（`puresound.streaming.StreamingDpcrnOrt`）在 DPCRN
匯出模型上逐位元組（byte-identical）一致。

## LiveKit Agent 的用法雛形

`examples/livekit_frame_processor.py` 是一個最簡雛形（sketch），展示 LiveKit
Agent 宿主專案（host project）可以如何在 runtime 外包一層 adapter：

```python
class PureSoundLiveKitFrameProcessor:
    def __init__(self, onnx_path: str, manifest_path: str | None = None, provider: str = "auto"):
        self.runtime = PureSoundStreamingRuntime(onnx_path, manifest_path, provider=provider)

    def process_pcm16(self, pcm: bytes | memoryview | np.ndarray) -> np.ndarray:
        if isinstance(pcm, np.ndarray):
            samples = pcm.astype(np.int16, copy=False)
        else:
            samples = np.frombuffer(pcm, dtype=np.int16)
        return self.runtime.process_int16(samples)

    def flush_pcm16(self) -> np.ndarray:
        return self.runtime.flush_int16()
```

這個 SDK 刻意不 import LiveKit。每個宿主專案要自行把這個 adapter 接進自己的
`rtc.FrameProcessor`（或等價機制），並把回傳的 int16 PCM 轉換回自己的
audio frame 型別。

## 支援的 Profile

目前這個 SDK 支援 `stft_frame_ort` 這個 processor profile，DPARN 與 DPCRN
的 voice-isolate 匯出模型都使用它（manifest 的 `model_type` 不同，但逐幀
（per-frame）STFT streaming 的介面規格是共用的）。未來的 PureSound 模型可以
透過匯出相容的 manifest 沿用同一個 SDK，或是新增另一個 processor profile。

對外公開的 runtime 刻意保持與模型無關（model-neutral）：

```python
from puresound_streaming import PureSoundStreamingRuntime
```

模型專屬的行為應該放在以 manifest `processor` 名稱註冊的 processor 類別裡，
而不是放進針對特定模型的 runtime 別名（alias）中。
