"""Minimal sketch for using the portable SDK in a LiveKit Agent project.

This file is intentionally not imported by tests because LiveKit is an optional
dependency for host applications.
"""

import numpy as np

from puresound_streaming import PureSoundStreamingRuntime


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
