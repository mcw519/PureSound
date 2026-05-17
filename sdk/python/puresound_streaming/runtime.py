import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


class InferenceSessionLike(Protocol):
    def get_providers(self) -> list[str]:
        ...

    def run(self, output_names, inputs):
        ...


class StreamingProcessor(Protocol):
    def reset(self, batch_size: int = 1) -> None:
        ...

    def process_samples(self, samples: np.ndarray) -> np.ndarray:
        ...

    def flush(self) -> np.ndarray:
        ...

    def run_frame(self, frame: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class StreamingRuntimeConfig:
    model_type: str
    processor: str
    sample_rate: int
    fft_length: int
    win_length: int
    hop_length: int
    freq_bins: int
    state_input_names: list[str]
    state_output_names: list[str]
    output_names: list[str]
    state_shapes: dict[str, list[int]]

    @classmethod
    def from_manifest(cls, manifest: dict) -> "StreamingRuntimeConfig":
        return cls(
            model_type=str(manifest.get("model_type", "unknown")),
            processor=str(manifest.get("processor", "stft_frame_ort")),
            sample_rate=int(manifest["sample_rate"]),
            fft_length=int(manifest["fft_length"]),
            win_length=int(manifest["win_length"]),
            hop_length=int(manifest["hop_length"]),
            freq_bins=int(manifest["freq_bins"]),
            state_input_names=list(manifest["state_input_names"]),
            state_output_names=list(manifest["state_output_names"]),
            output_names=list(manifest["output_names"]),
            state_shapes={
                name: [int(dim) for dim in shape]
                for name, shape in manifest["state_shapes"].items()
            },
        )


class StftFrameOrtProcessor:
    """STFT-frame ONNX processor used by current DPARN streaming exports."""

    def __init__(self, config: StreamingRuntimeConfig, session: InferenceSessionLike):
        self.config = config
        self.session = session
        self.sample_rate = config.sample_rate
        self.fft_length = config.fft_length
        self.win_length = config.win_length
        self.hop_length = config.hop_length
        self.freq_bins = config.freq_bins
        self.window = np.hanning(self.win_length + 1)[:-1].astype(np.float32)
        self.reset()

    def reset(self, batch_size: int = 1) -> None:
        if batch_size != 1:
            raise ValueError("waveform streaming currently supports batch_size=1")
        self.state = {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in self.config.state_shapes.items()
        }
        self.input_buffer = np.zeros(0, dtype=np.float32)
        self.ola = np.zeros(0, dtype=np.float32)
        self.ola_norm = np.zeros(0, dtype=np.float32)

    def run_frame(self, noisy_frame: np.ndarray) -> np.ndarray:
        noisy_frame = np.asarray(noisy_frame, dtype=np.float32)
        expected = (1, self.freq_bins, 2)
        if noisy_frame.shape != expected:
            raise ValueError(f"noisy_frame must have shape {expected}")
        ort_inputs = {"noisy_frame": noisy_frame}
        ort_inputs.update(self.state)
        outputs = self.session.run(self.config.output_names, ort_inputs)
        enhanced = outputs[0]
        for name, value in zip(self.config.state_input_names, outputs[1:]):
            self.state[name] = value
        return enhanced

    def process_samples(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.input_buffer = np.concatenate([self.input_buffer, samples])
        chunks = []
        while self.input_buffer.shape[0] >= self.win_length:
            frame = self.input_buffer[: self.win_length]
            self.input_buffer = self.input_buffer[self.hop_length :]
            chunks.append(self._add_ola_frame(self._process_frame(frame)))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        chunks = []
        while self.input_buffer.size > 0:
            frame = np.zeros(self.win_length, dtype=np.float32)
            n = min(self.input_buffer.size, self.win_length)
            frame[:n] = self.input_buffer[:n]
            self.input_buffer = self.input_buffer[
                min(self.hop_length, self.input_buffer.size) :
            ]
            chunks.append(self._add_ola_frame(self._process_frame(frame)))
        if self.ola.size:
            tail = self.ola / np.maximum(self.ola_norm, 1e-8)
            chunks.append(tail.astype(np.float32))
            self.ola = np.zeros(0, dtype=np.float32)
            self.ola_norm = np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(frame * self.window, n=self.fft_length).astype(np.complex64)
        noisy_frame = np.stack([spec.real, spec.imag], axis=-1)
        noisy_frame = noisy_frame.reshape(1, self.freq_bins, 2).astype(np.float32)
        enhanced = self.run_frame(noisy_frame)[0]
        enhanced_complex = enhanced[:, 0] + 1j * enhanced[:, 1]
        wav = np.fft.irfft(enhanced_complex, n=self.fft_length).astype(np.float32)
        return wav[: self.win_length] * self.window

    def _add_ola_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.ola.size < self.win_length:
            pad = self.win_length - self.ola.size
            self.ola = np.pad(self.ola, (0, pad))
            self.ola_norm = np.pad(self.ola_norm, (0, pad))
        self.ola[: self.win_length] += frame
        self.ola_norm[: self.win_length] += self.window * self.window
        emit = self.ola[: self.hop_length].copy()
        norm = self.ola_norm[: self.hop_length].copy()
        emit = emit / np.maximum(norm, 1e-8)
        self.ola = self.ola[self.hop_length :]
        self.ola_norm = self.ola_norm[self.hop_length :]
        return emit.astype(np.float32)


PROCESSOR_REGISTRY = {
    "stft_frame_ort": StftFrameOrtProcessor,
}


class PureSoundStreamingRuntime:
    """Manifest-driven PureSound ONNX streaming runtime."""

    def __init__(
        self,
        onnx_path: str | Path,
        manifest_path: str | Path | None = None,
        provider: str = "auto",
        session: InferenceSessionLike | None = None,
    ):
        self.onnx_path = Path(onnx_path)
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else self.onnx_path.with_suffix(".json")
        )
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.config = StreamingRuntimeConfig.from_manifest(self.manifest)
        if session is None:
            import onnxruntime

            providers = self.resolve_providers(
                provider, onnxruntime.get_available_providers()
            )
            session = onnxruntime.InferenceSession(str(self.onnx_path), providers=providers)
        self.session = session
        self.providers = self.session.get_providers()
        self.processor = self.create_processor(self.config, self.session)
        self.sample_rate = self.processor.sample_rate
        self.fft_length = self.processor.fft_length
        self.win_length = self.processor.win_length
        self.hop_length = self.processor.hop_length
        self.freq_bins = self.processor.freq_bins

    @staticmethod
    def create_processor(
        config: StreamingRuntimeConfig,
        session: InferenceSessionLike,
    ) -> StreamingProcessor:
        processor_cls = PROCESSOR_REGISTRY.get(config.processor)
        if processor_cls is None:
            supported = ", ".join(sorted(PROCESSOR_REGISTRY))
            raise ValueError(
                f"Unsupported streaming processor: {config.processor}. "
                f"Supported processors: {supported}"
            )
        return processor_cls(config, session)

    @staticmethod
    def resolve_providers(provider: str, available: Sequence[str]) -> list[str]:
        provider = provider.lower()
        if provider == "cpu":
            return ["CPUExecutionProvider"]
        if provider == "cuda":
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ["CPUExecutionProvider"]
        if provider == "auto":
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ["CPUExecutionProvider"]
        raise ValueError("provider must be one of: auto, cpu, cuda")

    def reset(self, batch_size: int = 1) -> None:
        self.processor.reset(batch_size=batch_size)

    def run_frame(self, frame: np.ndarray) -> np.ndarray:
        return self.processor.run_frame(frame)

    def process_samples(self, samples: np.ndarray) -> np.ndarray:
        return self.processor.process_samples(samples)

    def flush(self) -> np.ndarray:
        return self.processor.flush()

    def process_int16(self, samples: np.ndarray) -> np.ndarray:
        float_samples = np.asarray(samples, dtype=np.float32).reshape(-1) / 32768.0
        enhanced = self.process_samples(float_samples)
        return self.float_to_int16(enhanced)

    def flush_int16(self) -> np.ndarray:
        return self.float_to_int16(self.flush())

    @staticmethod
    def float_to_int16(samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32)
        samples = np.clip(samples, -1.0, 1.0)
        return (samples * 32767.0).astype(np.int16)
