import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


_PROVIDER_ALIASES = {
    "auto": "auto",
    "cpu": "cpu",
    "cuda": "cuda",
    "coreml": "coreml",
    "mps": "coreml",
}
_CPU_PROVIDER = "CPUExecutionProvider"
_CUDA_PROVIDER = "CUDAExecutionProvider"
_COREML_PROVIDER = "CoreMLExecutionProvider"


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
    #: Frames the graph's output lags its input by. Non-zero for a look-ahead
    #: export (every current DPCRN export is 3), and the dry blend has to
    #: compensate for it or it mixes in the wrong slice of the input.
    streaming_delay_frames: int = 0
    #: Post-graph over-suppression relief, from the manifest. The graph does not
    #: contain it, so the runtime applies it -- otherwise the deployed system is
    #: not the one the benchmarks measured.
    dry_blend: float = 1.0
    spec_floor: float = 0.0

    @classmethod
    def from_manifest(cls, manifest: dict) -> "StreamingRuntimeConfig":
        # `recommended_inference` is the key the exports have always carried;
        # absent means no relief, which is also what `Postprocessor()` defaults
        # to. Spelled here rather than imported: the SDK must not import
        # puresound, so the name is duplicated on purpose and
        # `test_sdk_postprocess.py` pins the two copies together.
        postprocess = manifest.get("recommended_inference") or {}
        spec_floor = float(postprocess.get("spec_floor", 0.0))
        if spec_floor > 0.0:
            # It needs the mixture spectrum at the same frame and a magnitude to
            # scale, which this processor could supply -- but no shipped export
            # sets it, so refusing beats a silent no-op.
            raise ValueError(
                f"manifest requests spec_floor={spec_floor}, which this runtime "
                "does not implement. Re-export with spec_floor=0.0 or add it here."
            )
        dry_blend = float(postprocess.get("dry_blend", 1.0))
        if not 0.0 < dry_blend <= 1.0:
            raise ValueError(f"manifest dry_blend must be in (0, 1], got {dry_blend}")
        return cls(
            streaming_delay_frames=int(manifest.get("streaming_delay_frames", 0)),
            dry_blend=dry_blend,
            spec_floor=spec_floor,
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
        self.dry_blend = config.dry_blend
        # How far back in the input the sample aligned with the next emitted
        # output sample sits. The overlap-add itself is index-aligned -- an
        # identity graph reconstructs the input at the same indices -- so this is
        # purely the graph's own look-ahead latency.
        self.dry_delay = config.streaming_delay_frames * self.hop_length
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
        # Raw input kept for the dry blend, with the absolute index its first
        # sample has. Trimmed as it is consumed so a long stream does not grow it.
        self.dry_history = np.zeros(0, dtype=np.float32)
        self.dry_history_start = 0
        self.emitted = 0

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
        self._remember_dry(samples)
        self.input_buffer = np.concatenate([self.input_buffer, samples])
        chunks = []
        while self.input_buffer.shape[0] >= self.win_length:
            frame = self.input_buffer[: self.win_length]
            self.input_buffer = self.input_buffer[self.hop_length :]
            chunks.append(self._blend_dry(self._add_ola_frame(self._process_frame(frame))))
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
            chunks.append(self._blend_dry(self._add_ola_frame(self._process_frame(frame))))
        if self.ola.size:
            tail = self.ola / np.maximum(self.ola_norm, 1e-8)
            chunks.append(self._blend_dry(tail.astype(np.float32)))
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

    def _remember_dry(self, samples: np.ndarray) -> None:
        if self.dry_blend >= 1.0:
            return
        self.dry_history = np.concatenate([self.dry_history, samples])

    def _blend_dry(self, enhanced: np.ndarray) -> np.ndarray:
        """Mix the untouched input back in, aligned to what the graph enhanced.

        ``dry_blend * enhanced + (1 - dry_blend) * input``, the same expression
        the offline module applies -- but the reference has to come from
        ``streaming_delay_frames`` back, because that is the input the graph's
        output at this index actually carries. Blending index-for-index would mix
        in a slice of the mixture 30 ms away from the speech it is relieving.

        Output samples with no corresponding input yet -- the first
        ``streaming_delay_frames`` worth, which is warm-up -- pass through
        unblended: there is nothing to blend them with.
        """
        if self.dry_blend >= 1.0 or enhanced.size == 0:
            return enhanced
        start = self.emitted - self.dry_delay
        self.emitted += enhanced.size
        out = enhanced.astype(np.float32, copy=True)

        # Absolute-index overlap between what we are emitting and what we still
        # hold. Sliced rather than looped: this runs per hop on the audio thread.
        history_end = self.dry_history_start + self.dry_history.size
        lo = max(start, self.dry_history_start)
        hi = min(start + enhanced.size, history_end)
        if hi > lo:
            span = slice(lo - start, hi - start)
            reference = self.dry_history[lo - self.dry_history_start : hi - self.dry_history_start]
            out[span] = (
                self.dry_blend * enhanced[span] + (1.0 - self.dry_blend) * reference
            )
        np.clip(out, -1.0, 1.0, out=out)
        # Everything before the next emit's reference is dead weight.
        keep_from = max(0, self.emitted - self.dry_delay - self.dry_history_start)
        if keep_from > 0:
            self.dry_history = self.dry_history[keep_from:]
            self.dry_history_start += keep_from
        return out

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
        choice = _PROVIDER_ALIASES.get(str(provider).strip().lower())
        if choice is None:
            choices = ", ".join(_PROVIDER_ALIASES)
            raise ValueError(f"provider must be one of: {choices}")
        available = list(available)
        if choice == "cpu":
            return [_CPU_PROVIDER]
        if choice == "cuda":
            return ([_CUDA_PROVIDER, _CPU_PROVIDER]
                    if _CUDA_PROVIDER in available else [_CPU_PROVIDER])
        if choice == "coreml":
            return ([_COREML_PROVIDER, _CPU_PROVIDER]
                    if _COREML_PROVIDER in available else [_CPU_PROVIDER])
        for candidate in (_CUDA_PROVIDER, _COREML_PROVIDER):
            if candidate in available:
                return [candidate, _CPU_PROVIDER]
        return [_CPU_PROVIDER]

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
