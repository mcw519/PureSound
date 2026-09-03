"""Manifest-driven streaming STFT ONNX processor."""

from __future__ import annotations

import time
import json
import threading
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .base import (
    CancelCheck,
    InferenceCancelled,
    ProgressCallback,
    check_cancelled,
    load_audio,
    report_progress,
)


class VoiceIsolationRuntime:
    """Runtime for ``stft_frame_ort`` voice-isolation artifacts.

    ``StreamingOrt`` owns the frame graph, overlap-add buffers and sidecar
    post-processing.  This adapter only translates named facade inputs into
    that runtime and packages a common result object.  A fresh streaming
    object is created for every offline ``infer`` call; the optional stream
    methods keep state local to this runtime instance for realtime clients.
    """

    task = "voice_isolation"

    def __init__(self, model, artifact, *, provider: str = "auto", root: str | Path | None = None):
        self.model = model
        self.artifact = artifact
        self.provider_requested = str(provider).lower()
        if self.provider_requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("provider must be one of: auto, cpu, cuda")
        self.root = Path(root) if root is not None else Path.cwd()
        self.onnx_path = Path(self._root_path(artifact.path))
        self.manifest_path = (
            Path(self._root_path(artifact.manifest)) if artifact.manifest else None
        )
        self._manifest = (
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest_path and self.manifest_path.is_file()
            else {}
        )
        self._stream = None
        self._session = None
        self._session_lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self.model.id

    def _root_path(self, path: str | None) -> Path:
        if path is None:
            raise ValueError("stft_frame_ort requires an ONNX sidecar manifest")
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate

    @property
    def sample_rate(self) -> int:
        if self._manifest:
            return int(self._manifest["sample_rate"])
        return int(self.model.audio.sample_rate)

    @property
    def capabilities(self) -> Mapping[str, Any]:
        return self.model.capabilities

    def _parameter_overrides(self, parameters: Mapping[str, Any] | None) -> tuple[dict[str, Any], bool]:
        parameters = dict(parameters or {})
        allowed = {
            "dry_blend",
            "spec_floor",
            "collect_extras",
            "return_heads",
            "include_heads",
            "heads",
            "vad_heads",
        }
        unknown = set(parameters) - allowed
        if unknown:
            raise ValueError("unsupported voice-isolation parameter(s): " + ", ".join(sorted(unknown)))
        collect_extras = bool(parameters.pop("collect_extras", False))
        collect_extras = bool(
            collect_extras
            or parameters.pop("return_heads", False)
            or parameters.pop("include_heads", False)
            or parameters.pop("heads", False)
            or parameters.pop("vad_heads", False)
        )
        if "dry_blend" in parameters:
            try:
                dry_blend = float(parameters["dry_blend"])
            except (TypeError, ValueError) as exc:
                raise ValueError("dry_blend must be a number in (0, 1]") from exc
            if not 0.0 < dry_blend <= 1.0:
                raise ValueError("dry_blend must be in (0, 1]")
            parameters["dry_blend"] = dry_blend
        if "spec_floor" in parameters:
            try:
                spec_floor = float(parameters["spec_floor"])
            except (TypeError, ValueError) as exc:
                raise ValueError("spec_floor must be a non-negative number") from exc
            if spec_floor < 0.0:
                raise ValueError("spec_floor must be non-negative")
            parameters["spec_floor"] = spec_floor
        return parameters, collect_extras

    def _make_stream(self, parameters: Mapping[str, Any] | None = None):
        from puresound.streaming import StreamingOrt

        overrides, collect_extras = self._parameter_overrides(parameters)
        # The ORT session is stateless and thread-safe; frame state, overlap-add
        # buffers, dry history and auxiliary history live on each StreamingOrt
        # wrapper.  Reusing only the session avoids model reloads without letting
        # one inference reset another.
        with self._session_lock:
            stream = StreamingOrt(
                onnx_path=self.onnx_path,
                manifest_path=self.manifest_path,
                provider=self.provider_requested,
                collect_extras=collect_extras,
                postprocess_overrides=overrides,
                session=self._session,
            )
            if self._session is None:
                self._session = stream.session
        return stream

    def infer(
        self,
        inputs: Mapping[str, Any],
        parameters: Mapping[str, Any] | None = None,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ):
        from puresound.inference.runtime import InferenceResult

        if set(inputs) != {"audio"}:
            missing = {"audio"} - set(inputs)
            extra = set(inputs) - {"audio"}
            details = []
            if missing:
                details.append("missing input(s): " + ", ".join(sorted(missing)))
            if extra:
                details.append("unknown input(s): " + ", ".join(sorted(extra)))
            raise ValueError("voice isolation expects named input 'audio' (" + "; ".join(details) + ")")
        check_cancelled(cancel_check)
        report_progress(progress_callback, 0.0, "loading_audio")
        overrides, collect_extras = self._parameter_overrides(parameters)
        samples, sample_rate = load_audio(
            inputs["audio"], sample_rate=self.sample_rate, target_dbfs=None
        )
        if samples.size == 0:
            raise ValueError("audio input is empty")
        check_cancelled(cancel_check)
        report_progress(progress_callback, 0.02, "loading_model")
        started = time.perf_counter()
        stream = self._make_stream({**overrides, "collect_extras": collect_extras})
        total_frames = max(1, int(np.ceil(samples.size / stream.hop_length)))
        completed_frames = 0

        def frame_complete() -> None:
            nonlocal completed_frames
            completed_frames += 1
            report_progress(
                progress_callback,
                min(0.99, completed_frames / total_frames),
                "processing_frames",
            )

        try:
            enhanced_parts = [
                stream.process_samples(
                    samples,
                    frame_callback=frame_complete,
                    cancel_check=cancel_check,
                ),
                stream.flush(
                    frame_callback=frame_complete,
                    cancel_check=cancel_check,
                ),
            ]
        except InterruptedError as exc:
            raise InferenceCancelled("inference cancelled") from exc
        check_cancelled(cancel_check)
        enhanced = np.concatenate([part for part in enhanced_parts if part.size])
        elapsed = time.perf_counter() - started
        outputs: dict[str, np.ndarray] = {"audio": enhanced.astype(np.float32, copy=False)}
        if collect_extras:
            outputs.update(stream.drain_extras())
        duration = samples.size / float(sample_rate)
        metadata = {
            "provider": ",".join(stream.providers),
            "providers": list(stream.providers),
            "sample_rate": sample_rate,
            "input_samples": int(samples.size),
            "output_samples": int(enhanced.size),
            "duration_seconds": duration,
            "elapsed_seconds": elapsed,
            "rtf": elapsed / duration if duration > 0 else None,
            "streaming_delay_frames": int(stream.manifest.get("streaming_delay_frames", 0)),
            "hop_length": int(stream.hop_length),
            "latency_samples": int(stream.dry_delay),
            "latency_ms": 1000.0 * float(stream.dry_delay) / sample_rate,
            "artifact_variant": self.artifact.variant,
        }
        result = InferenceResult(
            model_id=self.model.id,
            task=self.model.task,
            outputs=outputs,
            scores={},
            provider=metadata["provider"],
            elapsed_seconds=elapsed,
            rtf=metadata["rtf"],
            sample_rate=sample_rate,
            metadata=metadata,
        )
        report_progress(progress_callback, 1.0, "complete")
        return result

    # Realtime compatibility -------------------------------------------------
    # A caller obtains one runtime per realtime session, so these mutable
    # buffers can never leak across concurrent sessions.
    def _ensure_stream(self, parameters: Mapping[str, Any] | None = None):
        if self._stream is None:
            self._stream = self._make_stream(parameters)
        return self._stream

    @property
    def providers(self) -> list[str]:
        stream = self._ensure_stream()
        return list(stream.providers)

    @property
    def manifest(self) -> dict[str, Any]:
        return dict(self._manifest)

    @property
    def extra_names(self) -> list[str]:
        return list(self._manifest.get("extra_output_names", []))

    @property
    def hop_length(self) -> int:
        return int(self._manifest.get("hop_length", 160))

    def reset(self) -> None:
        self._ensure_stream().reset()

    def process_samples(self, samples: np.ndarray) -> np.ndarray:
        return self._ensure_stream().process_samples(samples)

    def flush(self) -> np.ndarray:
        return self._ensure_stream().flush()

    def drain_extras(self) -> dict[str, np.ndarray]:
        return self._ensure_stream().drain_extras()


StftFrameOrtProcessor = VoiceIsolationRuntime

__all__ = ["StftFrameOrtProcessor", "VoiceIsolationRuntime"]
