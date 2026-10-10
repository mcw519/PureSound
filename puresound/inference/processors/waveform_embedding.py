"""Waveform-to-embedding ONNX processor and speaker-verification adapter."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .base import (
    CancelCheck,
    ProgressCallback,
    check_cancelled,
    load_audio,
    report_progress,
)
from ..providers import normalize_provider, resolve_providers


class SpeakerVerificationRuntime:
    """Run two waveform embeddings and return cosine/threshold results."""

    task = "speaker_embedding"

    def __init__(
        self,
        model,
        artifact,
        *,
        provider: str = "auto",
        root: str | Path | None = None,
        session: Any | None = None,
    ):
        self.model = model
        self.artifact = artifact
        self.provider_requested = normalize_provider(provider)
        self.root = Path(root) if root is not None else Path.cwd()
        path = Path(artifact.path)
        self.onnx_path = path if path.is_absolute() else self.root / path
        if session is None:
            import onnxruntime

            providers = resolve_providers(
                provider, onnxruntime.get_available_providers()
            )
            session = onnxruntime.InferenceSession(
                str(self.onnx_path), providers=providers
            )
        self.session = session
        self.providers = list(self.session.get_providers())
        self.input_name = (artifact.input_names[0] if artifact.input_names else self.session.get_inputs()[0].name)
        self.output_name = (artifact.output_names[0] if artifact.output_names else self.session.get_outputs()[0].name)
        self.sample_rate = int(model.audio.sample_rate if model.audio else 16000)
        self.default_target_dbfs = (
            float(model.audio.target_dbfs)
            if model.audio and model.audio.target_dbfs is not None
            else None
        )
        threshold_spec = model.parameters.get("threshold")
        self.default_threshold = float(threshold_spec.default) if threshold_spec else 0.8

    @classmethod
    def from_session(
        cls,
        session: Any,
        *,
        model_id: str = "speaker-verification-ps-spk-v1-1",
        zoo=None,
    ) -> "SpeakerVerificationRuntime":
        """Adapt an existing ORT session for legacy demo/notebook callers."""
        from puresound.inference.zoo import ModelZoo

        zoo = zoo or ModelZoo.default()
        model = zoo.get(model_id)
        artifact = zoo.resolve_artifact(model.id)
        return cls(
            model,
            artifact,
            provider="auto",
            root=zoo.root,
            session=session,
        )

    @property
    def model_id(self) -> str:
        return self.model.id

    def _parameters(self, parameters: Mapping[str, Any] | None) -> tuple[float, float | None]:
        parameters = dict(parameters or {})
        unknown = set(parameters) - {"threshold", "target_dbfs"}
        if unknown:
            raise ValueError("unsupported speaker-verification parameter(s): " + ", ".join(sorted(unknown)))
        threshold = float(parameters.get("threshold", self.default_threshold))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        target = parameters.get("target_dbfs", self.default_target_dbfs)
        target_dbfs = None if target is None else float(target)
        return threshold, target_dbfs

    def _embedding(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            raise ValueError("audio input is empty")
        waveform = samples.reshape(1, -1).astype(np.float32, copy=False)
        result = self.session.run([self.output_name], {self.input_name: waveform})[0]
        embedding = np.asarray(result, dtype=np.float32).reshape(-1)
        if embedding.size == 0 or not np.all(np.isfinite(embedding)):
            raise ValueError("ONNX embedding output is empty or non-finite")
        return embedding

    def infer(
        self,
        inputs: Mapping[str, Any],
        parameters: Mapping[str, Any] | None = None,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ):
        from puresound.inference.runtime import InferenceResult

        expected = {"enrollment", "test"}
        if set(inputs) != expected:
            missing = expected - set(inputs)
            extra = set(inputs) - expected
            details = []
            if missing:
                details.append("missing input(s): " + ", ".join(sorted(missing)))
            if extra:
                details.append("unknown input(s): " + ", ".join(sorted(extra)))
            raise ValueError("speaker verification expects named inputs 'enrollment' and 'test' (" + "; ".join(details) + ")")
        check_cancelled(cancel_check)
        report_progress(progress_callback, 0.0, "loading_audio")
        threshold, target_dbfs = self._parameters(parameters)
        enrollment, sample_rate = load_audio(
            inputs["enrollment"], sample_rate=self.sample_rate, target_dbfs=target_dbfs
        )
        test, _ = load_audio(inputs["test"], sample_rate=self.sample_rate, target_dbfs=target_dbfs)
        check_cancelled(cancel_check)
        report_progress(progress_callback, 0.1, "embedding_enrollment")
        started = time.perf_counter()
        enrollment_embedding = self._embedding(enrollment)
        check_cancelled(cancel_check)
        report_progress(progress_callback, 0.5, "embedding_test")
        test_embedding = self._embedding(test)
        check_cancelled(cancel_check)
        report_progress(progress_callback, 0.95, "scoring")
        denominator = float(np.linalg.norm(enrollment_embedding) * np.linalg.norm(test_embedding))
        similarity = float(np.dot(enrollment_embedding, test_embedding) / denominator) if denominator > 1e-12 else 0.0
        elapsed = time.perf_counter() - started
        duration = (enrollment.size + test.size) / float(sample_rate)
        outputs = {
            "enrollment_embedding": enrollment_embedding,
            "test_embedding": test_embedding,
        }
        scores = {
            "cosine_similarity": similarity,
            "threshold": threshold,
            "verdict": bool(similarity > threshold),
        }
        metadata = {
            "requested_provider": self.provider_requested,
            "selected_provider": self.providers[0] if self.providers else "",
            "provider": ",".join(self.providers),
            "providers": list(self.providers),
            "sample_rate": sample_rate,
            "embedding_dim": int(enrollment_embedding.size),
            "elapsed_seconds": elapsed,
            "duration_seconds": duration,
            "rtf": elapsed / duration if duration > 0 else None,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "artifact_variant": self.artifact.variant,
        }
        result = InferenceResult(
            model_id=self.model.id,
            task=self.model.task,
            outputs=outputs,
            scores=scores,
            provider=metadata["provider"],
            elapsed_seconds=elapsed,
            rtf=metadata["rtf"],
            sample_rate=sample_rate,
            metadata=metadata,
        )
        report_progress(progress_callback, 1.0, "complete")
        return result


WaveformEmbeddingOrtProcessor = SpeakerVerificationRuntime

__all__ = ["SpeakerVerificationRuntime", "WaveformEmbeddingOrtProcessor", "resolve_providers"]
