"""Public facade for all repository ONNX inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .processors import PROCESSOR_REGISTRY
from .zoo import ModelZoo


class InferenceError(RuntimeError):
    """Raised when a model cannot be resolved or a processor is unavailable."""


@dataclass(frozen=True)
class InferenceResult:
    """Common result envelope returned by every processor."""

    model_id: str
    task: str
    outputs: Mapping[str, Any] = field(default_factory=dict)
    scores: Mapping[str, Any] = field(default_factory=dict)
    provider: str = ""
    elapsed_seconds: float = 0.0
    rtf: float | None = None
    sample_rate: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_seconds * 1000.0

    @property
    def named_outputs(self) -> Mapping[str, Any]:
        """Alias used by API clients that call the output map named outputs."""
        return self.outputs

    @property
    def latency_ms(self) -> float:
        return float(self.metadata.get("latency_ms", self.elapsed_ms))

    @property
    def latency_metadata(self) -> Mapping[str, Any]:
        return self.metadata

    def as_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly summary for CLI/API responses."""

        def convert(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                if include_arrays:
                    return value.tolist()
                return {"shape": list(value.shape), "dtype": str(value.dtype)}
            if isinstance(value, (np.floating, np.integer)):
                return value.item()
            if isinstance(value, Mapping):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        return {
            "model_id": self.model_id,
            "task": self.task,
            "outputs": convert(self.outputs),
            "scores": convert(self.scores),
            "provider": self.provider,
            "elapsed_seconds": self.elapsed_seconds,
            "rtf": self.rtf,
            "sample_rate": self.sample_rate,
            "metadata": convert(self.metadata),
        }


def _resolve_model_ref(model_ref: str | Path, zoo: ModelZoo):
    reference = str(model_ref)
    try:
        return zoo.get(reference)
    except KeyError:
        found = zoo.find_by_artifact(model_ref)
        if found is not None:
            return found[0]
    raise InferenceError(f"unknown model id or catalog artifact: {model_ref}")


def load_model(
    model_id: str | Path,
    *,
    provider: str = "auto",
    zoo: ModelZoo | None = None,
    variant: str | None = None,
):
    """Load a catalog model through its declared processor.

    ``model_id`` normally is a logical id.  A catalog artifact path is also
    accepted as a compatibility bridge for existing Gradio callers.
    """

    zoo = zoo or ModelZoo.default()
    model = _resolve_model_ref(model_id, zoo)
    try:
        artifact = zoo.resolve_artifact(model.id, variant=variant)
    except Exception as exc:
        raise InferenceError(str(exc)) from exc
    processor = PROCESSOR_REGISTRY.get(artifact.processor)
    if processor is None:
        raise InferenceError(
            f"model {model.id!r} declares unsupported processor {artifact.processor!r}"
        )
    return processor(model, artifact, provider=provider, root=zoo.root)


__all__ = ["InferenceError", "InferenceResult", "load_model"]
