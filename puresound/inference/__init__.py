"""Unified PureSound model-zoo and ONNX inference API."""

from .runtime import InferenceError, InferenceResult, load_model
from .providers import resolve_providers
from .processors import (
    CancelCheck,
    InferenceCancelled,
    PROCESSOR_REGISTRY,
    ProcessorProtocol,
    ProgressCallback,
    SpeakerVerificationRuntime,
    StftFrameOrtProcessor,
    WaveformEmbeddingOrtProcessor,
)
from .schema import (
    ArtifactSpec,
    AudioSpec,
    Catalog,
    LogicalModelSpec,
    ModelSpec,
    ModelZooCatalog,
    ParameterSpec,
)
from .zoo import ModelZoo, ModelZooError, ModelZooValidationError, ValidationReport, validate_catalog

__all__ = [
    "ArtifactSpec",
    "AudioSpec",
    "Catalog",
    "CancelCheck",
    "InferenceCancelled",
    "InferenceError",
    "InferenceResult",
    "ModelSpec",
    "ModelZooCatalog",
    "LogicalModelSpec",
    "ModelZoo",
    "ModelZooError",
    "ModelZooValidationError",
    "ParameterSpec",
    "PROCESSOR_REGISTRY",
    "ProcessorProtocol",
    "ProgressCallback",
    "SpeakerVerificationRuntime",
    "StftFrameOrtProcessor",
    "WaveformEmbeddingOrtProcessor",
    "resolve_providers",
    "ValidationReport",
    "validate_catalog",
    "load_model",
]
