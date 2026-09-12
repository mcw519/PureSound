"""Unified PureSound model-zoo and ONNX inference API."""

from .runtime import InferenceError, InferenceResult, load_model
from .providers import (
    COREML_PROVIDER,
    CPU_PROVIDER,
    CUDA_PROVIDER,
    PROVIDER_ALIASES,
    PROVIDER_CHOICES,
    available_providers,
    normalize_provider,
    provider_is_available,
    resolve_providers,
)
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
    "COREML_PROVIDER",
    "CPU_PROVIDER",
    "CUDA_PROVIDER",
    "PROVIDER_ALIASES",
    "PROVIDER_CHOICES",
    "available_providers",
    "normalize_provider",
    "provider_is_available",
    "resolve_providers",
    "ValidationReport",
    "validate_catalog",
    "load_model",
]
