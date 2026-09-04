"""Common ONNX Runtime provider resolution for every processor.

The public names intentionally describe the user choice rather than the
ONNX Runtime implementation detail. ``mps`` is accepted as a compatibility
alias for Apple's ``CoreMLExecutionProvider``: ONNX Runtime has no native MPS
execution provider, while CoreML can use the Mac GPU/Neural Engine when the
installed runtime exposes it.
"""

from __future__ import annotations

from collections.abc import Sequence

CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
COREML_PROVIDER = "CoreMLExecutionProvider"

PROVIDER_ALIASES = {
    "auto": "auto",
    "cpu": "cpu",
    "cuda": "cuda",
    "coreml": "coreml",
    # There is no MPSExecutionProvider in ONNX Runtime. Keep the alias so a
    # caller can express the platform intent without an invalid-provider error;
    # the actual session still reports CoreML.
    "mps": "coreml",
}
PROVIDER_CHOICES = tuple(PROVIDER_ALIASES)


def normalize_provider(provider: str) -> str:
    """Return a canonical public provider name and validate it."""

    choice = str(provider).strip().lower()
    try:
        return PROVIDER_ALIASES[choice]
    except KeyError as exc:
        choices = ", ".join(PROVIDER_CHOICES)
        raise ValueError(f"provider must be one of: {choices}") from exc


def available_providers() -> list[str]:
    """Return providers exposed by the active ONNX Runtime installation.

    Importing ONNX Runtime is deliberately lazy so catalog/CLI help can still
    work in an environment where the optional runtime wheel is not installed.
    """

    try:
        import onnxruntime
    except Exception:
        return []
    try:
        return list(onnxruntime.get_available_providers())
    except Exception:
        return []


def provider_is_available(provider: str, available: Sequence[str]) -> bool:
    """Whether a public provider maps to an available ORT provider."""

    choice = normalize_provider(provider)
    if choice == "auto":
        return True
    target = {
        "cpu": CPU_PROVIDER,
        "cuda": CUDA_PROVIDER,
        "coreml": COREML_PROVIDER,
    }[choice]
    return target in set(available)


def resolve_providers(provider: str, available: Sequence[str]) -> list[str]:
    """Resolve a public provider choice with deterministic CPU fallback.

    ``auto`` prefers CUDA, then CoreML, then CPU. Explicit ``cuda`` and
    ``coreml``/``mps`` requests retain the historical CPU fallback contract so
    existing callers remain usable on machines without the optional backend.
    The selected session's provider list is returned by each runtime, allowing
    callers to report when the fallback happened.
    """

    choice = normalize_provider(provider)
    available = list(available)
    if choice == "cpu":
        return [CPU_PROVIDER]
    if choice == "cuda":
        if CUDA_PROVIDER in available:
            return [CUDA_PROVIDER, CPU_PROVIDER]
        return [CPU_PROVIDER]
    if choice == "coreml":
        if COREML_PROVIDER in available:
            return [COREML_PROVIDER, CPU_PROVIDER]
        return [CPU_PROVIDER]
    for candidate in (CUDA_PROVIDER, COREML_PROVIDER):
        if candidate in available:
            return [candidate, CPU_PROVIDER]
    return [CPU_PROVIDER]


__all__ = [
    "COREML_PROVIDER",
    "CPU_PROVIDER",
    "CUDA_PROVIDER",
    "PROVIDER_ALIASES",
    "PROVIDER_CHOICES",
    "available_providers",
    "normalize_provider",
    "provider_is_available",
    "resolve_providers",
]
