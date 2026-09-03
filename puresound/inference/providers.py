"""Common ONNX Runtime provider resolution for every processor."""

from __future__ import annotations

from collections.abc import Sequence


def resolve_providers(provider: str, available: Sequence[str]) -> list[str]:
    """Resolve ``auto|cpu|cuda`` with deterministic CPU fallback."""
    choice = str(provider).lower()
    available = list(available)
    if choice == "cpu":
        return ["CPUExecutionProvider"]
    if choice in {"auto", "cuda"}:
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    raise ValueError("provider must be one of: auto, cpu, cuda")


__all__ = ["resolve_providers"]
