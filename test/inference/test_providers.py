from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from puresound.inference import (
    COREML_PROVIDER,
    CPU_PROVIDER,
    CUDA_PROVIDER,
    normalize_provider,
    provider_is_available,
    resolve_providers,
)


@pytest.mark.parametrize(
    "choice, available, expected",
    [
        ("auto", [CPU_PROVIDER, COREML_PROVIDER, CUDA_PROVIDER], [CUDA_PROVIDER, CPU_PROVIDER]),
        ("auto", [CPU_PROVIDER, COREML_PROVIDER], [COREML_PROVIDER, CPU_PROVIDER]),
        ("auto", [CPU_PROVIDER], [CPU_PROVIDER]),
        # An explicit accelerator that is not there falls back to the CPU.
        ("cuda", [CPU_PROVIDER], [CPU_PROVIDER]),
        ("coreml", [CPU_PROVIDER], [CPU_PROVIDER]),
        ("mps", [CPU_PROVIDER], [CPU_PROVIDER]),
        ("mps", [COREML_PROVIDER, CPU_PROVIDER], [COREML_PROVIDER, CPU_PROVIDER]),
    ],
)
def test_provider_resolution_prefers_cuda_then_coreml_then_cpu(choice, available, expected):
    assert resolve_providers(choice, available) == expected


def test_mps_is_a_coreml_alias_and_an_unknown_provider_lists_the_choices():
    assert normalize_provider("MPS") == "coreml"
    assert provider_is_available("mps", [COREML_PROVIDER]) is True
    with pytest.raises(ValueError, match="auto, cpu, cuda, coreml, mps"):
        normalize_provider("tpu")


def test_project_runtime_profiles_are_mutually_exclusive():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["requires-python"] == ">=3.12"
    base = "\n".join(project["project"]["dependencies"])
    assert "onnxruntime" not in base
    assert "faster-whisper" not in base
    cpu = "\n".join(project["project"]["optional-dependencies"]["cpu"])
    cuda = "\n".join(project["project"]["optional-dependencies"]["cuda"])
    assert "onnxruntime-gpu" not in cpu
    assert "onnxruntime-gpu==1.24.1" in cuda
    assert [
        {"extra": "cpu"},
        {"extra": "cuda"},
    ] in project["tool"]["uv"]["conflicts"]
