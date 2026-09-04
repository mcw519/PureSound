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


def test_auto_prefers_cuda_then_coreml_then_cpu():
    assert resolve_providers("auto", [CPU_PROVIDER, COREML_PROVIDER, CUDA_PROVIDER]) == [
        CUDA_PROVIDER,
        CPU_PROVIDER,
    ]
    assert resolve_providers("auto", [CPU_PROVIDER, COREML_PROVIDER]) == [
        COREML_PROVIDER,
        CPU_PROVIDER,
    ]
    assert resolve_providers("auto", [CPU_PROVIDER]) == [CPU_PROVIDER]


@pytest.mark.parametrize("choice", ["cuda", "coreml", "mps"])
def test_explicit_accelerator_falls_back_to_cpu(choice):
    assert resolve_providers(choice, [CPU_PROVIDER]) == [CPU_PROVIDER]


def test_mps_is_a_coreml_alias():
    assert normalize_provider("MPS") == "coreml"
    assert provider_is_available("mps", [COREML_PROVIDER]) is True
    assert resolve_providers("mps", [COREML_PROVIDER, CPU_PROVIDER]) == [
        COREML_PROVIDER,
        CPU_PROVIDER,
    ]


def test_invalid_provider_mentions_all_public_choices():
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
