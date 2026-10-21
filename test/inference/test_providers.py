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


@pytest.mark.parametrize(
    "choice, available, expected",
    [
        ("auto", [CPU_PROVIDER, COREML_PROVIDER], [CPU_PROVIDER]),
        ("auto", [CPU_PROVIDER, COREML_PROVIDER, CUDA_PROVIDER], [CUDA_PROVIDER, CPU_PROVIDER]),
        # Only `auto` changes: an explicit CoreML request is still honoured.
        ("coreml", [CPU_PROVIDER, COREML_PROVIDER], [COREML_PROVIDER, CPU_PROVIDER]),
        ("mps", [CPU_PROVIDER, COREML_PROVIDER], [COREML_PROVIDER, CPU_PROVIDER]),
    ],
)
def test_frame_graphs_keep_auto_off_coreml(choice, available, expected):
    assert resolve_providers(choice, available, frame_graph=True) == expected


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


@pytest.mark.parametrize(
    "machine, release, expected",
    [
        # macOS 14+ on Apple Silicon gets the SME-capable 1.24+ builds.
        ("arm64", "23.6.0", ">=1.24,<2"),
        ("arm64", "25.5.0", ">=1.24,<2"),
        # No 1.24+ wheel exists for macOS 13 or for Intel Macs.
        ("arm64", "22.6.0", "==1.23.2"),
        ("x86_64", "23.6.0", "==1.23.2"),
    ],
)
def test_every_cpu_profile_pins_one_onnxruntime_per_mac(machine, release, expected):
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet

    environment = {"sys_platform": "darwin", "platform_machine": machine,
                   "platform_release": release, "extra": "cpu"}
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    sdk = tomllib.loads(Path("sdk/python/pyproject.toml").read_text(encoding="utf-8"))
    pip = [line for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
           if line.startswith("onnxruntime")]
    for lines in (project["project"]["optional-dependencies"]["cpu"],
                  sdk["project"]["optional-dependencies"]["cpu"], pip):
        selected = [requirement for requirement in map(Requirement, lines)
                    if requirement.name == "onnxruntime"
                    and requirement.marker.evaluate(environment)]
        assert [requirement.specifier for requirement in selected] == [SpecifierSet(expected)]
