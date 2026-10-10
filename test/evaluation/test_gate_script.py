"""The shell driver must preserve stage and collection exit status through tee."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "egs/noise_suppression/run_full_gate.sh"


def _run_gate(tmp_path: Path, *args: str, fail_module: str = "",
              **extra_env: str) -> subprocess.CompletedProcess[str]:
    """Run the driver with a fake `uv` that fails only on `fail_module`."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *\"$FAKE_FAIL_MODULE\"*) [ -n \"$FAKE_FAIL_MODULE\" ] && exit 1 ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    env = os.environ.copy()
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        GATE_OUT=str(tmp_path / "runs"),
        FAKE_FAIL_MODULE=fail_module,
        **extra_env,
    )
    return subprocess.run(
        ["bash", str(SCRIPT), *args], cwd=SCRIPT.parents[2], env=env,
        capture_output=True, text=True,
    )


@pytest.mark.parametrize("module", ["evaluation.tools.rtf", "evaluation.tools.collect"])
def test_a_failure_survives_tee_and_fails_the_driver(tmp_path, module):
    result = _run_gate(tmp_path, "test-tag", "model.ckpt", "recipe.yaml", fail_module=module)
    assert result.returncode == 1


def test_successful_runs_use_distinct_output_directories(tmp_path):
    for _ in range(2):
        assert _run_gate(tmp_path, "test-tag", "model.ckpt", "recipe.yaml").returncode == 0
    assert len(list((tmp_path / "runs").glob("gate_test-tag.*"))) == 2


def test_a_precomputed_directory_replaces_the_model_stages_and_excludes_a_checkpoint(tmp_path):
    """A third-party model's output has no checkpoint to load and no model to
    time. The driver must neither run those stages nor demand their results at
    collection, or the record can never be written."""
    out = tmp_path / "theirs"
    out.mkdir()
    result = _run_gate(tmp_path, "theirs-tag", PRECOMPUTED_DIR=str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0/5 preflight" not in result.stdout
    assert "5/5 CPU real-time factor: SKIPPED" in result.stdout
    assert "--precomputed " in result.stdout and "--precomputed-name theirs" in result.stdout
    assert "--require-stage cpu_rtf" not in result.stdout
    assert "--inference precomputed=true" in result.stdout
    assert f"--checkpoint {out}" in result.stdout
    assert "--recipe precomputed --inference" in result.stdout

    both = _run_gate(tmp_path, "tag", "model.ckpt", "recipe.yaml", PRECOMPUTED_DIR=str(out))
    assert both.returncode == 2 and "two systems" in both.stderr
