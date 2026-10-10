"""Runs the static client's pure-logic tests (``test/web/*.test.cjs``) under
Node; skipped where Node is not installed."""

import shutil
import subprocess
from pathlib import Path

import pytest

TESTS = sorted(Path(__file__).parent.glob("*.test.cjs"))


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("script", TESTS, ids=[path.name for path in TESTS])
def test_the_static_client_logic_passes_its_node_tests(script):
    result = subprocess.run(["node", "--test", str(script)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
