import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _assert_help(script: str) -> None:
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "usage:" in result.stdout


def test_manual_test_scripts_expose_help():
    _assert_help("test/run_repo_checks.py")
    _assert_help("test/generate_simulated_training_data.py")


def test_voice_isolate_data_scripts_expose_help():
    _assert_help("test/simulate_room_scene.py")
