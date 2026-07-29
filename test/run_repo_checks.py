import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

QUICK_TESTS = [
    "test/test_metrics",
    "test/test_losses",
    "test/test_utils",
]

CHANGED_LINT_TARGETS = [
    "egs/noise_suppression/main.py",
    "egs/target_speaker_extraction/main.py",
    "egs/voice_isolate",
    "puresound/audio/room_simulator.py",
    "puresound/audio/vad.py",
    "puresound/metrics.py",
    "puresound/nnet/loss/vad.py",
    "puresound/system/miso.py",
    "puresound/system/siso.py",
    "test/conftest.py",
    "test/run_repo_checks.py",
    "test/test_losses",
    "test/test_metrics",
    "test/test_utils",
]


def _run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PureSound repo checks.")
    parser.add_argument(
        "--suite",
        choices=["quick", "full"],
        default="quick",
        help="quick runs focused smoke tests; full runs the whole pytest suite.",
    )
    parser.add_argument("--skip-ruff", action="store_true")
    parser.add_argument(
        "--ruff-scope",
        choices=["changed", "all"],
        default="changed",
        help="`all` is useful after legacy lint debt has been fixed.",
    )
    parser.add_argument("pytest_args", nargs="*", help="Extra args passed to pytest.")
    args = parser.parse_args()

    if not args.skip_ruff:
        lint_targets = ["."]
        if args.ruff_scope == "changed":
            lint_targets = CHANGED_LINT_TARGETS
        _run([sys.executable, "-m", "ruff", "check", *lint_targets])

    test_targets = QUICK_TESTS if args.suite == "quick" else ["test"]
    _run([sys.executable, "-m", "pytest", "-q", *test_targets, *args.pytest_args])


if __name__ == "__main__":
    main()
