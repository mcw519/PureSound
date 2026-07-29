"""Repo-wide checks: ruff (correctness rules, policy pinned in pyproject.toml)
plus the pytest suite. `--suite quick` runs the fast subset; `--suite full`
runs everything."""
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


def _run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["quick", "full"],
        default="quick",
        help="quick runs focused smoke tests; full runs the whole pytest suite.",
    )
    parser.add_argument("--skip-ruff", action="store_true")
    parser.add_argument("pytest_args", nargs="*", help="Extra args passed to pytest.")
    args = parser.parse_args()

    if not args.skip_ruff:
        _run([sys.executable, "-m", "ruff", "check", "."])

    test_targets = QUICK_TESTS if args.suite == "quick" else ["test"]
    _run([sys.executable, "-m", "pytest", "-q", *test_targets, *args.pytest_args])


if __name__ == "__main__":
    main()
