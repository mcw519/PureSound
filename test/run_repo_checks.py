"""Repo-wide checks: ruff (correctness rules, policy pinned in pyproject.toml)
plus the pytest suite.

Three tiers, cheapest first:

* ``--suite quick``    unit tests only (the ``test/*/`` subpackages), ~20 s.
                       Use while iterating on library code.
* ``--suite standard`` everything except tests marked ``slow``, ~35 s. Worth
                       it on a smaller machine, where the gap below widens.
* ``--suite full``     everything, ~2 min. The default: at that price there is
                       rarely a reason to verify less.

Timings are wall clock on 24 cores; the same suite takes over an hour serially.
Every tier runs under pytest-xdist with one compute thread per worker (see
``_single_threaded_env``). Override the worker count with ``--jobs N``
(``--jobs 0`` disables xdist, which is what you want when you need a debugger
or readable live output)."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Fast, dependency-light unit tests. Everything else lives at test/ root and is
# mostly RIR/room-acoustics, which is where the runtime goes.
QUICK_TESTS = [
    "test/test_audio",
    "test/test_losses",
    "test/test_metrics",
    "test/test_system",
    "test/test_utils",
]


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


def _single_threaded_env() -> dict[str, str]:
    """Pin each xdist worker to one compute thread.

    torch defaults to one intra-op thread per core, so N workers each spawn N
    threads: on a 24-core box `-n auto` means 288 threads contending for 24
    cores. Measured cost of leaving it alone: the 63 `slow` tests burned 339
    CPU-minutes to finish in 76, while the whole 668-test suite needed only
    273 CPU-minutes when workers were less oversubscribed. Under xdist the
    parallelism belongs across tests, not inside them.
    """
    return {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--suite",
        choices=["quick", "standard", "full"],
        default="full",
        help="quick: unit tests only; standard: everything but `slow`; full: everything.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="xdist worker count for standard/full (default: auto; 0 disables xdist).",
    )
    parser.add_argument("--skip-ruff", action="store_true")
    parser.add_argument("pytest_args", nargs="*", help="Extra args passed to pytest.")
    args = parser.parse_args()

    if not args.skip_ruff:
        _run([sys.executable, "-m", "ruff", "check", "."])

    pytest_cmd = [sys.executable, "-m", "pytest", "-q"]

    if args.suite == "quick":
        pytest_cmd += QUICK_TESTS
    else:
        pytest_cmd.append("test")

    # `full` is the only tier that runs the slow tests. Note quick needs the
    # filter too: test_utils/ holds the two offline-vs-streaming comparisons,
    # ~5 minutes between them on their own.
    if args.suite != "full":
        pytest_cmd += ["-m", "not slow"]

    env = None
    if args.jobs != 0:
        pytest_cmd += ["-n", "auto" if args.jobs is None else str(args.jobs)]
        env = _single_threaded_env()

    _run([*pytest_cmd, *args.pytest_args], env=env)


if __name__ == "__main__":
    main()
