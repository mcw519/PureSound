"""The library's output contract: on by default, silenceable, rank-zero only.

All three are easy to break by accident -- an added `print`, an `__init__`
import that skips autoconfigure, a handler installed without the rank filter --
and none of them show up in any other test.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from puresound.logging_setup import (
    LOGGER_NAME,
    RankZeroFilter,
    configure_library_logging,
    distributed_rank,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(snippet: str, **env_overrides) -> str:
    env = {**os.environ, **{k: str(v) for k, v in env_overrides.items()}}
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


EMIT = (
    "import logging, puresound;"
    "logging.getLogger('puresound.probe').info('MARKER')"
)


def test_library_output_is_on_by_default_and_goes_to_stdout():
    # stdout, not stderr: that is where the `print` calls this replaced went, so
    # a shell redirect that captured them still does.
    assert "MARKER" in _run(EMIT, PURESOUND_LOG_AUTOCONFIG="1")


def test_library_output_can_be_switched_off_entirely():
    assert "MARKER" not in _run(EMIT, PURESOUND_LOG_AUTOCONFIG="0")


def test_library_output_is_rank_zero_only():
    # The one property the acceptance criterion names: two DDP ranks importing
    # the same code must not print the corpus statistics twice.
    assert "MARKER" in _run(EMIT, RANK="0", LOCAL_RANK="0")
    assert "MARKER" not in _run(EMIT, RANK="1", LOCAL_RANK="1")


def test_a_record_can_opt_out_of_rank_filtering():
    snippet = (
        "import logging, puresound;"
        "logging.getLogger('puresound.probe')"
        ".warning('MARKER', extra={'all_ranks': True})"
    )
    assert "MARKER" in _run(snippet, RANK="1", LOCAL_RANK="1")


@pytest.mark.parametrize(
    "environ,expected",
    [
        ({}, 0),
        ({"RANK": "3"}, 3),
        ({"LOCAL_RANK": "2"}, 2),
        ({"RANK": "not-a-number", "LOCAL_RANK": "5"}, 5),
        ({"SLURM_PROCID": "7"}, 7),
    ],
)
def test_distributed_rank_reads_the_launcher_environment(monkeypatch, environ, expected):
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK"):
        monkeypatch.delenv(key, raising=False)
    for key, value in environ.items():
        monkeypatch.setenv(key, value)
    assert distributed_rank() == expected


def test_rank_filter_reads_rank_per_record(monkeypatch):
    """Not cached at construction: the handler is installed at import time,
    which on a spawned worker precedes the launcher setting LOCAL_RANK."""
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK"):
        monkeypatch.delenv(key, raising=False)
    rank_filter = RankZeroFilter()
    record = logging.LogRecord("puresound.probe", logging.INFO, __file__, 1, "m", (), None)

    assert rank_filter.filter(record) is True
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert rank_filter.filter(record) is False


def test_configure_is_idempotent_and_never_touches_foreign_handlers():
    logger = logging.getLogger(LOGGER_NAME)
    foreign = logging.NullHandler()
    logger.addHandler(foreign)
    before = list(logger.handlers)
    try:
        configure_library_logging()
        configure_library_logging()
        assert list(logger.handlers) == before, "repeat calls must not stack handlers"

        configure_library_logging(level=logging.WARNING, force=True)
        assert foreign in logger.handlers, "a handler we do not own must survive"
        assert len(logger.handlers) == len(before), "force must swap, not append"
        assert logger.level == logging.WARNING
    finally:
        logger.removeHandler(foreign)
        configure_library_logging(level=logging.INFO, force=True)
