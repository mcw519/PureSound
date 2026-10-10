"""Score a set across worker processes instead of one item at a time.

Scoring one item -- model inference, metrics, reading -- is CPU-bound and
single-threaded, so a serial loop uses one core of the machine.

Processes rather than threads, for the same reason the corpus resampler uses
them: the metric libraries and `torchaudio`'s sox path sit on C state that does
not survive being called from several threads at once.

The model is built once per worker, in an initialiser, rather than pickled with
every item. A checkpoint is tens of megabytes; shipping it per item would cost
more than the work.

Each worker is pinned to one compute thread. Without that, N workers each spawn
a pool per core and the machine spends its time in the scheduler.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterable, Sequence

_WORKER_STATE: dict[str, Any] = {}


def _initialise(builder: Callable[[], dict[str, Any]] | None) -> None:
    """Pin the worker's thread pools (`puresound.utils.pin_thread_pools`), then build its state."""
    from puresound.utils import pin_thread_pools

    pin_thread_pools()
    if builder is not None:
        _WORKER_STATE.update(builder())


def _apply(payload: tuple[Callable[..., Any], Any]) -> Any:
    function, item = payload
    return function(item, _WORKER_STATE)


def map_items(
    function: Callable[[Any, dict[str, Any]], Any],
    items: Sequence[Any],
    *,
    jobs: int | None = None,
    builder: Callable[[], dict[str, Any]] | None = None,
    progress: Callable[[int, int], None] | None = None,
    progress_every: int = 50,
) -> list[Any]:
    """Apply ``function(item, worker_state)`` to every item, in order.

    ``jobs=1`` runs inline with no pool -- what you want under a debugger, and
    what keeps a unit test from paying for process startup.
    """
    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) // 2)
    jobs = max(1, min(jobs, len(items) or 1))

    if jobs == 1:
        _initialise(builder)
        results = []
        for index, item in enumerate(items, start=1):
            results.append(function(item, _WORKER_STATE))
            if progress is not None and index % progress_every == 0:
                progress(index, len(items))
        if progress is not None and items:
            progress(len(items), len(items))
        return results

    payloads: Iterable[tuple[Callable[..., Any], Any]] = ((function, item) for item in items)
    results = []
    with ProcessPoolExecutor(
        max_workers=jobs, initializer=_initialise, initargs=(builder,)
    ) as pool:
        for index, value in enumerate(pool.map(_apply, payloads, chunksize=4), start=1):
            results.append(value)
            if progress is not None and index % progress_every == 0:
                progress(index, len(items))
    if progress is not None and items:
        progress(len(items), len(items))
    return results


def default_jobs() -> int:
    """Half the cores: the other half keeps a concurrent training run alive."""
    return max(1, (os.cpu_count() or 2) // 2)
