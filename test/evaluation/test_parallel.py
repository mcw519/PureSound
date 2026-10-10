"""Worker-parallel scoring: same answers, in the same order, from one loader."""

import os

import pytest

from puresound.evaluation.parallel import default_jobs, map_items


def _double(item, state):
    return item * 2 * state.get("factor", 1)


def _needs_state(item, state):
    return state["greeting"] + str(item)


def _thread_counts_in_worker(item, state):
    import numba
    import threadpoolctl
    import torch

    pools = {
        (p["user_api"], p.get("internal_api")): p["num_threads"]
        for p in threadpoolctl.threadpool_info()
    }
    return numba.get_num_threads(), torch.get_num_threads(), pools


@pytest.mark.parametrize(
    "items, jobs",
    [(list(range(20)), 1), (list(range(20)), 3), ([1], 16), ([], 4)],
    ids=["inline", "pooled", "more-jobs-than-items", "empty"],
)
def test_results_come_back_in_input_order(items, jobs):
    # More workers than items must neither hang nor spawn them; no items is not an error.
    assert map_items(_double, items, jobs=jobs) == [i * 2 for i in items]


def test_the_builder_runs_once_per_worker_and_its_state_reaches_the_function():
    """The model is built in the initialiser rather than shipped per item --
    a checkpoint is tens of megabytes and the work is milliseconds."""
    out = map_items(_needs_state, [1, 2], jobs=2, builder=lambda: {"greeting": "hi"})
    assert out == ["hi1", "hi2"]


def test_progress_reports_the_final_count():
    seen = []
    map_items(_double, list(range(7)), jobs=1, progress=lambda d, t: seen.append((d, t)),
              progress_every=3)
    assert seen[-1] == (7, 7)


def test_the_default_leaves_room_for_a_training_run():
    assert 1 <= default_jobs() <= max(1, (os.cpu_count() or 2) // 2)


def test_every_thread_pool_in_a_worker_is_pinned_to_one():
    """Workers are forked and inherit the parent's thread counts, so each pool is
    pinned by API in the initialiser -- numba and torch, and also the BLAS and
    OpenMP pools that PESQ and STOI (numpy/scipy) run in, which
    `torch.set_num_threads` never reaches."""
    pytest.importorskip("threadpoolctl")
    for numba_threads, torch_threads, pools in map_items(_thread_counts_in_worker, [0, 1], jobs=2):
        assert (numba_threads, torch_threads) == (1, 1)
        assert pools, "no thread pools reported; the check would pass vacuously"
        assert all(count == 1 for count in pools.values()), pools
