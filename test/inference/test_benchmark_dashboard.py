"""The benchmark dashboard (`tools/benchmark_dashboard.py`): how runs fold into
the history, and what the published page and its assets carry."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools/benchmark_dashboard.py"


@pytest.fixture(scope="module")
def dashboard():
    spec = importlib.util.spec_from_file_location("benchmark_dashboard_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(revision, system, created_at, rtf, *, description="A model."):
    return {
        "schema_version": 1,
        "created_at": created_at,
        "revision": revision,
        "repository": "https://github.com/example/PureSound",
        "machine": {"system": system, "machine": "x86_64", "cpu": f"{system} cpu", "logical_cpus": 4,
                    "onnxruntime": "1.26.0", "ci_image": None},
        "models": {"ns-model": {
            "display_name": "NS model", "task": "noise_suppression", "lifecycle": "released",
            "roles": ["default"], "description": description, "recommended_inference": {"dry_blend": 1.0},
            "card": "egs/noise_suppression/pretrained_ckpt/README.md",
            "artifacts": {"default": {"file": "ns.onnx", "description": None, "size_bytes": 2_000_000},
                          "flash": {"file": "ns_flash.onnx", "description": "int8", "size_bytes": 1_000_000}},
        }},
        "results": [
            {"model_id": "ns-model", "variant": "default", "rtf": rtf, "cold_start_seconds": 0.02,
             "native_ssm": False, "intra_op_threads": 1, "latency_ms": 30.0},
            {"model_id": "ns-model", "variant": "flash", "rtf": rtf * 0.9, "cold_start_seconds": 0.02,
             "native_ssm": False, "intra_op_threads": 1, "latency_ms": 30.0},
        ],
    }


def test_one_revision_is_one_run_and_a_rerun_replaces_it(dashboard):
    history = dashboard.merge({}, [_record("aaa", "Linux", "2026-10-01T00:00:00+00:00", 0.3),
                                   _record("aaa", "Darwin", "2026-10-01T00:01:00+00:00", 0.2)], keep=10)
    assert [run["revision"] for run in history["runs"]] == ["aaa"]
    assert sorted(history["runs"][0]["platforms"]) == ["linux", "macos"]

    history = dashboard.merge(history, [_record("aaa", "Linux", "2026-10-02T00:00:00+00:00", 0.4)], keep=10)
    (run,) = history["runs"]
    assert list(run["platforms"]) == ["linux"]
    assert run["platforms"]["linux"]["results"][0]["rtf"] == 0.4


def test_the_oldest_runs_drop_off_and_the_newest_cards_win(dashboard):
    history = {}
    for day in range(1, 5):
        record = _record(f"rev{day}", "Linux", f"2026-10-0{day}T00:00:00+00:00", 0.3,
                         description=f"as of day {day}")
        history = dashboard.merge(history, [record], keep=3)
    assert [run["revision"] for run in history["runs"]] == ["rev2", "rev3", "rev4"]
    assert history["models"]["ns-model"]["description"] == "as of day 4"
    assert history["repository"] == "https://github.com/example/PureSound"


def test_the_page_carries_cards_filters_and_the_playground_assets(dashboard, tmp_path):
    history = dashboard.merge({}, [_record("abc1234def", "Linux", "2026-10-01T00:00:00+00:00", 0.62),
                                   _record("abc1234def", "Windows", "2026-10-01T00:00:00+00:00", 0.31)],
                              keep=10)
    dashboard.write_site(tmp_path, history)
    page = (tmp_path / "index.html").read_text(encoding="utf-8")

    assert page.count('class="model-card bench-card is-default"') == 1
    assert 'data-filter="task"' in page and 'data-toggle="p-linux"' in page and 'data-toggle="v-flash"' in page
    # A default artifact over the budget is flagged; the flash one is not held to it.
    assert '<td class="num p-linux is-over">0.620</td>' in page
    assert '<td class="num p-linux">0.558</td>' in page
    assert ("https://github.com/example/PureSound/blob/abc1234def/"
            "egs/noise_suppression/pretrained_ckpt/README.md") in page
    # The playground's stylesheets, with fonts found beside them under any base path.
    for name in dashboard.SHARED_CSS:
        assert (tmp_path / "css" / name).is_file()
    base = (tmp_path / "css/base.css").read_text(encoding="utf-8")
    assert 'url("/fonts/' not in base and 'url("../fonts/IBMPlexSans-400.woff2")' in base
    assert (tmp_path / "fonts/IBMPlexSans-400.woff2").is_file()
    assert (tmp_path / "history.json").is_file()
