"""A record has to carry enough to be compared against another one, and the
records directory has to be readable without the recipes that produced it."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from puresound.evaluation.records import (
    GateRecord,
    StageResult,
    chain_commit,
    merge_stage_files,
    read_record,
    write_stage,
)
from puresound.evaluation.statistics import paired_bootstrap_ci, verdict

REPO = Path(__file__).resolve().parents[2]
RECORDS = REPO / "egs/noise_suppression/benchmarks/records"


def _stage(name="wer", role="gate", decision="pass", **overrides):
    fields = dict(
        name=name, metric="wer", role=role, n=200, value=0.27, baseline=0.29,
        difference={"point": -0.02, "ci_low": -0.03, "ci_high": -0.01},
        verdict=decision, direction="lower_is_better",
    )
    fields.update(overrides)
    return StageResult(**fields)


def _record(*stages):
    return GateRecord("t", "c.ckpt", "r.yaml", "abc1234", stages=list(stages))


@pytest.mark.parametrize(
    "stages, expected, unresolved",
    [
        ([_stage(decision="fail")], "fail", []),
        ([_stage(role="monitor", decision="fail"), _stage()], "pass", []),
        ([_stage(name="moderate_wer", decision="no-resolution")], "no-resolution", ["moderate_wer"]),
        ([_stage(role="monitor", decision="no-resolution"), _stage()], "pass", []),
        ([_stage(role="monitor")], "no-resolution", []),
    ],
    ids=["failing-gate", "failing-monitor", "unresolved-gate", "unresolved-monitor", "no-gates"],
)
def test_only_gate_stages_decide_the_verdict_and_a_record_without_one_does_not_pass(
    stages, expected, unresolved
):
    record = _record(*stages)
    assert record.verdict == expected
    assert record.unresolved == unresolved
    if unresolved:
        assert "could not resolve" in record.summary()


def test_the_record_round_trips_with_everything_a_comparison_needs(tmp_path):
    record = GateRecord(
        "candidate", "c.ckpt", "r.yaml", chain_commit(),
        inference={"dry_blend": 0.9}, stages=[_stage()],
    )
    path = record.write(tmp_path / "records" / "candidate.json")

    payload = read_record(path)
    for key in ("checkpoint", "recipe", "chain_commit", "inference", "stages", "verdict"):
        assert key in payload
    assert isinstance(payload["chain_commit"], str) and payload["chain_commit"]
    assert payload["stages"][0]["n"] == 200
    assert payload["stages"][0]["difference"]["ci_low"] == -0.03


def test_a_bad_role_or_verdict_is_rejected_at_construction():
    with pytest.raises(ValueError, match="role"):
        _stage(role="advisory")
    with pytest.raises(ValueError, match="verdict"):
        _stage(decision="probably")


def test_stages_written_by_separate_processes_merge_back(tmp_path):
    write_stage(tmp_path / "a.json", _stage(name="a"))
    write_stage(tmp_path / "b.json", [_stage(name="b"), _stage(name="c")])

    stages = merge_stage_files([tmp_path / "a.json", tmp_path / "b.json"])
    assert [stage.name for stage in stages] == ["a", "b", "c"]


def test_from_interval_keeps_both_the_absolute_value_and_the_delta():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0.30, 0.08, 200)
    treatment = baseline - 0.02 + rng.normal(0, 0.01, 200)
    interval = paired_bootstrap_ci(treatment, baseline, aggregate=np.mean)

    stage = StageResult.from_interval(
        "moderate_wer", metric="wer", role="gate",
        value=float(treatment.mean()), baseline=float(baseline.mean()),
        interval=interval, verdict=verdict(interval, direction="lower_is_better"),
        direction="lower_is_better",
    )

    # The delta decides the verdict; the absolute value is what stops a big
    # improvement on a terrible starting point reading as a good end state.
    assert stage.verdict == "pass"
    assert stage.value == pytest.approx(treatment.mean())
    assert stage.baseline == pytest.approx(baseline.mean())
    assert stage.n == 200


# --------------------------------------------------------------------------- #
# The naming convention of the records directory (benchmarks/README.md)
# --------------------------------------------------------------------------- #


def _records():
    return sorted(RECORDS.glob("*.json"))


def test_every_shipped_noise_suppression_checkpoint_has_its_record():
    """Also what stops the parametrised test below passing on an empty directory."""
    released = {
        json.loads(p.read_text(encoding="utf-8")).get("lineage", {}).get("released_as")
        for p in _records()
    }
    catalog = yaml.safe_load((REPO / "model_zoo/catalog.yaml").read_text(encoding="utf-8"))
    shipped = {entry["id"] for entry in catalog["models"] if entry["task"] == "noise_suppression"}
    assert shipped and shipped <= released, shipped - released


@pytest.mark.parametrize("path", _records(), ids=lambda p: p.stem)
def test_a_record_is_named_by_its_tag_and_says_what_it_was_testing(path):
    """The tag is the filename, the lineage names the variable under test, and a
    run name never carries `vN`: a version is assigned when a checkpoint is
    promoted and lives in `lineage.released_as` and the catalog."""
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record.get("tag") == path.stem
    lineage = record.get("lineage")
    assert lineage and lineage.get("variable"), "lineage names no variable under test"
    if not path.stem.startswith("ext_"):
        offending = [p for p in path.stem.split("_") if len(p) > 1 and p[0] == "v" and p[1:].isdigit()]
        assert not offending, f"{offending} in the run name; put the version in lineage.released_as"
