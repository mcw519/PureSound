"""The driver's last step: stage files in, one record out, exit code meaningful."""

import json

import pytest

from puresound.evaluation.records import StageResult, write_stage
from puresound.evaluation.tools.collect import main, parse_inference


def _stage(name, role="gate", decision="pass"):
    return StageResult(
        name=name, metric="pesq_wb", role=role, n=100, value=2.6, baseline=2.1,
        difference={"point": 0.5, "ci_low": 0.4, "ci_high": 0.6},
        verdict=decision, direction="higher_is_better",
    )


def _write_stages(tmp_path, stages):
    paths = []
    for index, stage in enumerate(stages):
        path = tmp_path / f"s{index}.json"
        write_stage(path, stage)
        paths.append(str(path))
    return paths


def _main(paths, out, *extra):
    return main([*paths, "--tag", "t", "--checkpoint", "c.ckpt", "--recipe", "r.yaml",
                 *extra, "--out", str(out)])


def test_inference_values_are_parsed_into_types_and_a_malformed_one_is_rejected():
    assert parse_inference(["dry_blend=0.9", "onset_guard=true", "mode=fast", "n=3"]) == {
        "dry_blend": 0.9, "onset_guard": True, "mode": "fast", "n": 3,
    }
    with pytest.raises(ValueError, match="NAME=VALUE"):
        parse_inference(["dry_blend"])


def test_stages_from_separate_files_become_one_record(tmp_path):
    out = tmp_path / "rec.json"
    code = _main(_write_stages(tmp_path, [_stage("a"), _stage("b")]), out,
                 "--inference", "dry_blend=0.9")
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert code == 0
    assert [stage["name"] for stage in payload["stages"]] == ["a", "b"]
    assert payload["chain_commit"]
    assert payload["inference"] == {"dry_blend": 0.9}


@pytest.mark.parametrize(
    "stages, code, verdict, unresolved",
    [
        ([_stage("a"), _stage("b", decision="fail")], 1, "fail", []),
        ([_stage("a"), _stage("b", role="monitor", decision="fail")], 0, "pass", []),
        ([_stage("a", decision="no-resolution")], 1, "no-resolution", ["a"]),
    ],
    ids=["failing-gate", "failing-monitor", "unresolved-gate"],
)
def test_the_exit_status_follows_the_gate_stages_only(tmp_path, stages, code, verdict, unresolved):
    out = tmp_path / "rec.json"
    assert _main(_write_stages(tmp_path, stages), out) == code
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verdict"] == verdict
    assert payload["unresolved_gates"] == unresolved


@pytest.mark.parametrize("case", ["missing-required-stage", "missing-stage-file"])
def test_an_incomplete_collection_fails_and_writes_nothing(tmp_path, case):
    out = tmp_path / "record.json"
    if case == "missing-required-stage":
        code = _main(_write_stages(tmp_path, [_stage("present")]), out,
                     "--require-stage", "missing")
    else:
        code = _main([str(tmp_path / "missing.json")], out)
    assert code == 1
    assert not out.exists()
