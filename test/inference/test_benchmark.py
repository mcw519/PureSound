"""Benchmarking the catalog's streaming artifacts: what is selected, what a
record carries, and the ``puresound models benchmark`` command."""

from __future__ import annotations

import json

import pytest

from puresound.cli import main
from puresound.inference import ModelZoo
from puresound.inference.benchmark import STREAMING_PROCESSOR, run_benchmark, streaming_artifacts


@pytest.fixture(scope="module")
def zoo():
    return ModelZoo.default()


def test_every_streaming_artifact_is_selected_and_nothing_else(zoo):
    pairs = streaming_artifacts(zoo)
    expected = {
        (model.id, artifact.variant)
        for model in zoo.list()
        for artifact in model.artifacts
        if artifact.processor == STREAMING_PROCESSOR
    }
    assert {(model.id, artifact.variant) for model, artifact in pairs} == expected
    assert all(artifact.processor == STREAMING_PROCESSOR for _, artifact in pairs)


def test_selection_narrows_by_model_and_variant(zoo):
    pairs = streaming_artifacts(zoo, ["voice-isolate-dpcrn-v8"], ["flash"])
    assert [(model.id, artifact.variant) for model, artifact in pairs] == [
        ("voice-isolate-dpcrn-v8", "flash")
    ]


@pytest.mark.parametrize(
    "models, variants, error, match",
    [
        # Each of these would otherwise measure nothing and still succeed.
        (["speaker-verification-ps-spk-v1-1"], None, ValueError, "no streaming artifact"),
        (None, ["no-such-variant"], ValueError, "no-such-variant"),
        (["no-such-model"], None, KeyError, "no-such-model"),
    ],
)
def test_a_selection_that_measures_nothing_is_refused(zoo, models, variants, error, match):
    with pytest.raises(error, match=match):
        streaming_artifacts(zoo, models, variants)


def test_a_record_names_the_graph_and_threads_its_timing_belongs_to(zoo):
    record = run_benchmark(zoo, models=["voice-isolate-dpcrn-v8"], variants=["default"],
                           seconds=0.5, repeats=2)

    assert record["schema_version"] == 1
    assert record["machine"]["onnxruntime"]
    (result,) = record["results"]
    assert result["model_id"] == "voice-isolate-dpcrn-v8" and result["variant"] == "default"
    assert result["rtf"] > 0 and len(result["rtf_repeats"]) == 2
    assert result["provider"] == "CPUExecutionProvider"
    assert result["execution_graph"] == "dpcrn_v8.onnx" and result["native_ssm"] is False
    # Only an int8 artifact has a distance from the float graph to report.
    assert result["int8_relative_error"] is None
    # The manifest recommends one thread for the optimized streaming graphs.
    assert result["intra_op_threads"] == 1
    # The record carries the catalog card of what it measured, so it reads alone.
    card = record["models"]["voice-isolate-dpcrn-v8"]
    assert card["task"] == "voice_isolation" and card["display_name"]
    assert card["card"] == "egs/voice_isolate/pretrained_ckpt/README.md"
    assert set(card["artifacts"]) == {"default", "flash"}
    assert all(artifact["size_bytes"] > 0 for artifact in card["artifacts"].values())


def test_the_command_prints_a_table_and_writes_the_record(tmp_path, capsys):
    path = tmp_path / "bench.json"
    assert main(["models", "benchmark", "--model", "voice-isolate-dpcrn-v8", "--variant", "flash",
                 "--seconds", "0.5", "--repeats", "1", "--json", str(path)]) == 0
    assert "voice-isolate-dpcrn-v8" in capsys.readouterr().out
    record = json.loads(path.read_text(encoding="utf-8"))
    assert [(r["model_id"], r["variant"]) for r in record["results"]] == [
        ("voice-isolate-dpcrn-v8", "flash")
    ]
    assert 0 < record["results"][0]["int8_relative_error"] < 0.1
