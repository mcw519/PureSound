"""World jobs share the existing store, exports and cancellation lifecycle."""

import io
import json
import time
import zipfile
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from puresound.audio.rir.scene.world_presets import world_preset
from puresound.web import WebService
from puresound.web.server import WebServiceError, _float_wav
from puresound.web.world import validate_request, sweep_scenes, world_zip
from puresound.audio.rir.render.dynamic import render_dynamic_scene
from puresound.evaluation.world import world_metrics


def request():
    scene = replace(
        world_preset("turn", duration_s=5.0), max_order=0, late_reverb=False
    )
    return {"kind": "world_render", "scene": scene.to_dict()}


def wait(service, job):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = service.jobs.snapshot(job["job_id"])
        if state["status"] in {"succeeded", "failed", "cancelled"}:
            return state
        time.sleep(0.01)
    pytest.fail("world job did not finish")


def test_world_job_package_replays_and_exports(tmp_path):
    service = WebService(history_dir=tmp_path)
    job = wait(service, service.submit_world(request()))
    assert job["status"] == "succeeded", job["error"]
    report = job["result"]
    token = report["run_id"]
    with zipfile.ZipFile(
        io.BytesIO(service.run_output(token, "scene-package").data)
    ) as archive:
        saved = json.loads(archive.read("scene.json"))
        spec = validate_request({"scene": saved["scene"]})
        assets = {
            key: sf.read(io.BytesIO(archive.read(value["file"])), dtype="float32")[0]
            for key, value in saved["assets"].items()
        }
        rendered = render_dynamic_scene(spec, assets)
        original = sf.read(io.BytesIO(archive.read("input.wav")), dtype="float32")[0]
        np.testing.assert_array_equal(rendered.mixture, original)
        assert rendered.metadata["asset_hashes"] == report["metadata"]["asset_hashes"]
    package, content_type, _ = service.job_export(job["job_id"], "zip")
    assert content_type == "application/zip"
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        assert {
            "scene-package.zip",
            "reference-target.wav",
            "reference-speech.wav",
            "reference-near.wav",
            "source-speaker-a.wav",
        } <= set(archive.namelist())
    page, _, _ = service.job_export(job["job_id"], "report")
    assert b"reference-speech" in page and b"all speech" in page
    restored = WebService(history_dir=tmp_path).jobs.snapshot(job["job_id"])
    assert restored["result"]["metadata"]["renderer"] == report["metadata"]["renderer"]


def test_sweep_bounds_failures_and_single_cell_retry():
    payload = {
        **request(),
        "kind": "world_sweep",
        "axes": [
            {"parameter": "noise_gain_change_db", "values": [-24, -12]},
            {"parameter": "target_distance_change_m", "values": [0, 100]},
        ],
    }
    cells = sweep_scenes(payload)
    assert [c["index"] for c in cells] == [0, 1, 2, 3]
    assert "error" in cells[1] and "scene" in cells[0]
    assert [c["index"] for c in sweep_scenes({**payload, "cell_indices": [2]})] == [2]
    payload["axes"][0]["values"] = list(range(14))
    with pytest.raises(ValueError, match="25"):
        sweep_scenes(payload)
    with pytest.raises(WebServiceError):
        WebService().submit_world({**request(), "scene": {}})


def test_cancel_sweep_retains_cell_states(monkeypatch):
    service = WebService()
    calls = []

    def render(payload, spec, assets, progress):
        calls.append(1)
        if len(calls) == 2:
            service.jobs.cancel(job.job_id)
            progress(0.5, "rendering")
        return {"metrics": {}}

    monkeypatch.setattr(service, "_world_result", render)
    job = service.jobs.create("acoustic-world", kind="world_sweep")
    payload = {
        **request(),
        "kind": "world_sweep",
        "axes": [
            {"parameter": "noise_gain_change_db", "values": [-24, -12, -6]},
            {"parameter": "interferer_gain_change_db", "values": [-6]},
        ],
    }
    service._run_world_job(job.job_id, payload)
    state = service.jobs.snapshot(job.job_id)
    assert state["status"] == "cancelled"
    assert [cell["status"] for cell in state["result"]["cells"]] == [
        "succeeded",
        "cancelled",
        "cancelled",
    ]
    package, content_type, _ = service.job_export(job.job_id, "zip")
    assert content_type == "application/zip"
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        saved = json.loads(archive.read("report.json"))
        assert saved["status"] == "cancelled"
    page, _, _ = service.job_export(job.job_id, "report")
    assert b"cancelled" in page and b"model limit map" in page


def test_empty_near_reference_reports_residual_and_policy_does_not_mutate_output():
    spec = replace(
        world_preset("turn", duration_s=0.2),
        near_radius_m=0.2,
        max_order=0,
        late_reverb=False,
    )
    assets = {s.asset_id: np.ones(3200, dtype=np.float32) * 0.01 for s in spec.sources}
    rendered = render_dynamic_scene(spec, assets)
    output = rendered.mixture.copy()
    before = output.copy()
    metrics = world_metrics(rendered, output)
    assert metrics["near"]["output"] is None
    assert all(
        w["improvement_db"] is None and w["residual_dbfs"] is not None
        for w in metrics["near"]["windows"]
    )
    np.testing.assert_array_equal(output, before)
    assert metrics["target"]["output"] is not None
    report = {"scene": spec.to_dict(), "scene_assets": {}, "metrics": metrics}
    assert world_zip(report, {}, {}, _float_wav)


def test_import_validation_rejects_unsafe_identifiers_before_ui_display():
    payload = request()
    payload["scene"]["sources"][0]["source_id"] = "<img onerror=alert(1)>"
    with pytest.raises(WebServiceError, match="IDs"):
        WebService().validate_world(payload)
    valid = WebService().validate_world(request())
    assert valid["scene"]["schema_version"] == "puresound.dynamic_scene.v2"


def test_overview_reports_the_scene_limits_and_every_room_type():
    from puresound.audio.rir.scene.dynamic import MAX_SPEED_M_S, MIN_MIC_DISTANCE_M
    from puresound.audio.rir.scene.materials import ROOM_TYPE_RECIPES

    overview = WebService().world_overview()
    assert overview["limits"]["speed_m_s"] == MAX_SPEED_M_S
    assert overview["limits"]["min_mic_distance_m"] == MIN_MIC_DISTANCE_M
    assert overview["materials"] == sorted(ROOM_TYPE_RECIPES)
    talkers = [
        tr["directivity_id"]
        for tr in overview["presets"]["approach"]["room"]["sources"]
        if tr["transducer_id"].startswith("speaker")
    ]
    assert set(talkers) == {"speech_human"}


def test_world_speech_samples_keep_the_complete_original_utterance():
    from pathlib import Path
    from puresound.web.world import ASSETS

    service = WebService()
    original, rate = sf.read(Path(__file__).parents[1] / "test_case/1272-141231-0008.flac")
    for name in ("speaker-a", "speaker-a-2"):
        audio, actual_rate = sf.read(service.static_dir / ASSETS[name])
        assert actual_rate == rate == 16000
        assert len(audio) == len(original)
        normalized = original * (10 ** (-3 / 20) / np.max(np.abs(original)))
        np.testing.assert_allclose(audio, normalized, atol=1 / 32768)
    durations = {a["id"]: a["duration_s"] for a in service.world_overview()["assets"]}
    assert durations["speaker-a"] == len(original) / rate


def test_materials_are_redrawn_for_a_room_type_and_seed():
    service = WebService()
    base = request()["scene"]
    classroom = service.world_materials({"scene": base, "room_type": "classroom"})["scene"]
    assert classroom["room"]["room_type"] == "classroom"
    assert classroom["room"]["dimensions_m"] == base["room"]["dimensions_m"]
    reseeded = service.world_materials({"scene": {**classroom, "seed": 7}})["scene"]
    assert reseeded["room"]["room_type"] == "classroom"
    assert reseeded["room"]["materials"] != classroom["room"]["materials"]
    with pytest.raises(WebServiceError, match="room type"):
        service.world_materials({"scene": base, "room_type": "cathedral"})
    # A scene mid-edit (a keyframe too close to the microphone) can still
    # change room type; only the room is checked.
    broken = json.loads(json.dumps(base))
    broken["sources"][0]["keyframes"][0]["position_m"] = [1.05, 2.5, 1.4]
    office = service.world_materials({"scene": broken, "room_type": "office", "seed": 3})["scene"]
    assert office["seed"] == 3 and office["sources"] == broken["sources"]


def test_running_sweeps_poll_cell_scores_without_audio():
    from puresound.web.world import sweep_progress

    assert sweep_progress(None) is None and sweep_progress({"metrics": {}}) is None
    result = {
        "axes": [{"parameter": "noise_db", "values": [0]}, {"parameter": "interferer_db", "values": [0, 6]}],
        "cells": [
            {"index": 0, "x": 0, "y": 0, "status": "succeeded", "result": {"output_urls": {"input": "/x"}, "metrics": {"fixed": {"output": {"si_sdr_db": 4.5}}, "near": {"output": None}}}},
            {"index": 1, "x": 0, "y": 6, "status": "queued"},
        ],
    }
    progress = sweep_progress(result)
    assert progress["cells"][0]["si_sdr_db"] == {"fixed": 4.5, "near": None}
    assert progress["cells"][1]["status"] == "queued" and "result" not in progress["cells"][1]


def test_a_streaming_model_a_few_ms_short_is_padded():
    from puresound.web.world import render_report

    spec = validate_request(request())
    assets = {s.asset_id: np.ones(3200, dtype=np.float32) * 0.01 for s in spec.sources}
    report, audio, _ = render_report(spec, assets, lambda x, sr: x[:-128] * 0.5, None, lambda v, p: None)
    assert len(audio["aligned"]) == len(audio["input"])
    assert not np.any(audio["aligned"][-128:])
    with pytest.raises(ValueError, match="length"):
        render_report(spec, assets, lambda x, sr: x[: len(x) // 2], None, lambda v, p: None)



def busy_request():
    return {"kind": "world_sweep", "scene": world_preset("busy", duration_s=2.0).to_dict()}


def test_sweep_offsets_move_every_source_of_a_kind():
    base = world_preset("busy", duration_s=2.0)
    payload = {
        **busy_request(),
        "axes": [
            {"parameter": "noise_gain_change_db", "values": [-6]},
            {"parameter": "interferer_gain_change_db", "values": [3]},
        ],
    }
    [cell] = sweep_scenes(payload)
    gains = {s.source_id: s.gain_db for s in cell["scene"].sources}
    for source in base.sources:
        change = {"noise": -6, "interferer": 3, "target": 0}[source.role]
        assert gains[source.source_id] == source.gain_db + change
    payload["axes"][1] = {"parameter": "target_distance_change_m", "values": [0.5]}
    [cell] = sweep_scenes(payload)
    mic = np.array(base.room.mic_pos)
    before = np.array(base.sources[0].keyframes[0].position_m)
    after = np.array(cell["scene"].sources[0].keyframes[0].position_m)
    assert np.linalg.norm(after - mic) == pytest.approx(np.linalg.norm(before - mic) + 0.5)


def test_sweep_rejects_a_parameter_with_nothing_to_change():
    scene = request()["scene"]
    scene["sources"] = [s for s in scene["sources"] if s["role"] != "noise"]
    scene["room"]["sources"] = [t for t in scene["room"]["sources"] if t["transducer_id"] != "noise"]
    payload = {
        "kind": "world_sweep",
        "scene": scene,
        "axes": [
            {"parameter": "noise_gain_change_db", "values": [-6, 0]},
            {"parameter": "interferer_gain_change_db", "values": [0]},
        ],
    }
    with pytest.raises(ValueError, match="no noise source"):
        sweep_scenes(payload)


def test_overview_offers_task_defaults_and_every_sample():
    overview = WebService().world_overview()
    assert overview["policy_defaults"] == {"noise_suppression": "speech", "voice_isolation": "near"}
    assert {"speaker-a-2", "hum", "babble"} <= {a["id"] for a in overview["assets"]}
    assert overview["limits"]["sources"] == 8
    assert overview["sweep_parameters"] == ["noise_gain_change_db", "interferer_gain_change_db", "target_distance_change_m"]
    assert overview["sweep_roles"] == {"noise_gain_change_db": "noise", "interferer_gain_change_db": "interferer", "target_distance_change_m": "target"}


def test_a_v1_scene_package_still_validates():
    v1 = world_preset("turn").to_dict()
    v1.pop("reference_rir")
    v1["schema_version"] = "puresound.dynamic_scene.v1"
    v1["fixed_source_id"] = "speaker-a"
    for source in v1["sources"]:
        source["role"] = "noise" if source["role"] == "noise" else "speech"
    scene = WebService().validate_world({"scene": v1})["scene"]
    assert scene["schema_version"].endswith("v2")
    assert [s["role"] for s in scene["sources"]] == ["target", "interferer", "noise"]


def _v1_report():
    scene = world_preset("turn").to_dict()
    scene.pop("reference_rir")
    scene["schema_version"] = "puresound.dynamic_scene.v1"
    scene["fixed_source_id"] = "speaker-a"
    for source in scene["sources"]:
        source["role"] = "noise" if source["role"] == "noise" else "speech"
    scores = {"windows": [], "input": None, "output": None, "purpose": ""}
    return {
        "scene": scene,
        "metrics": {"fixed": scores, "near": scores},
        "output_urls": {"input": "/a", "target-fixed": "/b", "target-near": "/c", "source-noise": "/d"},
    }


def test_history_reopens_v1_renders_and_sweeps_under_todays_names():
    service = WebService()
    render = service.jobs.create("acoustic-world", kind="world_render")
    service.jobs.start(render.job_id, phase="preparing", progress=0)
    service.jobs.succeed(render.job_id, _v1_report())
    report = service.job(render.job_id)["result"]
    assert [s["role"] for s in report["scene"]["sources"]] == ["target", "interferer", "noise"]
    assert set(report["metrics"]) == {"target", "near"}
    assert set(report["output_urls"]) == {"input", "reference-target", "reference-near", "source-noise"}
    assert report["output_urls"]["reference-target"] == "/b"

    sweep = service.jobs.create("acoustic-world", kind="world_sweep")
    service.jobs.start(sweep.job_id, phase="sweep", progress=0)
    service.jobs.succeed(sweep.job_id, {"axes": [], "cells": [{"index": 0, "result": _v1_report()}, {"index": 1, "error": "x"}]})
    cells = service.job(sweep.job_id)["result"]["cells"]
    assert set(cells[0]["result"]["metrics"]) == {"target", "near"}
    assert cells[1] == {"index": 1, "error": "x"}
    listed = {job["job_id"]: job for job in service.list_jobs()}
    assert set(listed[render.job_id]["result"]["metrics"]) == {"target", "near"}
    assert set(listed[sweep.job_id]["result"]["cells"][0]["result"]["metrics"]) == {"target", "near"}


def test_target_distance_change_never_carries_a_talker_past_the_microphone():
    scene = world_preset("turn").to_dict()
    cells = sweep_scenes({
        "scene": scene,
        "axes": [
            {"parameter": "target_distance_change_m", "values": [-0.5, -1.5]},
            {"parameter": "noise_gain_change_db", "values": [0]},
        ],
    })
    assert "scene" in cells[0]
    assert "past the microphone" in cells[1]["error"]


def test_a_render_records_the_audio_behind_each_asset():
    service = WebService()
    payload = {**request(), "assets": {"speaker-a": {"sample": "speaker-a-2"}}}
    job = wait(service, service.submit_world(payload))
    assert job["status"] == "succeeded", job["error"]
    assert job["result"]["assets"] == {
        "speaker-a": {"sample": "speaker-a-2"},
        "speaker-b": {"sample": "speaker-b"},
        "fan": {"sample": "fan"},
    }


@pytest.mark.parametrize(
    "path, value",
    [(("room",), []), (("room", "materials"), 3), (("room", "environment"), "x")],
)
@pytest.mark.parametrize("route", ["validate_world", "world_materials"])
def test_a_scene_of_the_wrong_shape_is_the_requests_error(path, value, route):
    payload = request()
    holder = payload["scene"]
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    with pytest.raises(WebServiceError, match="malformed scene"):
        getattr(WebService(), route)(payload)
