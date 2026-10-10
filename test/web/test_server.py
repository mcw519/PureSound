"""The web playground's service API: catalog, playground inference, background
jobs, model comparison, live streaming, history, exports and word checks, plus
the packaged static page. Runtimes are faked; what is under test is the
service's own contract."""

from __future__ import annotations

import base64
import io
import json
import time
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from puresound.inference import InferenceCancelled, InferenceResult
from puresound.web import WebService
import puresound.web.server as web_server
from puresound.web.server import WebServiceError
from puresound.web.measurements import align_streaming_output


def _wav_data_url(samples: int = 160) -> str:
    payload = io.BytesIO()
    with wave.open(payload, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * samples)
    return "data:audio/wav;base64," + base64.b64encode(payload.getvalue()).decode("ascii")


def test_web_catalog_payload_is_path_stable_default_first_and_lists_the_onset_guard():
    service = WebService()
    models = service.list_models()

    assert models
    assert models[0]["id"] == "voice-isolate-dpcrn-curriculum-v1"
    assert all(not str(item["source_checkpoint"]).startswith("/") for item in models if item["source_checkpoint"])
    inspected = service.inspect_model("voice-isolate-dpcrn-v8")
    artifact = inspected["artifacts"][0]
    assert artifact["filename"] == "dpcrn_v8.onnx"
    assert artifact["available"] is True

    parameters = inspected["parameters"]
    assert parameters["onset_guard"]["type"] == "bool"
    assert parameters["onset_guard"]["default"] is False
    assert parameters["onset_guard_t_arm_s"] == {
        "type": "float",
        "default": 1.0,
        "minimum": 0.3,
        "maximum": 5.0,
        "choices": None,
        "description": parameters["onset_guard_t_arm_s"]["description"],
    }
    assert parameters["onset_guard_t_forget_s"]["maximum"] == 60.0
    assert parameters["onset_guard_tau_dn_s"]["minimum"] == 0.1
    assert parameters["onset_guard_margin_db"]["minimum"] == 3.0


def test_web_infer_materializes_upload_and_exposes_download(monkeypatch):
    service = WebService()

    class FakeRuntime:
        def infer(
            self,
            *,
            inputs,
            parameters,
            progress_callback=None,
            cancel_check=None,
        ):
            assert set(inputs) == {"audio"}
            assert parameters == {"dry_blend": 0.8}
            assert cancel_check is None
            progress_callback(1.0, "complete")
            return InferenceResult(
                model_id="voice-isolate-dpcrn-v8",
                task="voice_isolation",
                outputs={"audio": np.zeros(160, dtype=np.float32)},
                provider="CPUExecutionProvider",
                elapsed_seconds=0.01,
                rtf=0.1,
                sample_rate=16_000,
                metadata={"latency_ms": 10.0},
            )

    monkeypatch.setattr(service, "_runtime", lambda *args: FakeRuntime())
    response = service.infer(
        {
            "model_id": "voice-isolate-dpcrn-v8",
            "provider": "cpu",
            "inputs": {"audio": {"filename": "input.wav", "data": _wav_data_url()}},
            "parameters": {"dry_blend": 0.8},
        }
    )

    assert response["output_urls"]["audio"].startswith("/api/runs/")
    token = response["run_id"]
    stored = service.runs.get(token, "audio")
    assert stored is not None
    assert stored.content_type == "audio/wav"
    assert stored.data[:4] == b"RIFF"


def _guard_request():
    return {
        "onset_guard": True,
        "onset_guard_t_arm_s": 1.5,
        "onset_guard_t_forget_s": 8.0,
        "onset_guard_tau_dn_s": 1.0,
        "onset_guard_margin_db": 10.0,
    }


def _guard_runtime(seen):
    class FakeRuntime:
        def infer(self, *, inputs, parameters, progress_callback=None, cancel_check=None):
            seen.append(dict(parameters))
            if progress_callback is not None:
                progress_callback(1.0, "complete")
            return InferenceResult(
                model_id="voice-isolate-dpcrn-v8",
                task="voice_isolation",
                outputs={"audio": np.ones(160, dtype=np.float32) * 0.1},
                provider="CPUExecutionProvider",
                elapsed_seconds=0.01,
                rtf=0.1,
                sample_rate=16_000,
                metadata={"onset_guard": {"t_arm_s": 1.5, "tau_dn_s": 1.0}},
            )

    return FakeRuntime()


def _wait_for(service, job_id, seconds=5.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        current = service.job(job_id)
        if current["status"] in {"succeeded", "failed", "cancelled"}:
            return current
        time.sleep(0.01)
    return service.job(job_id)


def test_web_forwards_the_onset_guard_parameters_through_every_entry_point(monkeypatch):
    """Playground, background job and comparison all hand the knobs to the
    runtime unchanged and report the operating point that actually ran."""
    service = WebService()
    seen: list[dict] = []
    monkeypatch.setattr(service, "_runtime", lambda *args: _guard_runtime(seen))
    audio = {"audio": {"filename": "input.wav", "data": _wav_data_url()}}

    response = service.infer(
        {"model_id": "voice-isolate-dpcrn-v8", "inputs": audio, "parameters": _guard_request()}
    )
    assert seen == [_guard_request()]
    assert response["metadata"]["onset_guard"] == {"t_arm_s": 1.5, "tau_dn_s": 1.0}

    job = service.submit_inference(
        {"model_id": "voice-isolate-dpcrn-v8", "inputs": audio, "parameters": {"onset_guard": False}}
    )
    assert _wait_for(service, job["job_id"])["status"] == "succeeded"
    assert seen[1:] == [{"onset_guard": False}]

    report = service.measure(
        {
            "inputs": audio,
            "models": ["voice-isolate-dpcrn-v8", "voice-isolate-dpcrn-curriculum-v1"],
            "parameters": _guard_request(),
        }
    )
    assert seen[2:] == [_guard_request(), _guard_request()]
    assert report["request"]["parameters"] == _guard_request()
    assert report["models"][0]["onset_guard"] == {"t_arm_s": 1.5, "tau_dn_s": 1.0}


def _audio_request(**extra):
    return {"inputs": {"audio": {"filename": "input.wav", "data": _wav_data_url()}}, **extra}


_V8 = "voice-isolate-dpcrn-v8"


@pytest.mark.parametrize(
    "method, payload, message",
    [
        pytest.param("infer", _audio_request(model_id=_V8, provider="cpu", parameters={"onset_guard": True, "onset_guard_t_arm_s": 12.0}), "onset_guard", id="guard-out-of-range"),
        pytest.param("infer", _audio_request(model_id=_V8, provider="cpu", parameters={"onset_guard": True, "onset_guard_margin_db": "loud"}), "onset_guard", id="guard-not-a-number"),
        pytest.param("infer", _audio_request(model_id=_V8, provider="cpu", parameters={"onset_guard": "on"}), "onset_guard", id="guard-not-a-bool"),
        pytest.param("infer", {"model_id": _V8, "inputs": {"audio": "/tmp/input.wav"}}, "local input paths are disabled", id="local-path"),
        pytest.param("infer", _audio_request(model_id=_V8, stages="nope"), "stages must be an array", id="stages-not-a-list"),
        pytest.param("infer", _audio_request(model_id=_V8, stages=[{"model_id": "speaker-verification-ps-spk-v1"}]), "only audio-in/audio-out", id="stage-not-enhancement"),
        pytest.param("infer", _audio_request(model_id=_V8, stages=[{"model_id": "no-such-model"}]), "unknown model", id="stage-unknown"),
        pytest.param("infer", _audio_request(model_id=_V8, stages=[{"model_id": "noise-suppression-dpcrn-mamba-v2"}] * 4), "at most", id="too-many-stages"),
        *[
            pytest.param("measure", _audio_request(models=[_V8], measurements={"timing_repeats": repeats}), "timing_repeats", id=f"timing-repeats-{repeats}")
            for repeats in (0, 6, "many")
        ],
        *[
            pytest.param("measure", _audio_request(candidates=candidates), "candidates", id=f"candidates-{index}")
            for index, candidates in enumerate([[], [{"parameters": {}}], [{"model_id": _V8, "parameters": 3}]])
        ],
        pytest.param("submit_measurement", {"inputs": {}}, "inputs.audio or inputs.clips", id="measurement-without-input"),
        pytest.param("transcribe", {"backend": "nope", "tracks": [{"id": "a", "source": {}}]}, "backend must be one of", id="asr-backend"),
        pytest.param("transcribe", {"backend": "whisper", "tracks": []}, "tracks must be a non-empty array", id="asr-no-tracks"),
        pytest.param("transcribe", {"backend": "whisper", "tracks": [{"id": "a", "source": {}}], "reference_track": "b"}, "reference_track", id="asr-reference-track"),
    ],
)
def test_web_rejects_a_malformed_request(method, payload, message):
    with pytest.raises(web_server.WebServiceError, match=message):
        getattr(WebService(), method)(payload)


def test_web_service_passes_bind_address_to_http_server(monkeypatch):
    received = {}

    class FakeHttpServer:
        def __init__(self, address, handler):
            received["address"] = address
            received["handler"] = handler

    monkeypatch.setattr(web_server, "_Server", FakeHttpServer)
    server = WebService().create_server("192.0.2.10", 8123)

    assert received["address"] == ("192.0.2.10", 8123)
    assert server is not None


def test_web_measurement_report_compares_selected_models(monkeypatch):
    # The comparison is over audio-in/audio-out models, not one task: a
    # noise-suppression model sits beside the voice-isolation one.
    service = WebService()

    monkeypatch.setattr(
        web_server,
        "reference_free_metrics",
        lambda samples, sample_rate, *, include_dnsmos: {
            "dnsmos": {"dnsmos_ovr": 3.5}
        }
        if include_dnsmos
        else {"dnsmos": {}},
    )

    class FakeRuntime:
        def infer(
            self,
            *,
            inputs,
            parameters,
            progress_callback=None,
            cancel_check=None,
        ):
            assert set(inputs) == {"audio"}
            assert parameters == {"dry_blend": 0.8}
            progress_callback(1.0, "complete")
            return InferenceResult(
                model_id="voice-isolate-dpcrn-v8",
                task="voice_isolation",
                outputs={"audio": np.ones(160, dtype=np.float32) * 0.1},
                provider="CPUExecutionProvider",
                elapsed_seconds=0.01,
                rtf=0.1,
                sample_rate=16_000,
            )

    monkeypatch.setattr(service, "_runtime", lambda *args: FakeRuntime())
    report = service.measure(
        {
            "inputs": {"audio": {"filename": "input.wav", "data": _wav_data_url()}},
            "models": ["voice-isolate-dpcrn-v8", "noise-suppression-dpcrn-mamba-v1"],
            "provider": "cpu",
            "parameters": {"dry_blend": 0.8},
            "measurements": {"include_dnsmos": True},
        }
    )

    assert report["input"]["sample_rate"] == 16_000
    assert report["request"]["models"] == ["voice-isolate-dpcrn-v8", "noise-suppression-dpcrn-mamba-v1"]
    assert report["request"]["provider"] == "cpu"
    assert [m["model_id"] for m in report["models"]] == [
        "voice-isolate-dpcrn-v8",
        "noise-suppression-dpcrn-mamba-v1",
    ]
    assert all("error" not in m for m in report["models"])
    assert report["models"][0]["output"]["rms_dbfs"] < 0
    assert report["models"][0]["quality"] == {}
    assert report["models"][0]["reference_free"]["dnsmos"]["dnsmos_ovr"] == 3.5
    assert report["models"][0]["output_url"].startswith("/api/runs/")
    assert report["summary"] == {"succeeded": 2, "failed": 0}
    assert report["request"]["measurements"] == {"include_dnsmos": True, "timing_repeats": 1}


def test_streaming_output_alignment_removes_latency_and_bounds_duration():
    output = np.concatenate([np.zeros(3), np.arange(8, dtype=np.float32)])

    aligned = align_streaming_output(output, latency_samples=3, target_samples=5)

    np.testing.assert_array_equal(aligned, np.arange(5, dtype=np.float32))


def test_web_measurement_fails_when_every_model_fails(monkeypatch):
    service = WebService()

    class BrokenRuntime:
        def infer(self, **kwargs):
            raise RuntimeError("broken graph")

    monkeypatch.setattr(service, "_runtime", lambda *args: BrokenRuntime())

    with pytest.raises(web_server.WebServiceError, match="all selected models failed"):
        service.measure(
            {
                "inputs": {"audio": {"filename": "input.wav", "data": _wav_data_url()}},
                "models": ["voice-isolate-dpcrn-v8"],
            }
        )


@pytest.mark.parametrize("kind", ["inference", "measurement"])
def test_web_background_job_reports_completion_and_keeps_result(monkeypatch, kind):
    service = WebService()

    class FakeRuntime:
        def infer(self, *, inputs, parameters, progress_callback=None, cancel_check=None):
            progress_callback(0.5, "processing_frames")
            return InferenceResult(
                model_id="voice-isolate-dpcrn-v8",
                task="voice_isolation",
                outputs={"audio": np.ones(160, dtype=np.float32) * 0.1},
                provider="CPUExecutionProvider",
                elapsed_seconds=0.01,
                rtf=0.1,
                sample_rate=16_000,
            )

    monkeypatch.setattr(service, "_runtime", lambda *args: FakeRuntime())
    inputs = {"audio": {"filename": "input.wav", "data": _wav_data_url()}}
    if kind == "inference":
        job = service.submit_inference({"model_id": "voice-isolate-dpcrn-v8", "provider": "cpu", "inputs": inputs})
    else:
        job = service.submit_measurement({"inputs": inputs, "models": ["voice-isolate-dpcrn-v8"]})
    current = _wait_for(service, job["job_id"])

    assert current["status"] == "succeeded"
    assert current["kind"] == kind
    assert current["progress"] == 1.0
    if kind == "inference":
        assert current["result"]["output_urls"]["audio"].startswith("/api/runs/")
    else:
        assert current["result"]["models"][0]["model_id"] == "voice-isolate-dpcrn-v8"


def test_web_async_job_cancels_during_processor_progress(monkeypatch):
    service = WebService()

    class FakeRuntime:
        def infer(
            self,
            *,
            inputs,
            parameters,
            progress_callback=None,
            cancel_check=None,
        ):
            for step in range(1, 101):
                if cancel_check():
                    raise InferenceCancelled("inference cancelled")
                progress_callback(step / 100, "processing_frames")
                time.sleep(0.002)
            raise AssertionError("job should have been cancelled")

    monkeypatch.setattr(service, "_runtime", lambda *args: FakeRuntime())
    job = service.submit_inference(
        {
            "model_id": "voice-isolate-dpcrn-v8",
            "inputs": {"audio": {"filename": "input.wav", "data": _wav_data_url()}},
        }
    )
    deadline = time.time() + 5
    while time.time() < deadline and service.job(job["job_id"])["progress"] <= 0.12:
        time.sleep(0.002)
    service.cancel_job(job["job_id"])
    while time.time() < deadline:
        current = service.job(job["job_id"])
        if current["status"] == "cancelled":
            break
        time.sleep(0.002)

    assert current["status"] == "cancelled"
    assert current["progress"] < 1.0
    assert current["result"] is None


def _read_wav(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0


def test_web_infer_aligns_enhancement_output_to_the_input(monkeypatch):
    service = WebService()
    latency = 40

    class DelayedRuntime:
        def infer(self, *, inputs, parameters, progress_callback=None, cancel_check=None):
            # A streaming model: `latency` samples late, plus a flush tail.
            output = np.concatenate([np.zeros(latency), np.full(160, 0.25), np.zeros(30)])
            return InferenceResult(
                model_id="voice-isolate-dpcrn-v8",
                task="voice_isolation",
                outputs={"audio": output.astype(np.float32)},
                provider="CPUExecutionProvider",
                elapsed_seconds=0.01,
                rtf=0.1,
                sample_rate=16_000,
                metadata={"latency_samples": latency},
            )

    monkeypatch.setattr(service, "_runtime", lambda *args: DelayedRuntime())
    response = service.infer(
        {
            "model_id": "voice-isolate-dpcrn-v8",
            "inputs": {"audio": {"filename": "input.wav", "data": _wav_data_url(160)}},
        }
    )

    assert set(response["output_urls"]) == {"audio", "aligned", "input", "removed"}
    assert response["alignment"]["latency_samples"] == latency
    token = response["run_id"]
    raw = _read_wav(service.runs.get(token, "audio").data)
    aligned = _read_wav(service.runs.get(token, "aligned").data)
    source = _read_wav(service.runs.get(token, "input").data)
    removed = _read_wav(service.runs.get(token, "removed").data)
    assert raw.size == latency + 160 + 30
    assert aligned.size == source.size == removed.size == 160
    np.testing.assert_allclose(aligned, 0.25, atol=1e-4)
    # The input is silence, so what the model "removed" is minus its output.
    np.testing.assert_allclose(removed, -0.25, atol=1e-4)
    assert response["host"]["cpu_count"] >= 1


def test_web_measure_reports_the_median_of_timing_repeats(monkeypatch):
    service = WebService()
    rtfs = iter([0.9, 0.3, 0.5])

    class TimedRuntime:
        def infer(self, *, inputs, parameters, progress_callback=None, cancel_check=None):
            return InferenceResult(
                model_id="voice-isolate-dpcrn-v8",
                task="voice_isolation",
                outputs={"audio": np.full(160, 0.1, dtype=np.float32)},
                provider="CPUExecutionProvider",
                elapsed_seconds=0.01,
                rtf=next(rtfs),
                sample_rate=16_000,
            )

    monkeypatch.setattr(service, "_runtime", lambda *args: TimedRuntime())
    report = service.measure(
        {
            "inputs": {"audio": {"filename": "input.wav", "data": _wav_data_url()}},
            "models": ["voice-isolate-dpcrn-v8"],
            "measurements": {"timing_repeats": 3},
        }
    )

    model = report["models"][0]
    assert model["rtf_runs"] == [0.9, 0.3, 0.5]
    assert model["rtf"] == pytest.approx(0.5)
    assert report["request"]["measurements"]["timing_repeats"] == 3


def test_web_runtime_warms_up_enhancement_models_once(monkeypatch):
    service = WebService()
    calls = []

    class Model:
        task = "voice_isolation"

    class FakeRuntime:
        model = Model()
        sample_rate = 16_000

        def infer(self, *, inputs, parameters):
            calls.append(np.asarray(inputs["audio"]).size)

    monkeypatch.setattr(web_server, "load_model", lambda *args, **kwargs: FakeRuntime())
    first = service._runtime("voice-isolate-dpcrn-v8", "cpu", None)
    second = service._runtime("voice-isolate-dpcrn-v8", "cpu", None)

    assert first is second
    assert calls == [8_000]


def _constant_runtime(value: float = 0.1, seen: list | None = None):
    class ConstantRuntime:
        def infer(self, *, inputs, parameters, progress_callback=None, cancel_check=None):
            if seen is not None:
                seen.append(dict(parameters))
            return InferenceResult(
                model_id="voice-isolate-dpcrn-v8",
                task="voice_isolation",
                outputs={"audio": np.full(160, value, dtype=np.float32)},
                provider="CPUExecutionProvider",
                elapsed_seconds=0.01,
                rtf=0.1,
                sample_rate=16_000,
                metadata={"dry_blend": parameters.get("dry_blend", 0.9)},
            )

    return ConstantRuntime()


def test_web_measure_scores_the_unprocessed_input_as_a_baseline(monkeypatch):
    service = WebService()
    monkeypatch.setattr(service, "_runtime", lambda *args: _constant_runtime())
    scored = []
    monkeypatch.setattr(
        web_server,
        "reference_free_metrics",
        lambda samples, sample_rate, *, include_dnsmos: scored.append(np.asarray(samples).copy()) or {"dnsmos": {"dnsmos_ovr": 2.0 + len(scored)}},
    )
    report = service.measure(
        {
            "inputs": {
                "audio": {"filename": "noisy.wav", "data": _wav_data_url()},
                "reference": {"filename": "clean.wav", "data": _wav_data_url()},
            },
            "models": ["voice-isolate-dpcrn-v8"],
            "measurements": {"include_dnsmos": True},
        }
    )

    baseline = report["baseline"]
    assert baseline["label"] == "Unprocessed input"
    # The input is scored first, by the same scorer as every model.
    np.testing.assert_allclose(scored[0], 0.0)
    assert baseline["reference_free"]["dnsmos"]["dnsmos_ovr"] == 3.0
    assert report["models"][0]["reference_free"]["dnsmos"]["dnsmos_ovr"] == 4.0
    assert "si_sdr_db" in baseline["quality"]
    token = baseline["output_url"].split("/")[3]
    assert service.runs.get(token, "input").data[:4] == b"RIFF"
    assert report["reference_url"].endswith("/reference")
    assert report["input_files"] == {"audio": "noisy.wav", "reference": "clean.wav"}


def test_web_measure_runs_each_candidate_with_its_own_parameters(monkeypatch):
    service = WebService()
    seen: list = []
    monkeypatch.setattr(service, "_runtime", lambda *args: _constant_runtime(seen=seen))
    report = service.measure(
        {
            "inputs": {"audio": {"filename": "input.wav", "data": _wav_data_url()}},
            "parameters": {"onset_guard": False},
            "candidates": [
                {"model_id": "voice-isolate-dpcrn-v8", "label": "v8 release"},
                {"model_id": "voice-isolate-dpcrn-v8", "parameters": {"dry_blend": 1.0, "onset_guard": True}, "label": "v8 · blend 1.00 · guard"},
            ],
        }
    )

    assert seen == [{"onset_guard": False}, {"onset_guard": True, "dry_blend": 1.0}]
    assert [model["label"] for model in report["models"]] == ["v8 release", "v8 · blend 1.00 · guard"]
    assert [model["candidate_id"] for model in report["models"]] == [0, 1]
    assert report["models"][1]["dry_blend"] == 1.0
    assert report["request"]["models"] == ["voice-isolate-dpcrn-v8", "voice-isolate-dpcrn-v8"]


def test_web_infer_attaches_measurements_and_head_curves_on_request(monkeypatch):
    service = WebService()

    class HeadRuntime:
        def infer(self, *, inputs, parameters, progress_callback=None, cancel_check=None):
            return InferenceResult(
                model_id="voice-isolate-dpcrn-v8",
                task="voice_isolation",
                outputs={"audio": np.ones(160, dtype=np.float32) * 0.1, "vad": np.linspace(0, 1, 10, dtype=np.float32)[:, None]},
                provider="CPUExecutionProvider",
                elapsed_seconds=0.01,
                rtf=0.1,
                sample_rate=16_000,
                metadata={"hop_length": 16},
            )

    monkeypatch.setattr(service, "_runtime", lambda *args: HeadRuntime())
    monkeypatch.setattr(
        web_server,
        "reference_free_metrics",
        lambda samples, sample_rate, *, include_dnsmos: {
            "dnsmos": {"dnsmos_ovr": 4.2}
        }
        if include_dnsmos
        else {"dnsmos": {}},
    )
    response = service.infer(
        {
            "model_id": "voice-isolate-dpcrn-v8",
            "inputs": {"audio": {"filename": "input.wav", "data": _wav_data_url()}},
            "parameters": {"collect_extras": True},
            "measurements": {"include_dnsmos": True},
        }
    )

    assert response["measurements"]["output"]["sample_rate"] == 16_000
    assert response["measurements"]["reference_free"]["dnsmos"]["dnsmos_ovr"] == 4.2
    curve = response["extras"]["vad"]
    assert curve["frames"] == 10
    assert curve["hop_seconds"] == pytest.approx(0.001)
    assert curve["series"][0][0] == 0.0 and curve["series"][0][-1] == 1.0
    assert curve["range"] == [0.0, 1.0]
    assert response["input_files"] == {"audio": "input.wav"}


def test_web_model_benchmarks_quote_the_rows_naming_the_checkpoint():
    text = "\n".join([
        "# Title",
        "## Versions",
        "",
        "| version | result |",
        "|---|---|",
        "| `model_v1.ckpt` | first |",
        "| **`model_v1-1.ckpt`** | second, [link](x.md)<br>more |",
        "",
        "| other | table |",
        "|---|---|",
        "| model_v1 | stem match |",
    ])
    rows = web_server._markdown_rows_naming(text, {"model_v1-1.ckpt", "model_v1"})
    assert [row["cells"][1][1] for row in rows] == ["second, link / more", "stem match"]
    assert rows[0]["section"] == "Versions"
    assert rows[0]["cells"][0] == ["version", "**`model_v1-1.ckpt`**"]

    # The catalog's benchmark references are read through it.
    service = WebService()
    released = service.model_benchmarks("noise-suppression-dpcrn-mamba-v2")
    kinds = {reference["kind"] for reference in released["references"]}
    assert kinds == {"markdown", "record"}
    markdown = next(reference for reference in released["references"] if reference["kind"] == "markdown")
    assert markdown["rows"] and "dpcrn_mamba_v2.ckpt" in markdown["rows"][0]["cells"][0][1]
    record = next(reference for reference in released["references"] if reference["kind"] == "record")["record"]
    assert record["stages"] and {"name", "value", "verdict", "difference"} <= set(record["stages"][0])
    voice = service.model_benchmarks("voice-isolate-dpcrn-v8")
    assert voice["checkpoint"] == "dpcrn_v8.ckpt"
    assert len(voice["references"][0]["rows"]) == 1


def test_web_history_survives_a_restart(tmp_path, monkeypatch):
    first = WebService(history_dir=tmp_path)
    token = first.runs.put({"aligned": web_server.StoredOutput(b"RIFFdata", "audio/wav", "enhanced.wav", 0.0)})
    job = first.jobs.create("voice-isolate-dpcrn-v8")
    first.jobs.start(job.job_id, phase="running", progress=0.1)
    first.jobs.succeed(job.job_id, {"output_urls": {"aligned": f"/api/runs/{token}/aligned"}})
    running = first.jobs.create("voice-isolate-dpcrn-v8")

    second = WebService(history_dir=tmp_path)
    stored = second.runs.get(token, "aligned")
    assert stored is not None and stored.data == b"RIFFdata" and stored.filename == "enhanced.wav"
    jobs = {snapshot["job_id"]: snapshot for snapshot in second.list_jobs()}
    assert jobs[job.job_id]["status"] == "succeeded"
    assert jobs[job.job_id]["result"]["output_urls"]["aligned"].endswith("/aligned")
    # A job that never finished has nothing to show and is not restored.
    assert running.job_id not in jobs


def test_web_stores_keep_their_bound(tmp_path):
    """Run history on disk and uploads both evict the oldest entry first."""
    store = web_server.RunStore(max_runs=2, directory=tmp_path)
    tokens = []
    for index in range(3):
        tokens.append(store.put({"audio": web_server.StoredOutput(bytes([index]), "audio/wav", "a.wav", 0.0)}))
        time.sleep(0.01)
    assert store.get(tokens[0], "audio") is None
    assert store.get(tokens[2], "audio").data == bytes([2])
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(tokens[1:])

    uploads = web_server.UploadStore(max_files=2)
    first = uploads.put(b"a", "a.wav")["upload_id"]
    time.sleep(0.01)
    second = uploads.put(b"b", "b.wav")["upload_id"]
    time.sleep(0.01)
    third = uploads.put(b"c", "c.wav")["upload_id"]
    assert uploads.get(first) is None
    assert uploads.get(second)[0].read_bytes() == b"b" and uploads.get(third)[0].read_bytes() == b"c"


def test_web_upload_with_an_extension_too_long_for_the_file_system_is_still_stored():
    uploads = web_server.UploadStore()
    stored = uploads.put(b"RIFF", "clip." + "b" * 300)
    path, name = uploads.get(stored["upload_id"])
    assert path.read_bytes() == b"RIFF" and path.suffix == ".wav" and name.startswith("clip.")


def test_web_upload_cut_short_is_refused_not_stored():
    import socket

    service = WebService()
    server = _serve(service)
    try:
        with socket.create_connection(("127.0.0.1", server.server_port), timeout=5) as connection:
            connection.sendall(b"POST /api/uploads HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Length: 100\r\n\r\n" + b"x" * 10)
            connection.shutdown(socket.SHUT_WR)
            reply = b""
            while chunk := connection.recv(65536):
                reply += chunk
    finally:
        server.shutdown()
        server.server_close()
    assert reply.startswith(b"HTTP/1.0 400 ") and not list(service.uploads.directory.iterdir())


def _silent_flac(seconds: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(16_000 * seconds, dtype="float32"), 16_000, format="FLAC")
    return buffer.getvalue()


def test_web_audio_that_decodes_to_far_more_than_the_upload_limit_is_refused():
    import http.client

    # Ten minutes of silence is a few KiB as FLAC and 19 MiB as samples.
    flac = _silent_flac(600)
    assert len(flac) < 1024 * 1024
    service = WebService(max_upload_bytes=1024 * 1024)
    server = _serve(service)
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        connection.request("POST", "/api/uploads", body=flac, headers={"X-Filename": "silence.flac"})
        response = connection.getresponse()
        assert response.status == 413 and b"decodes to" in response.read()
        connection.request("POST", "/api/uploads", body=_silent_flac(5), headers={"X-Filename": "short.flac"})
        response = connection.getresponse()
        assert response.status == 201
    finally:
        server.shutdown()
        server.server_close()
    assert len(list(service.uploads.directory.iterdir())) == 1
    payload = {"model_id": "voice-isolate-dpcrn-v8", "inputs": {"audio": {"filename": "silence.flac", "data": base64.b64encode(flac).decode("ascii")}}}
    with pytest.raises(WebServiceError, match="decodes to"):
        service.infer(payload)


def _serve(service):
    import threading

    server = service.create_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _status(server, method, path, headers=None, body=None):
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        response.read()
        return response.status
    finally:
        connection.close()


@pytest.mark.parametrize(
    "host, status",
    [
        (None, 200),
        ("localhost:7860", 200),
        ("[::1]:7860", 200),
        ("localhost.:7860", 200),
        ("192.168.1.20:7860", 200),
        ("evil.example", 403),
        ("evil.example:7860", 403),
        ("127.0.0.1.evil.example", 403),
    ],
)
def test_web_loopback_server_refuses_a_host_name_that_is_not_its_own(host, status):
    # A page on another site reaches a loopback server by rebinding its own
    # name to 127.0.0.1; the request then names that site, not the server.
    server = _serve(WebService())
    try:
        assert _status(server, "GET", "/api/health", {"Host": host} if host else None) == status
    finally:
        server.shutdown()
        server.server_close()


def test_web_allowed_hosts_name_what_else_a_server_may_be_reached_by(monkeypatch):
    monkeypatch.setenv("PURESOUND_ALLOWED_HOSTS", "from-env.example, other.example:8080")
    server = _serve(WebService(allowed_hosts=["proxy.example"]))
    try:
        for host in ("proxy.example", "from-env.example", "other.example"):
            assert _status(server, "GET", "/api/health", {"Host": host}) == 200, host
        assert _status(server, "GET", "/api/health", {"Host": "evil.example"}) == 403
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("bind, restricted", [("127.0.0.1", True), ("localhost", True), ("::1", True), ("0.0.0.0", False), ("192.0.2.10", False)])
def test_web_only_a_server_bound_to_loopback_restricts_the_host_names_it_answers_to(monkeypatch, bind, restricted):
    class FakeHttpServer:
        def __init__(self, address, handler):
            self.handler = handler

    monkeypatch.setattr(web_server, "_Server", FakeHttpServer)
    allowed = WebService().create_server(bind, 0).handler.allowed_hosts

    assert (allowed is not None) is restricted
    assert not restricted or {"localhost", "127.0.0.1"} <= allowed


@pytest.mark.parametrize(
    "origin, status",
    [
        (None, 202),
        ("http://127.0.0.1:{port}", 202),
        ("http://evil.example", 403),
        ("http://localhost:{port}", 403),
        ("null", 403),
    ],
)
def test_web_writes_come_only_from_the_servers_own_pages(origin, status):
    server = _serve(WebService())
    try:
        headers = {"Content-Type": "text/plain"}
        if origin:
            headers["Origin"] = origin.format(port=server.server_port)
        body = json.dumps({"model_id": "no-such-model", "inputs": {"audio": {"data": "x"}}})
        assert _status(server, "POST", "/api/jobs", headers, body) == status
        upgrade = {**headers, "Upgrade": "websocket", "Connection": "Upgrade", "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ=="}
        if status == 403:
            assert _status(server, "GET", "/api/live", upgrade) == 403
        # Reading is not a write: another origin's GET is answered (the browser
        # keeps its response from that page).
        assert _status(server, "GET", "/api/health", {"Origin": "http://evil.example"}) == 200
    finally:
        server.shutdown()
        server.server_close()


def test_web_proxy_origin_is_accepted_once_trusted():
    server = _serve(WebService(allowed_hosts=["proxy.example"]))
    try:
        body = json.dumps({"model_id": "no-such-model", "inputs": {"audio": {"data": "x"}}})
        headers = {"Origin": "https://proxy.example", "Host": "127.0.0.1:1"}
        assert _status(server, "POST", "/api/jobs", headers, body) == 202
        assert _status(server, "POST", "/api/jobs", {**headers, "Origin": "https://evil.example"}, body) == 403
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("change", [{"sample_rate": "abc"}, {"sample_rate": [16_000]}])
def test_web_measure_route_answers_a_request_it_cannot_use_with_422(change):
    server = _serve(WebService())
    try:
        body = json.dumps({"inputs": {"audio": {"data": "x"}}, **change})
        assert _status(server, "POST", "/api/measure", {"Content-Type": "application/json"}, body) == 422
    finally:
        server.shutdown()
        server.server_close()


def test_web_uploads_are_sent_once_and_referenced_by_id(monkeypatch):
    import http.client

    service = WebService()
    seen = []

    class PathRuntime:
        def infer(self, *, inputs, parameters, progress_callback=None, cancel_check=None):
            seen.append(inputs["audio"])
            return InferenceResult(
                model_id="voice-isolate-dpcrn-v8", task="voice_isolation",
                outputs={"audio": np.zeros(160, dtype=np.float32)}, sample_rate=16_000,
            )

    monkeypatch.setattr(service, "_runtime", lambda *args: PathRuntime())
    server = _serve(service)
    try:
        body = base64.b64decode(_wav_data_url().split(",", 1)[1])
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.request("POST", "/api/uploads", body=body, headers={"X-Filename": "my%20clip.wav", "Content-Type": "audio/wav"})
        response = connection.getresponse()
        uploaded = __import__("json").loads(response.read())
        assert response.status == 201 and uploaded["filename"] == "my_clip.wav" and uploaded["bytes"] == len(body)
        connection.request("GET", f"/api/uploads/{uploaded['upload_id']}")
        assert connection.getresponse().read() and True
    finally:
        server.shutdown()
        server.server_close()
    for _ in range(2):
        service.infer({"model_id": "voice-isolate-dpcrn-v8", "inputs": {"audio": {"upload_id": uploaded["upload_id"], "filename": "my clip.wav"}}})
    # Both runs read the one stored file, and it outlives them.
    assert len(seen) == 2 and seen[0] == seen[1] and seen[0].is_file()
    with pytest.raises(web_server.WebServiceError, match="upload not found"):
        service.infer({"model_id": "voice-isolate-dpcrn-v8", "inputs": {"audio": {"upload_id": "0" * 32}}})


class _ChainRuntime:
    """Scales its input: noise suppression by 0.5, voice isolation by 3."""

    def __init__(self, model_id, gain, latency):
        self.model_id, self.gain, self.latency = model_id, gain, latency
        self.sample_rate = 16_000
        self.inputs = []

    def infer(self, *, inputs, parameters, progress_callback=None, cancel_check=None):
        from puresound.inference.processors.base import load_audio

        audio, _ = load_audio(inputs["audio"], sample_rate=16_000)
        self.inputs.append(audio.copy())
        output = np.concatenate([np.zeros(self.latency), audio * self.gain, np.zeros(10)])
        return InferenceResult(
            model_id=self.model_id, task="voice_isolation",
            outputs={"audio": output.astype(np.float32)}, sample_rate=16_000,
            elapsed_seconds=0.01, rtf=0.1, metadata={"latency_samples": self.latency, "latency_ms": self.latency / 16.0, "duration_seconds": audio.size / 16_000},
        )


def _tone_url(samples: int = 160, value: float = 0.2) -> str:
    payload = io.BytesIO()
    with wave.open(payload, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes((np.full(samples, value) * 32767).astype("<i2").tobytes())
    return "data:audio/wav;base64," + base64.b64encode(payload.getvalue()).decode("ascii")


def test_web_infer_runs_stages_first_on_the_aligned_signal(monkeypatch):
    service = WebService()
    runtimes = {
        "noise-suppression-dpcrn-mamba-v2": _ChainRuntime("ns", 0.5, latency=32),
        "voice-isolate-dpcrn-v8": _ChainRuntime("vi", 3.0, latency=48),
    }
    monkeypatch.setattr(service, "_runtime", lambda model_id, *args: runtimes[model_id])
    response = service.infer({
        "model_id": "voice-isolate-dpcrn-v8",
        "inputs": {"audio": {"filename": "input.wav", "data": _tone_url()}},
        "stages": [{"model_id": "noise-suppression-dpcrn-mamba-v2"}],
    })

    # The voice model received the noise model's output with its latency removed.
    np.testing.assert_allclose(runtimes["voice-isolate-dpcrn-v8"].inputs[0], 0.1, atol=1e-3)
    token = response["run_id"]
    np.testing.assert_allclose(_read_wav(service.runs.get(token, "stage-1").data), 0.1, atol=1e-3)
    np.testing.assert_allclose(_read_wav(service.runs.get(token, "aligned").data), 0.3, atol=1e-3)
    pipeline = response["pipeline"]
    assert [stage["model_id"] for stage in pipeline["stages"]] == ["noise-suppression-dpcrn-mamba-v2"]
    assert pipeline["latency_ms"] == pytest.approx(2.0 + 3.0)


def test_web_measure_aggregates_several_clips(monkeypatch):
    # A paired difference needs five clips, all on one side, to resolve.
    three = web_server._paired_summary([0.5, 0.6, 0.7])
    assert three["n"] == 3 and three["mean"] == pytest.approx(0.6) and three["wins"] == 3
    assert three["ci_low"] is not None and three["resolved"] is False
    five = web_server._paired_summary([0.5, 0.6, 0.7, 0.55, 0.65])
    assert five["resolved"] is True and five["ci_low"] > 0
    mixed = web_server._paired_summary([0.5, -0.6, 0.7, -0.55, 0.05, -0.1])
    assert mixed["resolved"] is False
    assert web_server._paired_summary([])["mean"] is None

    service = WebService()
    monkeypatch.setattr(service, "_runtime", lambda model_id, *args: _ChainRuntime(model_id, 0.5, latency=16))
    monkeypatch.setattr(
        web_server,
        "reference_free_metrics",
        lambda samples, sample_rate, *, include_dnsmos: {"dnsmos": {"dnsmos_ovr": 2.0 + float(np.abs(samples).mean()) * 10}},
    )
    clips = [{"audio": {"filename": f"clip{index}.wav", "data": _tone_url(value=0.1 + 0.02 * index)}} for index in range(5)]
    report = service.measure({"inputs": {"clips": clips}, "models": ["voice-isolate-dpcrn-v8"], "measurements": {"include_dnsmos": True}})

    assert len(report["clips"]) == 5
    assert [clip["input_files"]["audio"] for clip in report["clips"]] == [f"clip{index}.wav" for index in range(5)]
    row = report["aggregate"][0]
    assert row["clips_ok"] == 5 and row["clips_failed"] == 0
    dnsmos = row["metrics"]["dnsmos_ovr"]
    # Halving the level lowers this fake score on every clip: a resolved drop.
    assert dnsmos["n"] == 5 and dnsmos["mean"] < 0 and dnsmos["wins"] == 0 and dnsmos["resolved"] is True
    assert report["summary"] == {"clips": 5, "succeeded": 5, "failed": 0}

    # The background-job route accepts clips too.
    started = []
    monkeypatch.setattr(service, "_start_job", lambda job, target, payload: started.append(payload) or {"job_id": job.job_id})
    service.submit_measurement({"inputs": {"clips": clips[:1]}})
    assert started


def test_web_live_session_streams_frames_over_a_websocket(monkeypatch):
    import json as json_module
    import os
    import socket
    import struct

    class FakeStream:
        sample_rate, hop_length, win_length, dry_delay, dry_blend = 16_000, 160, 512, 480, 1.0
        providers = ["CPUExecutionProvider"]
        onset_guard = None

        def process_samples(self, samples):
            return np.asarray(samples, dtype=np.float32) * 0.5

    class FakeRuntime:
        def open_stream(self, parameters):
            assert parameters == {"dry_blend": 1.0}
            return FakeStream()

    service = WebService()
    monkeypatch.setattr(service, "_runtime", lambda *args: FakeRuntime())
    server = _serve(service)

    def frame(opcode, payload):
        mask = os.urandom(4)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        length = len(payload)
        header = bytes([0x80 | opcode, 0x80 | length]) if length < 126 else bytes([0x80 | opcode, 0x80 | 126]) + struct.pack("!H", length)
        return header + mask + masked

    def read(connection):
        def exact(count):
            data = b""
            while len(data) < count:
                chunk = connection.recv(count - len(data))
                assert chunk
                data += chunk
            return data

        first, second = exact(2)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", exact(2))[0]
        return first & 0x0F, exact(length)

    try:
        connection = socket.create_connection(("127.0.0.1", server.server_port), timeout=5)
        key = base64.b64encode(os.urandom(16)).decode()
        connection.sendall(f"GET /api/live HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode())
        handshake = b""
        while b"\r\n\r\n" not in handshake:
            handshake += connection.recv(1)
        assert handshake.startswith(b"HTTP/1.1 101")
        assert web_server.WebSocket.accept_key(key).encode() in handshake
        connection.sendall(frame(1, json_module.dumps({"model_id": "voice-isolate-dpcrn-v8", "parameters": {"dry_blend": 1.0}}).encode()))
        opcode, ready = read(connection)
        ready = json_module.loads(ready)
        assert opcode == 1 and ready["type"] == "ready" and ready["latency_samples"] == 480 and ready["win_length"] == 512
        samples = np.linspace(-0.5, 0.5, 32, dtype="<f4")
        connection.sendall(frame(2, struct.pack("<I", 7) + samples.tobytes()))
        opcode, reply = read(connection)
        sequence, _ = struct.unpack("<If", reply[:8])
        assert opcode == 2 and sequence == 7
        np.testing.assert_allclose(np.frombuffer(reply[8:], "<f4"), samples * 0.5)
        connection.sendall(frame(1, b'{"type": "stop"}'))
        opcode, stopped = read(connection)
        assert json_module.loads(stopped)["type"] == "stopped"
    finally:
        server.shutdown()
        server.server_close()


STYLESHEETS = ("tokens", "base", "components", "shell", "screens", "deck", "world", "annotate")
TOKENS = ("--page", "--canvas", "--sunken", "--line", "--line-strong", "--ink", "--ink-2", "--ink-3", "--nav", "--nav-2",
          "--on-nav", "--primary", "--on-primary", "--accent", "--mint", "--periwinkle", "--magenta", "--good", "--warn",
          "--bad", "--focus", "--font-sans", "--font-mono", "--radius-panel", "--radius-control", "--shadow-overlay",
          "--control-h")


def _styles(static: Path) -> str:
    """Every stylesheet the page links, in cascade order."""
    return "\n".join((static / "css" / f"{name}.css").read_text() for name in STYLESHEETS)


def _css_block(css: str, selector: str) -> str:
    start = css.index(selector + " {")
    return css[start:css.index("\n}", start)]


def test_web_design_tokens_and_fonts_are_packaged():
    import re as re_module

    static = WebService().static_dir
    fonts = static / "fonts"
    for name in ("IBMPlexSans-400", "IBMPlexSans-500", "IBMPlexSans-600", "IBMPlexMono-400", "IBMPlexMono-500"):
        assert (fonts / f"{name}.woff2").read_bytes()[:4] == b"wOF2", name
    assert "SIL Open Font License" in (fonts / "LICENSE.txt").read_text()

    tokens = (static / "css" / "tokens.css").read_text()
    light = _css_block(tokens, ":root")
    for token in TOKENS:
        assert re_module.search(rf"^\s*{re_module.escape(token)}:", light, re_module.M), token
    # Dark is the explicit choice and the system's choice when none was made;
    # both redefine the same colours.
    system_dark = tokens[tokens.index("@media (prefers-color-scheme: dark)"):]
    explicit_dark = _css_block(tokens, ':root[data-theme="dark"]')
    assert ':root:not([data-theme="light"])' in system_dark
    for token in ("--page", "--canvas", "--ink", "--line", "--primary", "--good", "--bad"):
        assert f"{token}:" in explicit_dark and f"{token}:" in system_dark, token

    base = (static / "css" / "base.css").read_text()
    faces = re_module.findall(r"@font-face\s*{[^}]*}", base)
    assert len(faces) == 5
    assert all(re_module.search(r'url\("/fonts/IBMPlex(Sans|Mono)-\d00\.woff2"\)', face) for face in faces)
    assert all("font-display: swap" in face for face in faces)

    html = (static / "index.html").read_text()
    links = re_module.findall(r'<link rel="stylesheet" href="/css/([a-z]+)\.css" />', html)
    assert tuple(links) == STYLESHEETS
    assert not (static / "styles.css").exists() and "/styles.css" not in html

    pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    assert '"web/static/css/*"' in pyproject and '"web/static/fonts/*"' in pyproject


def test_web_static_assets_are_packaged_and_wired():
    import json as json_module
    import re as re_module

    service = WebService()
    html = (service.static_dir / "index.html").read_text()
    favicon = (service.static_dir / "favicon.svg").read_text()
    audio_view = (service.static_dir / "audio-view.js").read_text()
    audio_worker = (service.static_dir / "audio-worker.js").read_text()
    app = (service.static_dir / "app.js").read_text()
    styles = _styles(service.static_dir)
    compare_deck = (service.static_dir / "compare-deck.js").read_text()
    panel = (service.static_dir / "transcribe.js").read_text()

    assert '<script src="/audio-view.js" defer></script>' in html
    assert '<link rel="icon" type="image/svg+xml" href="/favicon.svg" />' in html
    assert "waveform crossing through an open infinity loop" in favicon
    assert "fc4c02" in favicon
    assert '<img src="/favicon.svg" alt="" />' in html
    # Every player on the page is the comparison deck: previews are one-track decks.
    assert 'class="audio-inspector' not in html and html.count('class="preview-deck"') == 3
    assert "previewDeck(" in app
    assert '<script src="/compare-deck.js" defer></script>' in html
    assert html.index("/audio-view.js") < html.index("/compare-deck.js") < html.index("/transcribe.js") < html.index("/app.js")
    assert 'id="voice-deck"' in html
    assert "window.PureSoundCompareDeck" in compare_deck
    assert "source.loopStart" in compare_deck
    assert "new window.PureSoundCompareDeck" in app
    assert "handleShortcut" in app
    assert re_module.search(r'<span[^>]*>lenient</span><span[^>]*>strict</span>', html)
    assert re_module.search(r'<span[^>]*>mostly input</span><span[^>]*>model only</span>', html)
    assert 'data-screen="history"' in html
    assert 'id="measurement-deck"' in html
    assert "renderMeasurementDeck" in app
    assert "/benchmarks" in app
    assert 'mode: reference ? "diff" : "level"' in audio_view
    assert 'data.mode === "diff"' in audio_worker
    assert "SAFE_PEAK_DBFS = -1.5" in audio_view
    assert "model input and exported WAV stay unchanged" in compare_deck
    # The worker is sent one frame per column, never the whole recording.
    assert "function columnFrames(" in audio_view and "data.frames" in audio_worker
    for control in ('data-time-zoom="in"', 'data-freq-zoom="in"', "deck-tscroll", "deck-fscroll", 'data-setting="fftSize"', 'data-setting="scale"', "data-export-region"):
        assert control in compare_deck
    assert "sliceWav" in audio_view and "exportRegion" in compare_deck
    assert "self.postMessage({ id, type: \"progress\"" in audio_worker
    assert "/audio-worker.js" in audio_view
    assert 'provider.includes("CPU")' in app
    assert "updateProviderAvailability" in app
    assert "Apple MPS" in html
    assert 'api("/api/jobs"' in app
    assert 'kind: "measurement"' in app
    assert "measurementCsv" in app
    for element in ('id="measure-export-json"', 'id="measure-export-csv"', 'id="measure-form-note"',
                    'id="voice-onset-guard"', 'id="voice-onset-knobs"', 'id="measure-onset-guard"',
                    'id="measure-onset-knobs"', "DNSMOS"):
        assert element in html, element
    for name in ("include_dnsmos", "onsetGuardParameters", "onset_guard_margin_db"):
        assert name in app, name
    assert ".knob-grid[disabled]" in styles
    assert ".audio-limiter { display: inline-flex; align-items: center; justify-content: center;" in styles
    assert "overflow-wrap: anywhere" in styles

    # Clear buttons and the monitor explanation.
    for token in ('data-clear="voice"', 'data-clear="sv"', 'data-clear="measure"'):
        assert token in html
    for name in ("clearVoiceWorkspace", "clearSvWorkspace", "clearMeasurementReport", "updateClearButtons", "MONITOR_EXPLANATIONS"):
        assert name in app
    assert 'id="monitor-explain"' in html and 'role="radiogroup"' in html

    # The theme is applied before the stylesheet paints anything, and text on
    # the always-dark sidebar and players does not follow the theme swap.
    assert html.index('localStorage.getItem("puresound.theme")') < html.index("<body>")
    assert ':root[data-theme="dark"] {' in styles and "--on-nav: #ffffff;" in styles
    for rule in (".toast {", ".compare-deck {"):
        line = next(line for line in styles.splitlines() if line.startswith(rule))
        assert "color: var(--on-nav)" in line
    assert 'id="compare-records"' in html and "renderGateComparison" in app

    # ASR keys are typed into password fields and never written to browser storage.
    assert 'id="voice-transcribe"' in html and 'id="measurement-transcribe"' in html
    assert 'type="password" data-asr-azure-key' in panel and 'type="password" data-asr-eleven-key' in panel
    code = re_module.sub(r"/\*.*?\*/|//[^\n]*", "", panel, flags=re_module.S)
    assert not re_module.search(r"localStorage|sessionStorage|indexedDB|document\.cookie", code)
    assert 'kind: "transcription"' in panel

    # Sample clips: 16 kHz mono, and every speaker pair names clips that exist.
    index = json_module.loads((service.static_dir / "samples" / "index.json").read_text())
    assert {clip["task"] for clip in index["clips"]} == {"enhancement", "speaker", "world"}
    for clip in index["clips"]:
        with wave.open(str(service.static_dir / "samples" / clip["file"]), "rb") as wav:
            assert wav.getframerate() == 16_000 and wav.getnchannels() == 1
    ids = {clip["id"] for clip in index["clips"]}
    assert all({pair["enrollment"], pair["test"]} <= ids for pair in index["speaker_pairs"])


SCREENS = ("playground", "verify", "compare", "annotate", "world", "pipeline", "models", "history")
PRIMARY = {"playground": "voice-run", "verify": "sv-run", "compare": "measure-run", "annotate": "annotate-download",
           "world": "world-run", "pipeline": "pipe-run", "models": None, "history": None}


def _under(element, container) -> bool:
    return any(item is container for item in element["ancestors"])


class _Page:
    """index.html as elements with their ancestors, for contract checks."""

    def __init__(self, html: str):
        from html.parser import HTMLParser

        voids = {"img", "input", "br", "meta", "link", "hr", "source", "col", "wbr"}
        self.elements: list[dict] = []
        page = self

        class Parser(HTMLParser):
            stack: list[dict] = []

            def handle_starttag(self, tag, attrs):
                attrs = {key: value or "" for key, value in attrs}
                element = {"tag": tag, "attrs": attrs, "classes": set(attrs.get("class", "").split()),
                           "ancestors": list(self.stack)}
                page.elements.append(element)
                if tag not in voids:
                    self.stack.append(element)

            def handle_startendtag(self, tag, attrs):
                self.handle_starttag(tag, attrs)
                if tag not in voids:
                    self.stack.pop()

            def handle_endtag(self, tag):
                while self.stack:
                    if self.stack.pop()["tag"] == tag:
                        break

        Parser().feed(html)

    def by_id(self, element_id):
        return next(element for element in self.elements if element["attrs"].get("id") == element_id)

    def has_id(self, element_id):
        return any(element["attrs"].get("id") == element_id for element in self.elements)

    @staticmethod
    def inside(element, *, id=None, cls=None, tag=None):
        return any((id is None or item["attrs"].get("id") == id) and (cls is None or cls in item["classes"])
                   and (tag is None or item["tag"] == tag) for item in element["ancestors"])

    def within(self, container_id, *, cls=None, attr=None):
        return [element for element in self.elements if self.inside(element, id=container_id)
                and (cls is None or cls in element["classes"]) and (attr is None or attr in element["attrs"])]


def test_web_workbench_shell_contract():
    static = WebService().static_dir
    html = (static / "index.html").read_text()
    page = _Page(html)
    app = (static / "app.js").read_text()
    shell = (static / "shell.js").read_text()

    # Grouped navigation, in order; no global top bar.
    links = [element for element in page.elements if "nav-link" in element["classes"]]
    assert [link["attrs"]["data-screen"] for link in links] == list(SCREENS)
    assert all(link["attrs"]["href"] == f"#/{link['attrs']['data-screen']}" for link in links)
    groups = [element for element in page.elements if "nav-group" in element["classes"]]
    assert [[link["attrs"]["data-screen"] for link in links if _under(link, group)] for group in groups] == [
        ["playground", "verify", "compare"], ["annotate", "world", "pipeline"], ["models", "history"]]
    assert 'class="topbar' not in html and "topbar" not in app
    for control in ('data-lang="en"', 'data-lang="zh-TW"', 'data-theme-choice="auto"', 'data-theme-choice="light"',
                    'data-theme-choice="dark"', 'id="runtime-status"', 'id="nav-toggle"'):
        assert control in html, control

    # Every screen: a page header with a title and Help; the one primary action
    # of a tool screen in its header and nowhere else on it; an Inspector with
    # a labelled toggle; a Help dialog.
    for screen in SCREENS:
        section = page.by_id(f"screen-{screen}")
        assert section["tag"] == "section" and section["attrs"]["data-screen-panel"] == screen
        header = next(element for element in page.within(f"screen-{screen}") if "page-header" in element["classes"])
        in_header = [element for element in page.elements if _under(element, header)]
        assert any("page-title" in element["classes"] for element in in_header), screen
        assert any(element["attrs"].get("data-help") == screen for element in in_header), screen
        primaries = [element for element in page.within(f"screen-{screen}", cls="button-primary")]
        if PRIMARY[screen]:
            assert [element["attrs"].get("id") for element in primaries] == [PRIMARY[screen]], screen
            assert _under(primaries[0], header), screen
            assert any(element["attrs"].get("data-state-for") == screen for element in page.elements), screen
            inspector = page.by_id(f"inspector-{screen}")
            assert inspector["tag"] == "aside" and "inspector" in inspector["classes"] and inspector["attrs"].get("aria-label")
            assert page.inside(inspector, id=f"screen-{screen}") and page.inside(inspector, cls="workbench")
            toggle = next(element for element in in_header if element["attrs"].get("data-inspector-toggle") == screen)
            assert toggle["attrs"].get("aria-controls") == f"inspector-{screen}"
        else:
            assert primaries == [] and not page.has_id(f"inspector-{screen}"), screen
        dialog = page.by_id(f"help-{screen}")
        assert dialog["tag"] == "dialog" and "help-dialog" in dialog["classes"]

    order = [html.index(f'<script src="/{name}" defer></script>') for name in
             ("i18n-zh.js", "i18n.js", "routes.js", "shell.js", "compare-deck.js", "app.js")]
    assert order == sorted(order)
    for name in ("show", "current", "setState", "openInspector", "closeInspector", "toggleInspector", "openHelp",
                 "toast", "onShow"):
        assert f"{name}," in shell or f"{name} }}" in shell, name
    # The drawer: Esc and the backdrop close it, focus returns to Settings.
    for token in ('"Escape"', "inspector-backdrop", ".focus()", "(max-width: 1099px)", "puresound.screen", "puresound.inspector."):
        assert token in shell, token
    # Nothing announces a page load, and the old layout code is gone from app.js.
    assert "catalog models." not in app
    for name in ("setPlaygroundDrawer", "setMeasurementDrawer", "syncSettingsLayout", "routeFromHash", "switchScreen",
                 "setSidebarCollapsed", "THEME_KEY"):
        assert name not in app, name


def test_web_playground_and_verify_on_the_workbench():
    page = _Page((WebService().static_dir / "index.html").read_text())
    assert not page.within("screen-playground", cls="workspace-tabs")
    for element_id in ("voice-model", "voice-stage", "voice-variant", "voice-provider", "voice-dry-blend",
                       "voice-onset-guard", "voice-onset-t-arm", "voice-onset-t-forget", "voice-onset-tau-dn",
                       "voice-onset-margin", "voice-collect-extras"):
        assert page.inside(page.by_id(element_id), id="inspector-playground"), element_id
    for element_id in ("voice-audio", "voice-deck", "voice-result", "live-toggle"):
        assert page.inside(page.by_id(element_id), cls="workbench-main"), element_id
    for element_id in ("sv-enrollment", "sv-test", "sv-result"):
        assert page.inside(page.by_id(element_id), id="screen-verify") and page.inside(page.by_id(element_id), cls="workbench-main")
    for element_id in ("sv-model", "sv-provider", "sv-threshold"):
        assert page.inside(page.by_id(element_id), id="inspector-verify"), element_id
    for element_id in ("voice-run", "sv-run", "voice-cancel", "sv-cancel"):
        assert page.inside(page.by_id(element_id), cls="page-header"), element_id
    runners = [element for element in page.elements if "data-runner" in element["attrs"]]
    assert [element["attrs"]["data-runner"] for element in runners] == ["server", "device"]
    assert all(page.inside(element, id="inspector-playground") for element in runners)


def test_web_compare_pipeline_models_and_history_on_the_workbench():
    html = (WebService().static_dir / "index.html").read_text()
    page = _Page(html)
    assert not page.has_id("measurement-drawer") and "measurement-intro" not in html
    for element_id in ("measure-audio", "measure-reference", "measure-model-list", "measure-onset-guard",
                       "measure-timing-repeats"):
        assert page.inside(page.by_id(element_id), id="inspector-compare"), element_id
    for element_id in ("measure-run", "measure-cancel", "measure-export-json", "measure-export-zip"):
        assert page.inside(page.by_id(element_id), cls="page-header"), element_id
    assert page.inside(page.by_id("pipe-setup"), id="inspector-pipeline")
    run = page.by_id("pipe-run")
    assert page.inside(run, cls="page-header") and run["attrs"].get("form") == "pipe-setup" and run["attrs"]["type"] == "submit"
    assert "data-pipe-lang" not in html
    assert "stat-card" not in html and page.inside(page.by_id("model-grid"), id="screen-models")
    assert page.inside(page.by_id("validate-button"), id="screen-models")
    filters = [element for element in page.elements if "data-history-filter" in element["attrs"]]
    assert len(filters) == 5 and all(page.inside(element, cls="page-header") for element in filters)


def test_web_annotate_is_a_workbench_screen_and_the_standalone_tool_is_gone():
    static = WebService().static_dir
    repo = Path(__file__).resolve().parents[2]
    html = (static / "index.html").read_text()
    page = _Page(html)
    annotate = (static / "annotate.js").read_text()
    deck = (static / "compare-deck.js").read_text()
    app = (static / "app.js").read_text()
    assert not (repo / "tools" / "audio_annotator.html").exists()
    assert "Annotate" in (repo / "tools" / "README.md").read_text()
    assert page.inside(page.by_id("annotate-main"), id="screen-annotate")
    assert page.inside(page.by_id("annotate-inspector"), id="inspector-annotate")
    order = [html.index(f'<script src="/{name}" defer></script>') for name in ("shell.js", "annotate-model.js", "annotate.js", "app.js")]
    assert order == sorted(order)
    # The engine is scoped to its screen, the export comes from the model, and
    # its keys only act while the screen is on and nobody is typing.
    assert 'document.getElementById("screen-annotate")' in annotate and "document.querySelector(\"#" not in annotate
    for token in ("M.buildExport", "M.parseSpans", "M.shortcutTarget", 'current() === "annotate"', "window.PureSoundAnnotate", "addAudio"):
        assert token in annotate, token
    # Every deck can hand its audible track to Annotate; the deck's keys stand aside there.
    assert "PureSoundAnnotate" in deck and "data-annotate" in deck and 'show("annotate")' in deck
    assert 'current() === "annotate") return;' in app


def test_web_page_markup_is_balanced_and_workspaces_are_siblings():
    from html.parser import HTMLParser

    containers = {"div", "section", "aside", "label", "fieldset", "dl", "main", "header", "nav", "details", "table", "tbody", "thead", "tr"}

    class Balance(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack: list[str] = []
            self.problems: list[str] = []
            self.parents: dict[str, list[str]] = {}

        def handle_starttag(self, tag, attrs):
            if tag not in containers:
                return
            element_id = dict(attrs).get("id") or ""
            if element_id:
                self.parents[element_id] = [item for item in self.stack if item]
            self.stack.append(element_id or tag)

        def handle_endtag(self, tag):
            if tag in containers:
                if not self.stack:
                    self.problems.append(f"stray </{tag}> at line {self.getpos()[0]}")
                else:
                    self.stack.pop()

    html = (WebService().static_dir / "index.html").read_text()
    parser = Balance()
    parser.feed(html)
    assert parser.problems == [] and parser.stack == []
    # One missing </div> once nested one screen inside another, which hid it
    # whenever the outer one was not shown.
    screens = [f"screen-{name}" for name in SCREENS]
    for screen in screens:
        assert not set(parser.parents[screen]) & set(screens), screen


def test_web_serves_https_with_a_self_signed_certificate(tmp_path):
    import http.client
    import shutil
    import socket
    import ssl
    import threading

    if shutil.which("openssl") is None:
        pytest.skip("openssl is not installed")
    cert, key = web_server.self_signed_certificate(tmp_path / "tls", "127.0.0.1")
    assert (cert.stat().st_mode & 0o777) and (key.stat().st_mode & 0o777) == 0o600
    # Made once, then reused.
    assert web_server.self_signed_certificate(tmp_path / "tls", "127.0.0.1") == (cert, key)
    server = WebService().create_server("127.0.0.1", 0, tls=(cert, key))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        # Plain HTTP to the HTTPS port fails for that client only.
        with socket.create_connection(("127.0.0.1", server.server_port), timeout=5) as plain:
            plain.sendall(b"GET /api/health HTTP/1.0\r\n\r\n")
            plain.recv(64)
        context = ssl.create_default_context(cafile=str(cert))
        connection = http.client.HTTPSConnection("127.0.0.1", server.server_port, context=context, timeout=5)
        connection.request("GET", "/api/health")
        response = connection.getresponse()
        assert response.status == 200 and b'"status":"ok"' in response.read()
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    "host, entry",
    [("lab.example", "DNS:lab.example"), ("cafe", "DNS:cafe"), ("10.0.0.5", "IP:10.0.0.5"), ("fe80::1", "IP:fe80::1")],
)
def test_web_certificate_names_a_host_as_an_address_only_when_it_is_one(tmp_path, monkeypatch, host, entry):
    commands = []

    def fake_openssl(command, **options):
        commands.append(command)
        Path(command[command.index("-keyout") + 1]).write_text("key")
        Path(command[command.index("-out") + 1]).write_text("certificate")

    monkeypatch.setattr(web_server.shutil, "which", lambda name: "/usr/bin/openssl")
    monkeypatch.setattr(web_server.subprocess, "run", fake_openssl)
    web_server.self_signed_certificate(tmp_path / "tls", host)

    names = commands[0][commands[0].index("-addext") + 1].removeprefix("subjectAltName=").split(",")
    assert entry in names


@pytest.mark.parametrize(
    "target",
    [b"/../../etc/passwd", b"/app.js\x00.txt", b"/" + b"a" * 300, b"//etc/passwd", b"/samples/../../../etc/passwd"],
)
def test_web_static_route_answers_404_to_a_path_that_names_no_asset(target):
    import socket

    server = _serve(WebService())
    try:
        with socket.create_connection(("127.0.0.1", server.server_port), timeout=5) as connection:
            connection.sendall(b"GET " + target + b" HTTP/1.0\r\n\r\n")
            reply = b""
            while chunk := connection.recv(65536):
                reply += chunk
        assert reply.startswith(b"HTTP/1.0 404 ")
        assert b"asset not found" in reply and str(WebService().static_dir.parent).encode() not in reply
    finally:
        server.shutdown()
        server.server_close()


def test_web_every_response_is_cross_origin_isolated_and_the_old_demo_is_gone():
    import http.client

    server = _serve(WebService())
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        for path, status in (("/", 200), ("/app.js", 200), ("/audio-worker.js", 200), ("/device/worker.js", 200),
                             ("/api/health", 200), ("/missing.js", 404), ("/browser/", 404), ("/browser", 404)):
            connection.request("GET", path)
            response = connection.getresponse()
            response.read()
            assert response.status == status, path
            # WebAssembly threads need the page and every worker script isolated.
            assert response.getheader("Cross-Origin-Opener-Policy") == "same-origin", path
            assert response.getheader("Cross-Origin-Embedder-Policy") == "require-corp", path
        connection.request("GET", "/device/worker.js")
        response = connection.getresponse()
        response.read()
        assert response.getheader("Content-Type").startswith("application/javascript")
    finally:
        server.shutdown()
        server.server_close()


def test_web_device_inference_is_packaged_and_wired():
    static = WebService().static_dir
    root = Path(__file__).resolve().parents[2]
    html = (static / "index.html").read_text()
    app = (static / "app.js").read_text()
    capture = (static / "capture.js").read_text()
    device = (static / "device" / "device.js").read_text()
    worker = (static / "device" / "worker.js").read_text()

    assert not (static / "browser").exists()
    assert (static / "device" / "THIRD_PARTY_NOTICES.txt").read_text().startswith("ONNX Runtime Web")
    assert html.index("/capture.js") < html.index("/device/device.js") < html.index("/app.js")
    # One path per run: Playground's device runs go through the same module any screen uses.
    assert "PureSoundDevice.decode(" in app and "PureSoundDevice.process(" in app and "PureSoundDevice.liveLink(" in app
    assert "new Worker" not in app and "onnxruntime" not in app
    # One live session for both runners; only the link differs.
    assert app.count("new window.PureSoundCapture.LiveSession(") == 1 and "PureSoundCapture.serverLink(" in app
    assert "serverLink" in capture and "new WebSocket(" in capture
    assert 'new Worker(`${BASE}worker.js`, { type: "module" })' in device and 'const BASE = "/device/";' in device
    assert './vendor/ort/ort.wasm.min.mjs' in worker and './runtime/runtime.js' in worker
    builder = (root / "sdk" / "web" / "tools" / "build_assets.py").read_text()
    assert 'DEST = ROOT / "puresound/web/static/device"' in builder
    pyproject = (root / "pyproject.toml").read_text()
    assert '"web/static/device/*"' in pyproject and '"web/static/device/vendor/ort/*"' in pyproject
    assert "web/static/browser" not in pyproject


def test_web_live_session_whose_output_is_not_finite_ends_its_job_as_failed():
    class Stream:
        sample_rate, hop_length, win_length, dry_delay, dry_blend = 16_000, 160, 512, 0, 1.0
        providers = ["CPUExecutionProvider"]
        onset_guard = None

        def process_samples(self, samples):
            output = np.asarray(samples, dtype=np.float32).copy()
            output[0] = np.nan
            return output

    class Runtime:
        def open_stream(self, parameters):
            return Stream()

    class Socket:
        def __init__(self, messages):
            self.messages, self.sent = list(messages), []

        def receive(self):
            return self.messages.pop(0)

        def send(self, opcode, payload):
            self.sent.append(opcode)

        def send_json(self, value):
            self.sent.append(value)

    service = WebService()
    service._runtime = lambda *args: Runtime()
    chunk = np.full(8_000, 0.1, dtype="<f4").tobytes()
    socket = Socket(
        [(1, json.dumps({"model_id": "voice-isolate-dpcrn-v8"}).encode())]
        + [(2, b"\x00\x00\x00\x00" + chunk)] * 2
        + [(1, b'{"type": "stop"}')]
    )
    service.live_session(socket)

    assert socket.sent[-1]["type"] == "stopped"
    (job,) = service.jobs.list_snapshots()
    assert job["status"] == "failed" and "non-finite" in job["error"] and job["finished_at"] is not None


def test_web_live_session_is_kept_in_the_run_history_unless_too_short():
    service = WebService()

    class Stream:
        sample_rate, hop_length, win_length, dry_delay, dry_blend = 16_000, 160, 512, 480, 1.0
        providers = ["CPUExecutionProvider"]
        onset_guard = None

    job = service.jobs.create("voice-isolate-dpcrn-v8", kind="live")
    service.jobs.start(job.job_id, phase="streaming", progress=0.0)
    model = service.zoo.get("voice-isolate-dpcrn-v8")
    signal = np.full(16_000, 0.2, dtype=np.float32)
    # The reply stream is the input, `dry_delay` late and a window short.
    reply = np.concatenate([np.zeros(480, dtype=np.float32), signal * 0.5])[: 16_000 - 352]
    service._keep_live_session(job.job_id, model, Stream(), [signal], [reply], processed=16_000, busy=0.4, ended="stopped")

    snapshot = service.job(job.job_id)
    assert snapshot["kind"] == "live" and snapshot["status"] == "succeeded"
    result = snapshot["result"]
    assert result["rtf"] == pytest.approx(0.4) and result["live"]["ended"] == "stopped"
    token = result["run_id"]
    aligned = _read_wav(service.runs.get(token, "aligned").data)
    source = _read_wav(service.runs.get(token, "input").data)
    assert aligned.size == source.size == 16_000 - 352 - 480
    np.testing.assert_allclose(aligned, 0.1, atol=1e-3)
    np.testing.assert_allclose(_read_wav(service.runs.get(token, "removed").data), 0.1, atol=1e-3)

    short = service.jobs.create("voice-isolate-dpcrn-v8", kind="live")
    service.jobs.start(short.job_id, phase="streaming", progress=0.0)
    service._keep_live_session(short.job_id, model, Stream(), [np.zeros(320, np.float32)], [np.zeros(0, np.float32)], processed=320, busy=0.0, ended="disconnected")
    assert service.job(short.job_id)["status"] == "failed"


def _finished_measurement(service, clips=2):
    token = service.runs.put({
        "input": web_server.StoredOutput(b"RIFFin", "audio/wav", "input.wav", 0.0),
        "model-0": web_server.StoredOutput(b"RIFFout", "audio/wav", "m.wav", 0.0),
    })
    clip = {
        "input_files": {"audio": "Kitchen take.wav"},
        "baseline": {"output_url": f"/api/runs/{token}/input", "output": {"rms_dbfs": -30.0}, "reference_free": {"dnsmos": {"dnsmos_ovr": 2.5}}, "quality": {}},
        "reference_url": None,
        "models": [
            {"model_id": "voice-isolate-dpcrn-v8", "candidate_id": 0, "label": "v8 · blend 1.00", "output_url": f"/api/runs/{token}/model-0", "rtf": 0.4, "dry_blend": 1.0, "onset_guard": None, "output": {"rms_dbfs": -31.0}, "reference_free": {"dnsmos": {"dnsmos_ovr": 3.0}}, "quality": {}},
            {"model_id": "voice-isolate-dpcrn-v8", "candidate_id": 1, "label": "broken", "error": "boom"},
        ],
    }
    result = {"clips": [dict(clip) for _ in range(clips)], "aggregate": [], "request": {"provider": "cpu", "measurements": {"timing_repeats": 1}}, "host": {"load_average_1m": 1.0, "cpu_count": 4}} if clips > 1 else clip
    job = service.jobs.create("model-comparison", kind="measurement")
    service.jobs.start(job.job_id, phase="x", progress=0.1)
    service.jobs.succeed(job.job_id, result)
    return job.job_id


def test_web_exports_a_comparison_as_zip_and_report():
    import zipfile

    from puresound.web import exports

    service = WebService()
    job_id = _finished_measurement(service)
    snapshot = service.job(job_id)
    assert [name for name, _ in exports.job_files(snapshot)] == [
        "01_Kitchen_take/00_unprocessed.wav", "01_Kitchen_take/01_v8_blend_1.00.wav",
        "02_Kitchen_take/00_unprocessed.wav", "02_Kitchen_take/01_v8_blend_1.00.wav",
    ]
    body, content_type, filename = service.job_export(job_id, "zip")
    assert content_type == "application/zip" and filename.endswith(".zip")
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        assert archive.read("01_Kitchen_take/01_v8_blend_1.00.wav") == b"RIFFout"
        assert "report.json" in archive.namelist() and "MISSING.txt" not in archive.namelist()
    page, content_type, filename = service.job_export(job_id, "report", tz="Asia/Taipei")
    text = page.decode()
    assert content_type.startswith("text/html") and filename.endswith(".html")
    assert text.count('src="data:audio/wav;base64,') == 4
    assert "Unprocessed input" in text and "v8 · blend 1.00" in text and "boom" in text
    assert "+0.50" in text and "CST" in text

    # Playground outputs are named after what they are; an unfinished job has nothing to export.
    job = {"kind": "inference", "result": {"output_urls": {"audio": "/api/runs/" + "a" * 32 + "/audio", "aligned": "/api/runs/" + "a" * 32 + "/aligned", "stage-1": "/api/runs/" + "a" * 32 + "/stage-1"}, "pipeline": {"stages": [{"display_name": "Noise Suppression DPCRN-Mamba v2"}]}}}
    assert [name for name, _ in exports.job_files(job)] == ["output_stream.wav", "output.wav", "stage_1_Noise_Suppression_DPCRN-Mamba_v2.wav"]
    running = service.jobs.create("voice-isolate-dpcrn-v8")
    with pytest.raises(web_server.WebServiceError, match="has not succeeded"):
        service.job_export(running.job_id, "zip")


def test_web_export_routes_send_attachments():
    import http.client

    service = WebService()
    job_id = _finished_measurement(service, clips=1)
    server = _serve(service)
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", f"/api/jobs/{job_id}/outputs.zip")
        response = connection.getresponse()
        assert response.status == 200 and response.getheader("Content-Type") == "application/zip"
        assert response.getheader("Content-Disposition").startswith('attachment; filename="puresound-measurement-')
        assert response.read()[:2] == b"PK"
        connection.request("GET", f"/api/jobs/{job_id}/report.html?tz=UTC")
        response = connection.getresponse()
        assert response.status == 200 and b"PureSound comparison" in response.read()
    finally:
        server.shutdown()
        server.server_close()


def _fake_recogniser(texts):
    """A recogniser that returns canned words per call, timing each at 0.5 s."""
    from puresound.evaluation.transcribers import Transcript, Word

    calls = []

    def recognise(audio, rate, *, backend, model, language, credentials):
        calls.append({"backend": backend, "credentials": dict(credentials)})
        words = texts[len(calls) - 1].split()
        return Transcript(" ".join(words), tuple(Word(word, 0.5 * index, 0.5 * index + 0.4) for index, word in enumerate(words)), "en", backend, model or "fake", 0.01)

    return recognise, calls


def _run_outputs(service):
    token = service.runs.put({
        "input": web_server.StoredOutput(base64.b64decode(_tone_url(1600).split(",", 1)[1]), "audio/wav", "input.wav", 0.0),
        "aligned": web_server.StoredOutput(base64.b64decode(_tone_url(1600).split(",", 1)[1]), "audio/wav", "enhanced.wav", 0.0),
    })
    return [
        {"id": "input", "label": "Input", "source": {"url": f"/api/runs/{token}/input"}},
        {"id": "output", "label": "Output", "source": {"url": f"/api/runs/{token}/aligned"}},
    ]


def test_web_transcription_compares_tracks_with_the_input_transcript(monkeypatch):
    service = WebService()
    capabilities = service.asr_capabilities()
    assert capabilities["backends"] == ["whisper", "azure", "elevenlabs"]
    assert capabilities["whisper"]["default"] == "large-v3"

    recognise, calls = _fake_recogniser(["the far talker said hello", "hello"])
    monkeypatch.setattr(web_server, "run_transcriber", recognise)
    result = service.transcribe({"backend": "whisper", "tracks": _run_outputs(service)})

    assert result["mode"] == "track" and result["reference"]["track_id"] == "input" and result["unit"] == "word"
    source, output = result["tracks"]
    assert source["is_reference"] and source["counts"] is None
    assert output["counts"]["del"] == 4 and output["counts"]["hit"] == 1
    # Each missing word keeps the time it had in the reference transcript.
    missing = [op for op in output["alignment"] if op["op"] == "del"]
    assert [result["reference"]["times"][op["ref"]][0] for op in missing] == [0.0, 0.5, 1.0, 1.5]


def test_web_transcription_scores_against_a_reference_text_and_keeps_no_key(monkeypatch):
    service = WebService()
    recognise, calls = _fake_recogniser(["我們開會", "我們會"])
    monkeypatch.setattr(web_server, "run_transcriber", recognise)
    job = service.submit_transcription({
        "kind": "transcription",
        "backend": "elevenlabs",
        "credentials": {"key": "sk-very-secret"},
        "reference_text": "我們在開會",
        "tracks": _run_outputs(service),
    })
    deadline = time.time() + 5
    while service.job(job["job_id"])["status"] not in {"succeeded", "failed"} and time.time() < deadline:
        time.sleep(0.02)
    snapshot = service.job(job["job_id"])
    assert snapshot["status"] == "succeeded", snapshot["error"]
    result = snapshot["result"]
    assert result["mode"] == "reference" and result["unit"] == "character"
    assert [track["counts"]["del"] for track in result["tracks"]] == [1, 2]
    assert calls[0]["credentials"] == {"key": "sk-very-secret"}
    # The key reached the recogniser and nowhere else.
    assert "sk-very-secret" not in __import__("json").dumps(snapshot)
    # Word checks stay out of the run history.
    assert job["job_id"] not in {item["job_id"] for item in service.list_jobs(64)}


def test_web_transcription_failure_messages_are_scrubbed(monkeypatch):
    service = WebService()

    def broken(*args, credentials, **kwargs):
        raise RuntimeError(f"auth failed for {credentials['key']}")

    monkeypatch.setattr(web_server, "run_transcriber", broken)
    job = service.submit_transcription({"backend": "azure", "credentials": {"key": "azure-secret-xyz", "region": "eastus"}, "tracks": _run_outputs(service)})
    deadline = time.time() + 5
    while service.job(job["job_id"])["status"] not in {"succeeded", "failed"} and time.time() < deadline:
        time.sleep(0.02)
    snapshot = service.job(job["job_id"])
    assert snapshot["status"] == "failed" and "azure-secret-xyz" not in snapshot["error"] and "***" in snapshot["error"]


# --------------------------------------------------------------------------- #
# Pipeline inspector: trace jobs, their audio, and the screen's assets.


def _wait(service, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = service.job(job_id)
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def _payload(**change):
    return {"kind": "pipeline_trace", "recipe": "egs/ns/config/train_a.yaml", "foreground": {"sample": "reader-61"}, **change}


def test_a_trace_job_stores_its_audio_unclipped_and_returns_urls(pipeline_repo):
    service = WebService(pipeline_root=pipeline_repo)
    calls = {}

    def fake_trace_row(request, *, workspace_root, enhance, model, progress):
        calls.update(request=request, enhance=enhance, model=model, workspace=Path(workspace_root))
        progress(0.5, "synthesising the row")
        return {
            "report": {"schema": "puresound.pipeline-trace/1", "stages": [{"id": "row.emit", "audio": {"noisy": "23-row.emit-mixture"}}]},
            "audio": {"23-row.emit-mixture": np.array([0.0, 1.5, -0.25], dtype=np.float32)},
        }

    service.trace_row = fake_trace_row
    job = service.submit_pipeline_trace(_payload(seed=5))
    assert job["kind"] == "pipeline"
    finished = _wait(service, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    result = finished["result"]
    assert result["recipe"] == "egs/ns/config/train_a.yaml"
    url = result["audio_urls"]["23-row.emit-mixture"]
    token, name = url.split("/")[-2:]
    stored = service.run_output(token, name)
    samples, rate = sf.read(io.BytesIO(stored.data), dtype="float32")
    assert rate == 16000 and samples.max() == pytest.approx(1.5)
    assert calls["request"].seed == 5 and calls["enhance"] is None and calls["model"] is None
    assert not calls["workspace"].exists()
    json.dumps(result, allow_nan=False)


def test_a_trace_job_can_be_cancelled_between_steps(pipeline_repo):
    service = WebService(pipeline_root=pipeline_repo)

    def slow_trace_row(request, *, workspace_root, enhance, model, progress):
        for step in range(200):
            progress(step / 200, "scoring")
            time.sleep(0.01)
        raise AssertionError("not cancelled")

    service.trace_row = slow_trace_row
    job = service.submit_pipeline_trace(_payload())
    time.sleep(0.05)
    service.cancel_job(job["job_id"])
    assert _wait(service, job["job_id"])["status"] == "cancelled"


def test_a_model_at_another_rate_is_reported_and_not_run(pipeline_repo):
    service = WebService(pipeline_root=pipeline_repo)

    class Runtime:
        sample_rate = 48000
        model = type("M", (), {"task": "noise_suppression", "display_name": "Fake"})()

    service._runtime = lambda model_id, provider, variant: Runtime()
    seen = {}

    def fake_trace_row(request, *, workspace_root, enhance, model, progress):
        seen.update(enhance=enhance, model=model)
        return {"report": {"stages": []}, "audio": {"x": np.zeros(4, dtype=np.float32)}}

    service.trace_row = fake_trace_row
    job = service.submit_pipeline_trace(_payload(model_id="noise-suppression-dpcrn-mamba-v2"))
    assert _wait(service, job["job_id"])["status"] == "succeeded"
    assert seen["enhance"] is None and "48" in seen["model"]["skipped"]


def test_a_bad_request_fails_before_a_job_starts_and_the_overview_lists_the_inspector(pipeline_repo):
    service = WebService(pipeline_root=pipeline_repo)
    with pytest.raises(WebServiceError, match="offered training recipes"):
        service.submit_pipeline_trace(_payload(recipe="nope.yaml"))
    overview = service.pipeline_overview()
    assert overview["available"] is True and len(overview["recipes"]) == 2
    assert {noise["id"] for noise in overview["samples"]["noises"]} >= {"pink"}
    assert WebService().pipeline_overview()["available"] is False


def test_the_pipeline_screen_is_packaged_and_wired():
    static = WebService().static_dir
    html = (static / "index.html").read_text()
    app = (static / "app.js").read_text()
    screen = (static / "pipeline.js").read_text()
    styles = _styles(static)
    assert 'data-screen="pipeline"' in html and 'data-screen-panel="pipeline"' in html
    order = [html.index(f'<script src="/{name}" defer></script>') for name in ("compare-deck.js", "pipeline-model.js", "pipeline.js", "app.js")]
    assert order == sorted(order)
    for element in ('id="pipe-recipe"', 'id="pipe-seed"', 'id="pipe-foreground-sample"', 'id="pipe-noises"', 'id="pipe-rooms"',
                    'id="pipe-model"', 'id="pipe-run"', 'id="pipe-rail"', 'id="pipe-esnr-chart"', 'id="pipe-model-chart"',
                    'id="pipe-deck"', 'id="pipe-room-view"', 'id="pipe-rirs"'):
        assert element in html, element
    assert '"pipeline"' in app and "window.PureSoundApp" in app and "PureSoundPipeline" in app
    for token in ('kind: "pipeline_trace"', '"/api/pipeline"', "new window.PureSoundCompareDeck", "PureSoundPipelineModel",
                  "pipeline-stages.json", "PureSoundRoomView"):
        assert token in screen, token
    assert ".pipe-rail" in styles and ".pipe-stage" in styles


def test_the_room_view_loads_a_vendored_three_js_through_an_import_map():
    static = WebService().static_dir
    html = (static / "index.html").read_text()
    room = (static / "pipeline-room.js").read_text()
    vendor = static / "vendor" / "three"
    assert (vendor / "three.module.min.js").stat().st_size > 500_000
    assert "from 'three'" in (vendor / "addons" / "controls" / "OrbitControls.js").read_text()
    assert "MIT" in (vendor / "LICENSE").read_text()
    assert '<script type="importmap">' in html
    assert '"three": "/vendor/three/three.module.min.js"' in html and '"three/addons/": "/vendor/three/addons/"' in html
    # 670 KB of three.js is fetched on the first room drawn, not on every page load.
    assert "pipeline-room.js" not in html and 'import("/pipeline-room.js")' in (static / "pipeline.js").read_text()
    assert 'import * as THREE from "three"' in room and "OrbitControls" in room and "forceContextLoss" in room
    assert "window.PureSoundRoomView" in room and "puresound:room-view-ready" in room
    assert "prefers-reduced-motion" in room


def test_pipeline_traces_keep_their_audio_out_of_the_run_history(pipeline_repo, tmp_path):
    """A trace is some twenty float WAVs; stored with the playground's runs it
    would write them to disk and push the history's audio out of its bound."""
    history = tmp_path / "history"
    service = WebService(pipeline_root=pipeline_repo, history_dir=history)
    wav = web_server._float_wav(np.zeros(16, dtype=np.float32), 16_000)
    kept = [service.runs.put({"audio": web_server.StoredOutput(wav, "audio/wav", "a.wav", 0.0)}) for _ in range(2)]
    service.trace_row = lambda request, **_: {"report": {"stages": []}, "audio": {"x": np.zeros(16, dtype=np.float32)}}
    for seed in range(40):
        result = _wait(service, service.submit_pipeline_trace(_payload(seed=seed))["job_id"])["result"]
    assert all(service.run_output(token, "audio") is not None for token in kept)
    assert len([folder for folder in (history / "runs").iterdir() if folder.is_dir()]) == 2
    token, name = result["audio_urls"]["x"].split("/")[-2:]
    assert service.run_output(token, name) is not None


def test_a_pipeline_trace_refuses_a_model_that_is_not_an_enhancement_model(pipeline_repo):
    service = WebService(pipeline_root=pipeline_repo)
    speaker = next(model["id"] for model in service.list_models() if model["task"] == "speaker_embedding")
    with pytest.raises(WebServiceError, match="enhancement model"):
        service.submit_pipeline_trace(_payload(model_id=speaker))
    with pytest.raises(WebServiceError, match="not in the model zoo"):
        service.submit_pipeline_trace(_payload(model_id="no-such-model"))
