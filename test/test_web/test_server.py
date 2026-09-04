from __future__ import annotations

import base64
import io
import time
import wave

import numpy as np
import pytest

from puresound.cli import build_parser
from puresound.inference import InferenceCancelled, InferenceResult
from puresound.web import WebService
import puresound.web.server as web_server
from puresound.web.measurements import align_streaming_output


def _wav_data_url(samples: int = 160) -> str:
    payload = io.BytesIO()
    with wave.open(payload, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * samples)
    return "data:audio/wav;base64," + base64.b64encode(payload.getvalue()).decode("ascii")


def test_web_catalog_payload_is_path_stable_and_default_first():
    service = WebService()
    models = service.list_models()

    assert models
    assert models[0]["id"] == "voice-isolate-dpcrn-v8"
    assert all(not str(item["source_checkpoint"]).startswith("/") for item in models if item["source_checkpoint"])
    artifact = service.inspect_model("voice-isolate-dpcrn-v8")["artifacts"][0]
    assert artifact["filename"] == "dpcrn_v8.onnx"
    assert artifact["available"] is True


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


def test_web_infer_can_attach_reference_free_measurements(monkeypatch):
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
            "measurements": {"include_dnsmos": True},
        }
    )

    assert response["measurements"]["output"]["sample_rate"] == 16_000
    assert response["measurements"]["reference_free"]["dnsmos"]["dnsmos_ovr"] == 4.2


def test_web_upload_rejects_local_paths_by_default():
    service = WebService()
    try:
        service.infer(
            {
                "model_id": "voice-isolate-dpcrn-v8",
                "inputs": {"audio": "/tmp/input.wav"},
            }
        )
    except web_server.WebServiceError as exc:
        assert "local input paths are disabled" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("local input path unexpectedly accepted")


def test_web_cli_accepts_ip_and_port_arguments():
    args = build_parser().parse_args(["web", "--ip", "0.0.0.0", "--port", "9123"])

    assert args.host == "0.0.0.0"
    assert args.port == 9123


@pytest.mark.parametrize("port", ["0", "65536", "not-a-port"])
def test_web_cli_rejects_invalid_ports(port):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["web", "--port", port])


def test_web_service_passes_bind_address_to_http_server(monkeypatch):
    received = {}

    class FakeHttpServer:
        def __init__(self, address, handler):
            received["address"] = address
            received["handler"] = handler

    monkeypatch.setattr(web_server, "ThreadingHTTPServer", FakeHttpServer)
    server = WebService().create_server("192.0.2.10", 8123)

    assert received["address"] == ("192.0.2.10", 8123)
    assert server is not None


def test_web_audio_inspector_assets_are_packaged():
    service = WebService()
    html = (service.static_dir / "index.html").read_text()
    audio_view = (service.static_dir / "audio-view.js").read_text()
    audio_worker = (service.static_dir / "audio-worker.js").read_text()
    app = (service.static_dir / "app.js").read_text()
    styles = (service.static_dir / "styles.css").read_text()

    assert '<script src="/audio-view.js" defer></script>' in html
    assert html.count('class="audio-inspector') == 4
    assert "SPECTROGRAM · 0–8 KHZ" in audio_view
    assert "SAFE_PEAK_DBFS = -1.5" in audio_view
    assert "model input and exported WAV stay unchanged" in audio_view
    assert "self.postMessage({ id, type: \"progress\"" in audio_worker
    assert "/audio-worker.js" in audio_view
    assert 'provider.includes("CPU")' in app
    assert "updateProviderAvailability" in app
    assert "Apple MPS" in html
    assert 'api("/api/jobs"' in app
    assert 'kind: "measurement"' in app
    assert "measurementCsv" in app
    assert 'id="measure-export-json"' in html
    assert 'id="measure-export-csv"' in html
    assert 'id="measure-job-progress"' in html
    assert "Compare model outputs" in html
    assert "measurement-steps" in html
    assert 'id="measurement-drawer"' in html
    assert 'id="measurement-audio-results"' in html
    assert "setMeasurementDrawer" in app
    assert 'id="measure-form-note"></span>' in html
    assert "DNSMOS" in html
    assert "include_dnsmos" in app
    assert "sidebar-collapsed" in app
    assert 'id="playground-drawer-voice"' in html
    assert 'id="playground-drawer-sv"' in html
    assert 'id="playground-drawer-backdrop"' in html
    assert "setPlaygroundDrawer" in app
    assert ".playground-drawer.is-open" in styles
    assert ".measurement-icon, .step-number, .status-icon, .note-mark" in styles
    assert ".settings-icon,\n.catalog-check-chevron" in styles
    assert ".audio-limiter { display: inline-flex; align-items: center; justify-content: center;" in styles
    assert ".measurement-step strong, .measurement-step div > span" in styles
    assert "overflow-wrap: anywhere" in styles


def test_web_measurement_report_compares_selected_models(monkeypatch):
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
            "models": ["voice-isolate-dpcrn-v8"],
            "provider": "cpu",
            "parameters": {"dry_blend": 0.8},
            "measurements": {"include_dnsmos": True},
        }
    )

    assert report["input"]["sample_rate"] == 16_000
    assert report["request"]["models"] == ["voice-isolate-dpcrn-v8"]
    assert report["request"]["provider"] == "cpu"
    assert report["models"][0]["model_id"] == "voice-isolate-dpcrn-v8"
    assert report["models"][0]["output"]["rms_dbfs"] < 0
    assert report["models"][0]["quality"] == {}
    assert report["models"][0]["reference_free"]["dnsmos"]["dnsmos_ovr"] == 3.5
    assert report["models"][0]["output_url"].startswith("/api/runs/")
    assert report["summary"] == {"succeeded": 1, "failed": 0}
    assert report["request"]["measurements"] == {"include_dnsmos": True}


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


def test_web_async_job_reports_completion_and_keeps_result(monkeypatch):
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
            progress_callback(0.5, "processing_frames")
            return InferenceResult(
                model_id="voice-isolate-dpcrn-v8",
                task="voice_isolation",
                outputs={"audio": np.zeros(160, dtype=np.float32)},
                provider="CPUExecutionProvider",
                elapsed_seconds=0.01,
                rtf=0.1,
                sample_rate=16_000,
            )

    monkeypatch.setattr(service, "_runtime", lambda *args: FakeRuntime())
    job = service.submit_inference(
        {
            "model_id": "voice-isolate-dpcrn-v8",
            "provider": "cpu",
            "inputs": {"audio": {"filename": "input.wav", "data": _wav_data_url()}},
        }
    )
    deadline = time.time() + 5
    while time.time() < deadline:
        current = service.job(job["job_id"])
        if current["status"] in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.01)

    assert current["status"] == "succeeded"
    assert current["progress"] == 1.0
    assert current["result"]["output_urls"]["audio"].startswith("/api/runs/")


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


def test_web_measurement_runs_as_a_shared_background_job(monkeypatch):
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
    job = service.submit_measurement(
        {
            "inputs": {"audio": {"filename": "input.wav", "data": _wav_data_url()}},
            "models": ["voice-isolate-dpcrn-v8"],
        }
    )
    deadline = time.time() + 5
    while time.time() < deadline:
        current = service.job(job["job_id"])
        if current["status"] in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.002)

    assert current["kind"] == "measurement"
    assert current["status"] == "succeeded"
    assert current["result"]["models"][0]["model_id"] == "voice-isolate-dpcrn-v8"
