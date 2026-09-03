from __future__ import annotations

import base64
import io
import wave

import numpy as np
import pytest

from puresound.cli import build_parser
from puresound.inference import InferenceResult
from puresound.web import WebService
import puresound.web.server as web_server


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
        def infer(self, *, inputs, parameters):
            assert set(inputs) == {"audio"}
            assert parameters == {"dry_blend": 0.8}
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
    app = (service.static_dir / "app.js").read_text()
    styles = (service.static_dir / "styles.css").read_text()

    assert '<script src="/audio-view.js" defer></script>' in html
    assert html.count('class="audio-inspector') == 4
    assert "SPECTROGRAM · 0–8 KHZ" in audio_view
    assert "SAFE_PEAK_DBFS = -1.5" in audio_view
    assert "model input and exported WAV stay unchanged" in audio_view
    assert 'provider.includes("CPU")' in app
    assert "overflow-wrap: anywhere" in styles
