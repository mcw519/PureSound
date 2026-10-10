"""Application-only SDK comparison: delay/flush, cleanup and credential privacy."""
import io
import json
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from puresound.inference import InferenceCancelled
from puresound.web import WebService
from puresound.web import aicoustics as aic
from puresound.web.server import WebServiceError
from test.web.test_world import request, wait


@pytest.fixture
def sdk(monkeypatch):
    sessions = []

    class Processor:
        def __init__(self, model, key):
            self.key = key
            self.pending = np.zeros(37, dtype=np.float32)
            self.terminated = False
            sessions.append(self)

        def initialize(self, config):
            self.config = config

        def get_context(self):
            return self

        def set_parameter(self, parameter, value):
            self.level = value

        def get_audio_delay(self):
            return 37

        def process(self, samples):
            assert samples.shape == (64,)
            full = np.r_[self.pending, samples * 0.5]
            self.pending = full[64:]
            return full[:64]

        def terminate_session(self):
            self.terminated = True

    fake = SimpleNamespace(
        Processor=Processor,
        ProcessorConfig=SimpleNamespace(optimal=lambda model, sample_rate: SimpleNamespace(block_size=64)),
        ProcessorParameter=SimpleNamespace(EnhancementLevel=1),
        get_sdk_version=lambda: "test-sdk",
    )
    model = SimpleNamespace(get_id=lambda: "quail-vf-test-build")
    monkeypatch.setattr(aic, "_sdk", lambda: fake)
    monkeypatch.setattr(aic, "_model", lambda sdk, model_id: (model, "fixture-checksum"))
    return sessions


@pytest.mark.parametrize("length", [1, 13, 64, 77, 2049])
def test_sdk_delay_flush_and_no_input_mutation(sdk, length):
    samples = np.linspace(-0.1, 0.1, length, dtype=np.float32)
    original = samples.copy()
    result = aic.process(samples, 16000, {"api_key": "test-secret", "enhancement_level": 0.75})
    np.testing.assert_array_equal(result.output, samples * 0.5)
    np.testing.assert_array_equal(samples, original)
    assert sdk[-1].terminated and sdk[-1].level == 0.75
    assert result.metadata["audio_delay_samples"] == 37
    assert result.metadata["output_samples"] == length
    assert result.metadata["processed_samples"] % 64 == 0
    assert "test-secret" not in json.dumps(result.metadata)


def test_sdk_state_is_fresh_each_call_and_cancel_terminates(sdk):
    aic.process(np.ones(65, dtype=np.float32), 16000, {"api_key": "test-secret"})
    silent = aic.process(np.zeros(65, dtype=np.float32), 16000, {"api_key": "test-secret"})
    assert not np.any(silent.output)

    def cancel(value, phase):
        if phase == "aicoustics_inference" and len(sdk) == 3:
            raise InferenceCancelled("cancel")

    with pytest.raises(InferenceCancelled):
        aic.process(np.zeros(2000, dtype=np.float32), 16000, {"api_key": "test-secret"}, progress=cancel)
    assert len(sdk) == 3 and sdk[-1].terminated


def test_validation_and_disabled_provider(monkeypatch):
    assert aic.options({"enabled": False, "api_key": "test-secret"}) is None
    assert aic.options(None) is None
    for value in [True, {"enabled": True}, {"enabled": True, "api_key": "x", "enhancement_level": float("nan")}, {"enabled": True, "api_key": "x", "model_id": "unknown"}]:
        with pytest.raises(aic.AicousticsError):
            aic.options(value)
    monkeypatch.setattr(aic, "process", lambda *a, **k: pytest.fail("disabled comparator ran"))
    service = WebService()
    job = wait(service, service.submit_world({**request(), "aicoustics": {"enabled": False, "api_key": "test-secret"}}))
    assert job["status"] == "succeeded"
    assert job["result"]["comparisons"] == {}
    with pytest.raises(WebServiceError, match="SDK key"):
        service.submit_world({**request(), "aicoustics": {"enabled": True}})


def test_report_uses_same_mixture_and_exports_no_key(monkeypatch, tmp_path):
    monkeypatch.setattr(aic, "capabilities", lambda: {"available": True})
    seen = []

    def process(samples, rate, config, progress):
        seen.append(samples.copy())
        progress(1, "aicoustics_inference")
        return aic.AicousticsResult(samples * 0.5, {"display_name": "SDK fixture", "audio_delay_ms": 2, "rtf": 0.1})

    monkeypatch.setattr(aic, "process", process)
    service = WebService(history_dir=tmp_path)
    local_seen = []

    def enhance(samples, rate):
        local_seen.append(samples.copy())
        # An in-place local implementation cannot change the shared mixture
        # or what the optional comparator receives.
        samples *= 0.75
        return samples

    monkeypatch.setattr(service, "_pipeline_enhancer", lambda *args: (enhance, {"id": "local-fixture"}))
    payload = {**request(), "aicoustics": {"enabled": True, "api_key": "SENTINEL-SECRET-KEY"}}
    job = wait(service, service.submit_world(payload))
    assert job["status"] == "succeeded", job["error"]
    report = job["result"]
    comparison = report["comparisons"]["aicoustics"]
    assert comparison["status"] == "succeeded"
    assert {"target", "speech", "near"} == set(comparison["metrics"])
    import soundfile as sf
    raw = sf.read(io.BytesIO(service.run_output(report["run_id"], "input").data), dtype="float32")[0]
    np.testing.assert_array_equal(seen[0], raw)
    np.testing.assert_array_equal(local_seen[0], raw)
    output = sf.read(io.BytesIO(service.run_output(report["run_id"], "aligned").data), dtype="float32")[0]
    np.testing.assert_array_equal(output, raw * 0.75)
    assert "SENTINEL-SECRET-KEY" not in json.dumps(job)
    package = service.run_output(report["run_id"], "scene-package").data
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        assert {"aicoustics.wav", "aicoustics-removed.wav"} <= set(archive.namelist())
        assert b"SENTINEL-SECRET-KEY" not in archive.read("report.json")
    html, _, _ = service.job_export(job["job_id"], "report")
    assert b"aicoustics" in html and b"SENTINEL-SECRET-KEY" not in html
    restored = WebService(history_dir=tmp_path).jobs.snapshot(job["job_id"])
    assert "SENTINEL-SECRET-KEY" not in json.dumps(restored)


def test_sdk_error_is_safe_and_does_not_replace_local_audio(sdk, monkeypatch):
    def broken(self, audio):
        raise RuntimeError("credential test-secret leaked by upstream")

    monkeypatch.setattr(type(sdk[0]) if sdk else aic._sdk().Processor, "process", broken)
    with pytest.raises(aic.AicousticsError) as error:
        aic.process(np.zeros(160), 16000, {"api_key": "test-secret"})
    assert "test-secret" not in str(error.value)
    assert sdk[-1].terminated


def test_sweep_requests_are_redacted_and_error_keeps_local_result(monkeypatch):
    monkeypatch.setattr(aic, "capabilities", lambda: {"available": True})
    def fail(*args, **kwargs):
        raise aic.AicousticsError("Authorization unavailable")
    monkeypatch.setattr(aic, "process", fail)
    service = WebService()
    payload = {**request(), "kind": "world_sweep", "aicoustics": {"enabled": True, "api_key": "SENTINEL-SECRET-KEY"}, "axes": [{"parameter": "noise_gain_change_db", "values": [0]}, {"parameter": "interferer_gain_change_db", "values": [0]}]}
    job = wait(service, service.submit_world(payload))
    assert job["status"] == "succeeded"
    assert "SENTINEL-SECRET-KEY" not in json.dumps(job)
    cell = job["result"]["cells"][0]["result"]
    assert cell["comparisons"]["aicoustics"]["status"] == "failed"
    assert "aicoustics" not in cell["output_urls"]
    assert "aligned" in cell["output_urls"]
    assert "api_key" not in job["result"]["request"]["aicoustics"]
    from puresound.web.world import sweep_progress
    summary = sweep_progress(job["result"])
    assert summary["cells"][0]["aicoustics_status"] == "failed"
    assert summary["cells"][0]["aicoustics_error"] == "Authorization unavailable"
