"""The voice-isolation demo (`egs/voice_isolate/scripts/demo.py`): checkpoint
discovery, model caching, and the two inference backends it drives."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def demo():
    spec = importlib.util.spec_from_file_location(
        "voice_isolate_demo", REPO_ROOT / "egs" / "voice_isolate" / "scripts" / "demo.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_config(root: Path, gain_normalized_to: int | None = None) -> Path:
    config_path = root / "config" / "skim.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  target_sample_rate: 16000",
                f"  gain_normalized_to: {'' if gain_normalized_to is None else gain_normalized_to}",
                "  proc_output_folder: ./proc",
                "trainer:",
                "  work_folder: ./exp/work",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


class IdentityModel(torch.nn.Module):
    def forward(self, wav, **kwargs):
        return wav


def _fake_metrics(dnsmos_fails: bool):
    def dnsmos(*args, **kwargs):
        if dnsmos_fails:
            raise RuntimeError("dnsmos unavailable")
        return {}

    return type("FakeMetrics", (), {
        "noise_reduction": staticmethod(lambda noisy, enhanced: torch.tensor([0.0])),
        "dnsmos_p835": staticmethod(dnsmos),
    })


class FakeRuntime:
    """What `get_cached_ort_runtime` hands back: an identity stream, optionally
    with presence logits that lead the audio by `streaming_delay_frames`."""

    sample_rate = 16000
    hop_length = 160
    providers = ["CPUExecutionProvider"]

    def __init__(self, extra_names=(), lead=0):
        self.extra_names = list(extra_names)
        self.manifest = {"streaming_delay_frames": lead} if lead else {}
        self.lead = lead
        self.n = 0

    def process_samples(self, samples):
        self.n = len(samples)
        return samples

    def flush(self):
        return np.zeros(0, dtype="float32")

    def drain_extras(self):
        # `lead` leading frames say "absent", every later frame says "present"
        v = np.full(self.n // self.hop_length + 4, 8.0, dtype="float32")
        v[: self.lead] = -8.0
        return {"vad_logit": v}


@pytest.mark.parametrize(
    "backend,expected",
    [
        ("PyTorch offline", ["exp/work/lightning/epoch=1.ckpt"]),
        ("ORT streaming", ["exp/work/streaming.onnx"]),
    ],
)
def test_checkpoint_discovery_lists_only_the_backend_s_files_under_work_folder(
    demo, tmp_path, backend, expected
):
    config_path = _write_config(tmp_path)
    _touch(tmp_path / "exp" / "work" / "lightning" / "epoch=1.ckpt")
    _touch(tmp_path / "exp" / "work" / "streaming.onnx")
    _touch(tmp_path / "exp" / "fallback.pth")
    _touch(tmp_path / "exp" / "other.onnx")

    choices = demo.scan_checkpoints(config_path, backend=backend)

    assert [label for label, _ in choices] == expected
    assert [Path(value) for _, value in choices] == [(tmp_path / e).resolve() for e in expected]


@pytest.mark.parametrize(
    "backend,files,message",
    [
        ("PyTorch offline", [], "No checkpoints found"),
        # a checkpoint is not an ONNX model, and the hint names the extension
        ("ORT streaming", ["exp/work/model.ckpt"], "No ONNX models found"),
    ],
)
def test_refresh_reports_when_the_backend_has_nothing_to_offer(
    demo, tmp_path, backend, files, message
):
    config_path = _write_config(tmp_path)
    for relative in files:
        _touch(tmp_path / relative)

    update, status = demo.refresh_checkpoints(config_path, backend=backend)

    assert update["choices"] == []
    assert update["value"] is None
    assert message in status
    if backend == "ORT streaming":
        assert ".onnx" in status


def test_models_and_runtimes_are_cached_per_key(demo, tmp_path, monkeypatch):
    """A gated run must not reuse an ORT runtime built without collection:
    `drain_extras` would raise, or return nothing and gate on nothing."""
    config_path = _write_config(tmp_path)
    ckpt_path = _touch(tmp_path / "exp" / "work" / "model.ckpt")
    loads = []
    monkeypatch.setattr(demo.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(demo, "load_model_from_checkpoint",
                        lambda config, ckpt, device: loads.append(ckpt) or IdentityModel())
    demo.MODEL_CACHE.clear()
    first, first_device = demo.get_cached_model(config_path, ckpt_path)
    second, second_device = demo.get_cached_model(config_path, ckpt_path)
    assert first is second and len(loads) == 1
    assert first_device == second_device == torch.device("cpu")

    built = []

    class Fake:
        def __init__(self, onnx_path, provider="auto", collect_extras=False):
            built.append(bool(collect_extras))

        def reset(self):
            pass

    import puresound.streaming as streaming_pkg

    monkeypatch.setattr(streaming_pkg, "StreamingDparnOrt", Fake)
    demo.ORT_RUNTIME_CACHE.clear()
    try:
        a = demo.get_cached_ort_runtime("m.onnx", provider="cpu", collect_extras=False)
        b = demo.get_cached_ort_runtime("m.onnx", provider="cpu", collect_extras=True)
        c = demo.get_cached_ort_runtime("m.onnx", provider="cpu", collect_extras=True)
    finally:
        demo.ORT_RUNTIME_CACHE.clear()
    assert built == [False, True]
    assert a is not b and b is c


def test_the_default_torch_device_falls_back_to_mps(demo, monkeypatch):
    monkeypatch.setattr(demo.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(demo.torch.backends.mps, "is_available", lambda: True)
    assert demo.default_torch_device() == "mps"


def test_torch_enhancement_writes_outputs_keeps_gain_and_reports_progress(
    demo, tmp_path, monkeypatch, write_tone_wav
):
    """The uploaded audio is evaluated at its own level (no recipe gain
    normalisation), a missing DNSMOS is reported rather than fatal, and every
    step shows up in the progress log."""
    config_path = _write_config(tmp_path, gain_normalized_to=-28)
    input_path = tmp_path / "input.wav"
    checkpoint_path = _touch(tmp_path / "exp" / "work" / "model.ckpt")
    write_tone_wav(input_path, sample_rate=16000, duration=0.1)

    monkeypatch.setattr(demo, "get_cached_model",
                        lambda config_path, checkpoint_path: (IdentityModel(), torch.device("cpu")))
    monkeypatch.setattr(demo, "get_metrics_class", lambda: _fake_metrics(dnsmos_fails=True))
    open_calls = []
    original_open = demo.AudioIO.open
    monkeypatch.setattr(demo.AudioIO, "open",
                        lambda *a, **kw: open_calls.append(kw) or original_open(*a, **kw))
    events = []

    before, after, spectrogram, metrics_rows, status = demo.enhance_audio(
        config_path, checkpoint_path, input_path,
        progress=lambda value, desc=None: events.append(desc),
    )

    assert Path(before) == input_path
    assert Path(after).is_file()
    assert Path(spectrogram).is_file() and Path(spectrogram).suffix == ".png"
    assert open_calls[0]["target_lvl"] is None
    assert "Enhanced audio saved" in status
    assert "DNSMOS skipped" in status
    assert ["sample_rate", "16000", "16000", "Hz"] in metrics_rows
    assert any(row[0] == "dnsmos_p835" and "skipped" in row[3] for row in metrics_rows)

    assert "Loading config" in events
    assert "Running inference on cpu for 0.10s audio" in events
    assert "Rendering spectrogram comparison" in events
    assert events[-1] == "Done"
    assert "Progress:" in status
    assert "- Computing no-reference metrics" in status


def test_inference_errors_come_back_as_five_outputs(demo, tmp_path):
    outputs = demo.run_demo_inference(
        str(tmp_path / "missing.yaml"),
        str(tmp_path / "missing.ckpt"),
        str(tmp_path / "missing.wav"),
    )
    assert len(outputs) == 5
    assert outputs[1] is None and outputs[2] is None
    assert outputs[3] == []
    assert outputs[4].startswith("Error:")


def _ort_setup(demo, tmp_path, monkeypatch, write_tone_wav, runtime, real_metrics=False):
    config_path = _write_config(tmp_path)
    input_path = tmp_path / "input.wav"
    onnx_path = _touch(tmp_path / "exp" / "work" / "model.onnx")
    write_tone_wav(input_path, sample_rate=16000, duration=0.5)
    monkeypatch.setattr(demo, "get_cached_ort_runtime",
                        lambda onnx_path, provider="auto", collect_extras=False: runtime)
    if not real_metrics:
        monkeypatch.setattr(demo, "get_metrics_class", lambda: _fake_metrics(dnsmos_fails=True))
    return str(config_path), str(onnx_path), str(input_path)


@pytest.mark.slow  # the library's DNSMOS sessions
def test_ort_enhancement_runs_the_stream_and_scores_it(
    demo, tmp_path, monkeypatch, write_tone_wav
):
    """End to end through the library's own no-reference metrics."""
    config, onnx, wav = _ort_setup(
        demo, tmp_path, monkeypatch, write_tone_wav, FakeRuntime(), real_metrics=True
    )
    before, after, spectrogram, metrics_rows, status = demo.enhance_audio(
        config, onnx, wav, backend="ORT streaming", ort_provider="cpu"
    )
    assert Path(before) == Path(wav)
    assert Path(after).is_file()
    assert Path(spectrogram).is_file()
    assert "Running ORT streaming inference" in status
    assert ["sample_rate", "16000", "16000", "Hz"] in metrics_rows


def test_the_ort_gate_needs_logits_and_compensates_their_lead(
    demo, tmp_path, monkeypatch, write_tone_wav
):
    """A graph without logits must say so and name the config that produces
    one, not gate on nothing. A graph with them leads the audio by
    `streaming_delay_frames`; not compensating makes the gate act ~30 ms early
    and clip the front of every kept span, which no level summary would flag."""
    config, onnx, wav = _ort_setup(demo, tmp_path, monkeypatch, write_tone_wav, FakeRuntime())
    with pytest.raises(ValueError, match="vad_logit") as excinfo:
        demo.enhance_audio(config, onnx, wav, backend="ORT streaming", gate_mode="Soft")
    assert "infer_dpcrn_heads.yaml" in str(excinfo.value)

    runtime = FakeRuntime(extra_names=["vad_logit"], lead=3)
    monkeypatch.setattr(demo, "get_cached_ort_runtime",
                        lambda onnx_path, provider="auto", collect_extras=False: runtime)
    captured = {}
    real_gate = demo.apply_vad_gate

    def spy(enhanced, vad_logits, *a, **kw):
        captured["logits"] = vad_logits.clone()
        return real_gate(enhanced, vad_logits, *a, **kw)

    monkeypatch.setattr(demo, "apply_vad_gate", spy)
    outputs = demo.enhance_audio(config, onnx, wav, backend="ORT streaming", gate_mode="Soft")
    assert not outputs[4].startswith("Error:"), outputs[4]
    assert "logit lead=3 frames compensated" in outputs[4]
    # the leading "absent" frames are gone, so the gate cannot close over the
    # front of a kept span
    seq = captured["logits"].reshape(-1)
    assert float(seq.min()) > 0.0, f"leading absent frames survived: {seq[:5].tolist()}"
