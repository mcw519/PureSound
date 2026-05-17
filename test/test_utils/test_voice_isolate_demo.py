import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_demo_module():
    spec = importlib.util.spec_from_file_location(
        "voice_isolate_demo", REPO_ROOT / "egs" / "voice_isolate" / "demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_demo_config(
    root: Path, *, work_folder: str = "./exp/work", gain_nomalized_to: int | None = None
) -> Path:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "skim.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  target_sample_rate: 16000",
                f"  gain_nomalized_to: {'' if gain_nomalized_to is None else gain_nomalized_to}",
                "  proc_output_folder: ./proc",
                "trainer:",
                f"  work_folder: {work_folder}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def test_scan_checkpoints_uses_work_folder_and_exp_fallback(tmp_path):
    demo = _load_demo_module()
    config_path = _write_demo_config(tmp_path)
    work_ckpt = tmp_path / "exp" / "work" / "lightning" / "epoch=1.ckpt"
    fallback_ckpt = tmp_path / "exp" / "fallback.pth"
    work_ckpt.parent.mkdir(parents=True)
    fallback_ckpt.parent.mkdir(parents=True, exist_ok=True)
    work_ckpt.write_text("ckpt", encoding="utf-8")
    fallback_ckpt.write_text("ckpt", encoding="utf-8")

    choices = demo.scan_checkpoints(config_path)

    labels = [label for label, _value in choices]
    values = [Path(value) for _label, value in choices]
    assert "exp/work/lightning/epoch=1.ckpt" in labels
    assert "exp/fallback.pth" in labels
    assert work_ckpt.resolve() in values
    assert fallback_ckpt.resolve() in values


def test_refresh_checkpoints_reports_empty_choices(tmp_path):
    demo = _load_demo_module()
    config_path = _write_demo_config(tmp_path)

    update, status = demo.refresh_checkpoints(config_path)

    assert update["choices"] == []
    assert update["value"] is None
    assert "No checkpoints found" in status


def test_get_cached_model_reuses_same_config_checkpoint_pair(tmp_path, monkeypatch):
    demo = _load_demo_module()
    demo.MODEL_CACHE.clear()
    config_path = _write_demo_config(tmp_path)
    ckpt_path = tmp_path / "exp" / "work" / "model.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_text("ckpt", encoding="utf-8")
    calls = []

    class IdentityModel(torch.nn.Module):
        def forward(self, wav):
            return wav

    def fake_load_model_from_checkpoint(config_path, checkpoint_path, device):
        calls.append((config_path, checkpoint_path, device))
        return IdentityModel()

    monkeypatch.setattr(demo.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(demo, "load_model_from_checkpoint", fake_load_model_from_checkpoint)

    first_model, first_device = demo.get_cached_model(config_path, ckpt_path)
    second_model, second_device = demo.get_cached_model(config_path, ckpt_path)

    assert first_model is second_model
    assert first_device == second_device == torch.device("cpu")
    assert len(calls) == 1


def test_enhance_audio_writes_output_and_skips_dnsmos(
    tmp_path, monkeypatch, write_tone_wav
):
    demo = _load_demo_module()
    config_path = _write_demo_config(tmp_path)
    input_path = tmp_path / "input.wav"
    checkpoint_path = tmp_path / "exp" / "work" / "model.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text("ckpt", encoding="utf-8")
    write_tone_wav(input_path, sample_rate=16000, duration=0.1)

    class IdentityModel(torch.nn.Module):
        def forward(self, wav):
            return wav

    monkeypatch.setattr(
        demo,
        "get_cached_model",
        lambda config_path, checkpoint_path: (IdentityModel(), torch.device("cpu")),
    )

    class FakeMetrics:
        @staticmethod
        def noise_reduction(noisy, enhanced):
            return torch.tensor([0.0])

        @staticmethod
        def dnsmos_p835(*args, **kwargs):
            raise RuntimeError("dnsmos unavailable")

    monkeypatch.setattr(demo, "get_metrics_class", lambda: FakeMetrics)

    before, after, spectrogram, metrics_rows, status = demo.enhance_audio(
        config_path, checkpoint_path, input_path
    )

    assert Path(before) == input_path
    assert Path(after).is_file()
    assert Path(spectrogram).is_file()
    assert Path(spectrogram).suffix == ".png"
    assert "Enhanced audio saved" in status
    assert "DNSMOS skipped" in status
    assert ["sample_rate", "16000", "16000", "Hz"] in metrics_rows
    assert any(row[0] == "dnsmos_p835" and "skipped" in row[3] for row in metrics_rows)


def test_demo_keeps_uploaded_audio_gain_for_eval(tmp_path, monkeypatch, write_tone_wav):
    demo = _load_demo_module()
    config_path = _write_demo_config(tmp_path, gain_nomalized_to=-28)
    input_path = tmp_path / "input.wav"
    checkpoint_path = tmp_path / "exp" / "work" / "model.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text("ckpt", encoding="utf-8")
    write_tone_wav(input_path, sample_rate=16000, duration=0.1)
    open_calls = []
    original_open = demo.AudioIO.open

    class IdentityModel(torch.nn.Module):
        def forward(self, wav):
            return wav

    monkeypatch.setattr(
        demo,
        "get_cached_model",
        lambda config_path, checkpoint_path: (IdentityModel(), torch.device("cpu")),
    )
    monkeypatch.setattr(demo, "get_metrics_class", lambda: type(
        "FakeMetrics",
        (),
        {
            "noise_reduction": staticmethod(lambda noisy, enhanced: torch.tensor([0.0])),
            "dnsmos_p835": staticmethod(lambda *args, **kwargs: {}),
        },
    ))

    def wrapped_open(*args, **kwargs):
        open_calls.append(kwargs)
        return original_open(*args, **kwargs)

    monkeypatch.setattr(demo.AudioIO, "open", wrapped_open)

    demo.enhance_audio(config_path, checkpoint_path, input_path)

    assert open_calls[0]["target_lvl"] is None


def test_enhance_audio_reports_progress_steps(tmp_path, monkeypatch, write_tone_wav):
    demo = _load_demo_module()
    config_path = _write_demo_config(tmp_path)
    input_path = tmp_path / "input.wav"
    checkpoint_path = tmp_path / "exp" / "work" / "model.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text("ckpt", encoding="utf-8")
    write_tone_wav(input_path, sample_rate=16000, duration=0.1)
    events = []

    class IdentityModel(torch.nn.Module):
        def forward(self, wav):
            return wav

    class FakeMetrics:
        @staticmethod
        def noise_reduction(noisy, enhanced):
            return torch.tensor([0.0])

        @staticmethod
        def dnsmos_p835(*args, **kwargs):
            return {}

    class ProgressRecorder:
        def __call__(self, value, desc=None):
            events.append((value, desc))

    monkeypatch.setattr(
        demo,
        "get_cached_model",
        lambda config_path, checkpoint_path: (IdentityModel(), torch.device("cpu")),
    )
    monkeypatch.setattr(demo, "get_metrics_class", lambda: FakeMetrics)

    *_outputs, status = demo.enhance_audio(
        config_path, checkpoint_path, input_path, progress=ProgressRecorder()
    )

    descriptions = [desc for _value, desc in events]
    assert "Loading config" in descriptions
    assert "Running inference on cpu for 0.10s audio" in descriptions
    assert "Rendering spectrogram comparison" in descriptions
    assert descriptions[-1] == "Done"
    assert "Progress:" in status
    assert "- Computing no-reference metrics" in status


def test_run_demo_inference_returns_five_outputs_on_error(tmp_path):
    demo = _load_demo_module()

    outputs = demo.run_demo_inference(
        str(tmp_path / "missing.yaml"),
        str(tmp_path / "missing.ckpt"),
        str(tmp_path / "missing.wav"),
    )

    assert len(outputs) == 5
    assert outputs[1] is None
    assert outputs[2] is None
    assert outputs[3] == []
    assert outputs[4].startswith("Error:")


def test_enhance_audio_can_use_ort_streaming_backend(
    tmp_path, monkeypatch, write_tone_wav
):
    demo = _load_demo_module()
    config_path = _write_demo_config(tmp_path)
    input_path = tmp_path / "input.wav"
    onnx_path = tmp_path / "exp" / "work" / "model.onnx"
    onnx_path.parent.mkdir(parents=True)
    onnx_path.write_text("onnx", encoding="utf-8")
    write_tone_wav(input_path, sample_rate=16000, duration=0.1)

    class FakeRuntime:
        sample_rate = 16000
        providers = ["CPUExecutionProvider"]

        def process_samples(self, samples):
            return samples

        def flush(self):
            return demo.np.zeros(0, dtype="float32")

    monkeypatch.setattr(
        demo,
        "get_cached_ort_runtime",
        lambda onnx_path, provider="auto": FakeRuntime(),
    )

    class FakeMetrics:
        @staticmethod
        def noise_reduction(noisy, enhanced):
            return torch.tensor([0.0])

        @staticmethod
        def dnsmos_p835(*args, **kwargs):
            raise RuntimeError("dnsmos unavailable")

    monkeypatch.setattr(demo, "get_metrics_class", lambda: FakeMetrics)

    before, after, spectrogram, metrics_rows, status = demo.enhance_audio(
        config_path,
        onnx_path,
        input_path,
        backend="ORT streaming",
        ort_provider="cpu",
    )

    assert Path(before) == input_path
    assert Path(after).is_file()
    assert Path(spectrogram).is_file()
    assert "Running ORT streaming inference" in status
    assert ["sample_rate", "16000", "16000", "Hz"] in metrics_rows
