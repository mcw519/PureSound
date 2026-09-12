import importlib
import json
import sys
from pathlib import Path

import numpy as np


SDK_ROOT = Path(__file__).resolve().parents[2] / "sdk" / "python"


class IdentitySession:
    def __init__(self, providers=None):
        self.providers = providers or ["CPUExecutionProvider"]

    def get_providers(self):
        return self.providers

    def run(self, output_names, inputs):
        return [inputs["noisy_frame"], inputs["state"]]


def test_portable_sdk_imports_without_puresound(monkeypatch):
    monkeypatch.syspath_prepend(str(SDK_ROOT))
    before = {name for name in sys.modules if name == "puresound" or name.startswith("puresound.")}

    module = importlib.import_module("puresound_streaming")

    after = {name for name in sys.modules if name == "puresound" or name.startswith("puresound.")}
    assert hasattr(module, "PureSoundStreamingRuntime")
    assert hasattr(module, "StftFrameOrtProcessor")
    assert not hasattr(module, "DparnStreamingRuntime")
    assert after == before


def test_portable_sdk_processes_chunk_invariant_audio(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SDK_ROOT))
    runtime_module = importlib.import_module("puresound_streaming")
    onnx_path, manifest_path = _write_manifest(tmp_path)
    samples = np.linspace(-0.1, 0.1, 1400, dtype=np.float32)

    full = runtime_module.PureSoundStreamingRuntime(
        onnx_path, manifest_path, session=IdentitySession()
    )
    full_output = np.concatenate([full.process_samples(samples), full.flush()])

    chunked = runtime_module.PureSoundStreamingRuntime(
        onnx_path, manifest_path, session=IdentitySession()
    )
    parts = [
        chunked.process_samples(samples[:17]),
        chunked.process_samples(samples[17:333]),
        chunked.process_samples(samples[333:]),
        chunked.flush(),
    ]
    chunked_output = np.concatenate(parts)

    assert np.allclose(full_output, chunked_output, atol=1e-6)


def test_portable_sdk_int16_helpers(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SDK_ROOT))
    runtime_module = importlib.import_module("puresound_streaming")
    onnx_path, manifest_path = _write_manifest(tmp_path)
    runtime = runtime_module.PureSoundStreamingRuntime(
        onnx_path, manifest_path, session=IdentitySession()
    )

    pcm = np.zeros(600, dtype=np.int16)
    out = runtime.process_int16(pcm)
    tail = runtime.flush_int16()

    assert out.dtype == np.int16
    assert tail.dtype == np.int16


def test_portable_sdk_rejects_unknown_processor(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SDK_ROOT))
    runtime_module = importlib.import_module("puresound_streaming")
    onnx_path, manifest_path = _write_manifest(tmp_path, processor="future_model")

    try:
        runtime_module.PureSoundStreamingRuntime(
            onnx_path, manifest_path, session=IdentitySession()
        )
    except ValueError as exc:
        assert "Unsupported streaming processor" in str(exc)
        assert "stft_frame_ort" in str(exc)
    else:
        raise AssertionError("unknown processor should fail fast")


def _write_manifest(tmp_path, processor="stft_frame_ort"):
    onnx_path = tmp_path / "model.onnx"
    manifest_path = tmp_path / "model.json"
    onnx_path.write_bytes(b"fake")
    manifest = {
        "sample_rate": 16000,
        "model_type": "dparn_streaming_frame",
        "processor": processor,
        "fft_length": 512,
        "win_length": 512,
        "hop_length": 128,
        "freq_bins": 257,
        "state_input_names": ["state"],
        "state_output_names": ["next_state"],
        "output_names": ["enhanced_frame", "next_state"],
        "state_shapes": {"state": [1, 1, 1, 1]},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return onnx_path, manifest_path
