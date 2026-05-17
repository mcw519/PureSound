import json
import sys
from types import SimpleNamespace

import numpy as np
import torch

from puresound.streaming import (
    StreamingDparnOrt,
    load_streaming_dparn_model,
    validate_streaming_dparn_config,
)
from puresound.utils import load_hparam


def test_dparn_streaming_config_validates_current_recipe():
    config = load_hparam("egs/voice_isolate/config/dparn.yaml")
    manifest = validate_streaming_dparn_config(config)

    assert manifest["sample_rate"] == 16000
    assert manifest["fft_length"] == 512
    assert manifest["hop_length"] == 128
    assert manifest["freq_bins"] == 257


def test_dparn_streaming_frame_model_returns_frame_and_state():
    model = load_streaming_dparn_model("egs/voice_isolate/config/dparn.yaml")
    state = model.initial_state(batch_size=1)
    frame = torch.randn(1, 257, 2)

    enhanced, next_state = model.forward_frame(frame, state)

    assert enhanced.shape == frame.shape
    assert len(next_state.down_caches) == 5
    assert len(next_state.up_caches) == 5
    assert len(next_state.h_states) == 2
    assert next_state.down_caches[0].shape == (1, 2, 256, 1)


def test_streaming_ort_uses_cuda_then_cpu_when_available(tmp_path, monkeypatch):
    onnx_path, manifest_path = _write_fake_ort_files(tmp_path)

    class FakeSession:
        def __init__(self, path, providers):
            self.providers = providers

        def get_providers(self):
            return self.providers

        def run(self, output_names, inputs):
            return [inputs["noisy_frame"], inputs["state"]]

    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    runtime = StreamingDparnOrt(onnx_path, manifest_path, provider="cuda")

    assert runtime.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_streaming_ort_is_chunk_invariant_with_identity_session(tmp_path, monkeypatch):
    onnx_path, manifest_path = _write_fake_ort_files(tmp_path)

    class IdentitySession:
        def __init__(self, path, providers):
            self.providers = providers

        def get_providers(self):
            return self.providers

        def run(self, output_names, inputs):
            return [inputs["noisy_frame"], inputs["state"]]

    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=IdentitySession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    samples = np.linspace(-0.2, 0.2, 1700, dtype=np.float32)

    full = StreamingDparnOrt(onnx_path, manifest_path, provider="cpu")
    full_output = np.concatenate([full.process_samples(samples), full.flush()])

    chunked = StreamingDparnOrt(onnx_path, manifest_path, provider="cpu")
    parts = [
        chunked.process_samples(samples[:13]),
        chunked.process_samples(samples[13:701]),
        chunked.process_samples(samples[701:]),
        chunked.flush(),
    ]
    chunked_output = np.concatenate(parts)

    assert np.allclose(full_output, chunked_output, atol=1e-6)


def _write_fake_ort_files(tmp_path):
    onnx_path = tmp_path / "model.onnx"
    manifest_path = tmp_path / "model.json"
    onnx_path.write_bytes(b"fake")
    manifest = {
        "sample_rate": 16000,
        "fft_length": 512,
        "win_length": 512,
        "hop_length": 128,
        "freq_bins": 257,
        "state_input_names": ["state"],
        "state_output_names": ["next_state"],
        "input_names": ["noisy_frame", "state"],
        "output_names": ["enhanced_frame", "next_state"],
        "state_shapes": {"state": [1, 1, 1, 1]},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return onnx_path, manifest_path
