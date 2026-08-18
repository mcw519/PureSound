import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from puresound.streaming import (
    StreamingDparnOrt,
    load_streaming_dparn_model,
    validate_streaming_dparn_config,
)

# Self-contained minimal DPARN recipe (streaming-compliant): only the fields that
# differ from puresound.nnet.dparn.DPARN's defaults are set explicitly (5 down
# layers / 2 dparn blocks / stride_t=dilation_t=1 all come from the class
# defaults). Kept inline rather than pointing at an egs/ recipe so this test
# doesn't depend on any specific recipe directory existing.
MINIMAL_DPARN_CONFIG = {
    "schema_version": 2,
    "purpose": "inference",
    "task": "noise_suppression",
    "dataset": {"target_sample_rate": 16000},
    "trainer": {"work_folder": "./exp"},
    "model": {
        "lightning_module": {
            "type": "EncDecMaskBase",
            "module_args": {"mask_type": "complex"},
        },
        "encoder": {
            "type": "ConvEncDec",
            "encoder_args": {
                "fft_length": 512,
                "win_type": "hann",
                "win_length": 512,
                "hop_length": 128,
                "fmin": 0,
                "fmax": 8000,
                "sr": 16000,
                "trainable": False,
            },
        },
        "features": {
            "feats_type": "complex",
            "drop_stft_first_bin": True,
            "trainable": False,
            "include_specaug": False,
        },
        "backbone": {
            "type": "DPARN",
            "backbone_args": {
                "input_dim": 256,
                "norm_type": "cLN",
                "channels": [2, 32, 32, 32, 64, 128],
            },
        },
    },
}


def test_dparn_streaming_config_validates_current_recipe():
    manifest = validate_streaming_dparn_config(MINIMAL_DPARN_CONFIG)

    assert manifest["sample_rate"] == 16000
    assert manifest["fft_length"] == 512
    assert manifest["hop_length"] == 128
    assert manifest["freq_bins"] == 257


def test_dparn_streaming_frame_model_returns_frame_and_state(tmp_path):
    config_path = tmp_path / "dparn.yaml"
    config_path.write_text(yaml.safe_dump(MINIMAL_DPARN_CONFIG), encoding="utf-8")

    model = load_streaming_dparn_model(config_path)
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


# --------------------------------------------------------------------------- #
# Offline vs streaming
# --------------------------------------------------------------------------- #


def _pack_frame(bf_frame: torch.Tensor) -> torch.Tensor:
    enhanced = bf_frame.squeeze(-1).permute(0, 2, 1).contiguous()
    real, imag = torch.chunk(enhanced, chunks=2, dim=-1)
    return torch.cat([real, imag], dim=-1).reshape(1, -1, 2)


def _offline_vs_streaming_rel(config_path, seconds: float = 2.0):
    """Return (best_delay, relative_error) between the full-utterance offline
    forward and the per-frame streaming forward.

    Parity is a mathematical property of the port, so random weights suffice --
    same construction as `test_dpcrn_streaming._offline_vs_streaming_rel`.
    """
    from puresound.nnet.masker import Masker

    torch.manual_seed(0)
    frame_model = load_streaming_dparn_model(str(config_path)).eval()
    system_model = frame_model.system_model.eval()

    length = int(seconds * 16000)
    t = torch.arange(length, dtype=torch.float32) / 16000.0
    wav = (
        0.3 * torch.sin(2 * np.pi * 220 * t)
        + 0.2 * torch.sin(2 * np.pi * 700 * t)
        + 0.1 * torch.randn(length)
    ).unsqueeze(0)

    with torch.no_grad():
        tf = system_model.encoder(wav)
        feats_out, feats_enh = system_model.feats(tf)
        enh = Masker.apply_complex_mask_on_reim(feats_enh, system_model.backbone(feats_out))
        enh_bf = system_model.feats.back_forward(enh)
        n_frames = enh_bf.shape[-1]
        offline = torch.cat(
            [_pack_frame(enh_bf[..., i : i + 1]) for i in range(n_frames)], dim=0
        ).numpy()

        state = frame_model.initial_state(batch_size=1)
        streamed = []
        for i in range(tf.shape[2]):
            out, state = frame_model.forward_frame(tf[:, :, i, :], state)
            streamed.append(out)
        streaming = torch.cat(streamed, dim=0).numpy()

    warmup, tail = 60, 5
    best = None
    for delay in range(0, 5):
        n = n_frames - delay
        a = streaming[delay : delay + n][warmup : n - tail]
        b = offline[:n][warmup : n - tail]
        rel = float(np.max(np.abs(a - b))) / (float(np.max(np.abs(b))) + 1e-9)
        if best is None or rel < best[1]:
            best = (delay, rel)
    return best


@pytest.mark.slow  # full offline-vs-streaming comparison
def test_dparn_streaming_matches_offline(tmp_path):
    """DPARN has no look-ahead, so per-frame streaming must equal offline with
    zero net delay.

    DPCRN has had this check since it was ported; DPARN never did, and that is
    why its `_up_step` double-counted the transpose-conv bias for as long as it
    did. Measured then: 1.153e-01 relative. The correction now lives in
    `StreamingFrameModelBase._up_step`, shared by both.
    """
    config = tmp_path / "dparn.yaml"
    config.write_text(yaml.safe_dump(MINIMAL_DPARN_CONFIG, sort_keys=False))

    delay, rel = _offline_vs_streaming_rel(config)
    assert delay == 0, f"DPARN should stream with zero delay, got {delay}"
    assert rel < 1e-3, f"DPARN streaming != offline (rel={rel:.3e})"
