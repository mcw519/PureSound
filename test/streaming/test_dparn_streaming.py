"""`puresound.streaming.dparn`: the per-frame DPARN against the offline model."""

import numpy as np
import pytest
import torch
import yaml

from puresound.nnet.masker import Masker
from puresound.streaming import load_streaming_dparn_model, validate_streaming_dparn_config

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


@pytest.fixture
def dparn_recipe(tmp_path):
    path = tmp_path / "dparn.yaml"
    path.write_text(yaml.safe_dump(MINIMAL_DPARN_CONFIG, sort_keys=False), encoding="utf-8")
    return path


def test_the_config_validates_and_a_frame_returns_a_frame_and_state(dparn_recipe):
    manifest = validate_streaming_dparn_config(MINIMAL_DPARN_CONFIG)
    assert manifest["sample_rate"] == 16000
    assert manifest["fft_length"] == 512
    assert manifest["hop_length"] == 128
    assert manifest["freq_bins"] == 257

    model = load_streaming_dparn_model(dparn_recipe)
    frame = torch.randn(1, 257, 2)
    enhanced, next_state = model.forward_frame(frame, model.initial_state(batch_size=1))
    assert enhanced.shape == frame.shape
    assert len(next_state.down_caches) == 5
    assert len(next_state.up_caches) == 5
    assert len(next_state.h_states) == 2
    assert next_state.down_caches[0].shape == (1, 2, 256, 1)


def _pack_frame(bf_frame: torch.Tensor) -> torch.Tensor:
    enhanced = bf_frame.squeeze(-1).permute(0, 2, 1).contiguous()
    real, imag = torch.chunk(enhanced, chunks=2, dim=-1)
    return torch.cat([real, imag], dim=-1).reshape(1, -1, 2)


@pytest.mark.slow  # full offline-vs-streaming comparison
def test_dparn_streaming_matches_offline_with_zero_delay(dparn_recipe):
    """DPARN has no look-ahead, so per-frame streaming must equal offline with
    zero net delay. Parity is a mathematical property of the port, so random
    weights suffice. Guards the transpose-conv bias handling in
    `StreamingFrameModelBase._up_step`, shared with DPCRN."""
    torch.manual_seed(0)
    frame_model = load_streaming_dparn_model(str(dparn_recipe)).eval()
    system_model = frame_model.system_model.eval()

    length = 2 * 16000
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
    delay, rel = best
    assert delay == 0, f"DPARN should stream with zero delay, got {delay}"
    assert rel < 1e-3, f"DPARN streaming != offline (rel={rel:.3e})"
