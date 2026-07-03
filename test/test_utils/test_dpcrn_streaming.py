import numpy as np
import torch

from puresound.nnet.masker import Masker
from puresound.streaming import (
    load_streaming_dpcrn_model,
    validate_streaming_dpcrn_config,
)
from puresound.utils import load_hparam

CAUSAL_CONFIG = "egs/voice_isolate/config/train_dpcrn_wide_causal.yaml"
LOOKAHEAD_CONFIG = "egs/voice_isolate/config/train_dpcrn_wide_antisup.yaml"


def test_dpcrn_streaming_config_validates_causal_recipe():
    config = load_hparam(CAUSAL_CONFIG)
    manifest = validate_streaming_dpcrn_config(config)

    assert manifest["sample_rate"] == 16000
    assert manifest["fft_length"] == 512
    assert manifest["hop_length"] == 160
    assert manifest["freq_bins"] == 257
    assert manifest["feature_bins"] == 256


def test_dpcrn_streaming_frame_model_returns_frame_and_state():
    model = load_streaming_dpcrn_model(CAUSAL_CONFIG)
    state = model.initial_state(batch_size=1)
    frame = torch.randn(1, 257, 2)

    enhanced, next_state = model.forward_frame(frame, state)

    assert enhanced.shape == frame.shape
    assert len(next_state.down_caches) == 3
    assert len(next_state.up_caches) == 3
    assert len(next_state.h_states) == 2
    assert len(next_state.c_states) == 2
    assert next_state.down_caches[0].shape == (1, 2, 256, 1)


def _pack_frame(bf_frame: torch.Tensor) -> torch.Tensor:
    enhanced = bf_frame.squeeze(-1).permute(0, 2, 1).contiguous()
    real, imag = torch.chunk(enhanced, chunks=2, dim=-1)
    return torch.cat([real, imag], dim=-1).reshape(1, -1, 2)


def _offline_vs_streaming_rel(config: str, seconds: float = 4.0):
    """Return (best_delay, relative_error) between the full-utterance offline
    forward and the per-frame streaming forward. Parity is a mathematical property
    of the port, so random weights (no checkpoint) suffice."""
    torch.manual_seed(0)
    frame_model = load_streaming_dpcrn_model(config).eval()
    system_model = frame_model.system_model.eval()

    L = int(seconds * 16000)
    t = torch.arange(L, dtype=torch.float32) / 16000.0
    wav = (
        0.3 * torch.sin(2 * np.pi * 220 * t)
        + 0.2 * torch.sin(2 * np.pi * 700 * t)
        + 0.1 * torch.randn(L)
    ).unsqueeze(0)

    with torch.no_grad():
        tf = system_model.encoder(wav)
        feats_out, feats_enh = system_model.feats(tf)
        mask = system_model.backbone(feats_out)
        enh = Masker.apply_complex_mask_on_reim(feats_enh, mask)
        enh_bf = system_model.feats.back_forward(enh)
        T = enh_bf.shape[-1]
        offline = torch.cat([_pack_frame(enh_bf[..., i : i + 1]) for i in range(T)], dim=0).numpy()

        state = frame_model.initial_state(batch_size=1)
        stream_frames = []
        for i in range(tf.shape[2]):
            out, state = frame_model.forward_frame(tf[:, :, i, :], state)
            stream_frames.append(out)
        streaming = torch.cat(stream_frames, dim=0).numpy()

    warmup, tail = 120, 5
    best = None
    for d in range(0, 9):
        n = T - d
        a = streaming[d : d + n][warmup : n - tail]
        b = offline[:n][warmup : n - tail]
        rel = float(np.max(np.abs(a - b))) / (float(np.max(np.abs(b))) + 1e-9)
        if best is None or rel < best[1]:
            best = (d, rel)
    return best


def test_dpcrn_streaming_matches_offline_for_causal_model():
    """Causal recipe (delay=[0,0,0]): per-frame streaming == offline with ZERO net
    delay. Guards the block-step port and the transpose-conv bias fix in _up_step."""
    d, rel = _offline_vs_streaming_rel(CAUSAL_CONFIG)
    assert d == 0, f"causal model should stream with zero delay, got d={d}"
    assert rel < 1e-3, f"causal streaming != offline (rel={rel:.3e})"


def test_dpcrn_streaming_matches_offline_for_lookahead_model():
    """Look-ahead recipe (delay=[1,1,1], 3-frame): per-frame streaming reproduces
    offline exactly, delayed by the bottleneck latency (D=3). Guards the future
    -buffering path: RNN warmup gate, U-Net skip delay lines, and noisy-spectrum
    delay for mask application."""
    model = load_streaming_dpcrn_model(LOOKAHEAD_CONFIG)
    assert model.is_lookahead and model.bottleneck_delay == 3
    d, rel = _offline_vs_streaming_rel(LOOKAHEAD_CONFIG)
    assert d == model.bottleneck_delay, f"expected delay {model.bottleneck_delay}, got d={d}"
    assert rel < 1e-3, f"look-ahead streaming != offline (rel={rel:.3e})"
