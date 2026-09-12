from pathlib import Path

import numpy as np
import pytest
import torch

from puresound.nnet.masker import Masker
from puresound.streaming import (
    load_streaming_dpcrn_model,
    validate_streaming_dpcrn_config,
)
from puresound.utils import load_hparam

_REPO_ROOT = Path(__file__).resolve().parents[2]
CAUSAL_CONFIG = str(_REPO_ROOT / "egs/voice_isolate/config/exp/train_dpcrn_wide_causal.yaml")
LOOKAHEAD_CONFIG = str(_REPO_ROOT / "egs/voice_isolate/config/exp/train_dpcrn_wide_antisup.yaml")


def _archivable(relative: str) -> Path:
    """A checkpoint or export that may have been archived out of the tree.

    The model zoo ships two voice-isolation versions; the rest moved to the
    gitignored `pretrained_ckpt/backup/` when the catalog was trimmed. They are
    still the right fixtures for these tests, so look for them in both places
    and let the caller skip if neither has the file.
    """

    root = Path(__file__).resolve().parents[2] / "egs/voice_isolate/pretrained_ckpt"
    in_tree = root / relative
    return in_tree if in_tree.is_file() else root / "backup" / relative


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
            # with heads the frame model returns (wav, head_logits, state)
            res = frame_model.forward_frame(tf[:, :, i, :], state)
            out, state = res[0], res[-1]
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


@pytest.mark.slow  # full offline-vs-ORT comparison; ~2.5 min each
def test_dpcrn_streaming_matches_offline_for_causal_model():
    """Causal recipe (delay=[0,0,0]): per-frame streaming == offline with ZERO net
    delay. Guards the block-step port and the transpose-conv bias fix in _up_step."""
    d, rel = _offline_vs_streaming_rel(CAUSAL_CONFIG)
    assert d == 0, f"causal model should stream with zero delay, got d={d}"
    assert rel < 1e-3, f"causal streaming != offline (rel={rel:.3e})"


@pytest.mark.slow  # full offline-vs-ORT comparison; ~2.5 min each
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


def test_dpcrn_streaming_matches_offline_for_mamba_inter():
    """inter_type=mamba: MambaInter.step() rides the (h, c) state ports, so the
    manifest layout is unchanged. Per-frame streaming must reproduce offline at
    the same bottleneck delay, warmup gate included."""
    config = str(_REPO_ROOT / "egs/voice_isolate/config/exp/train_dpcrn_v13_mambainter.yaml")
    model = load_streaming_dpcrn_model(config)
    assert model.is_lookahead and model.bottleneck_delay == 3
    d, rel = _offline_vs_streaming_rel(config)
    assert d == model.bottleneck_delay, f"expected delay {model.bottleneck_delay}, got d={d}"
    assert rel < 1e-3, f"mamba streaming != offline (rel={rel:.3e})"


# --------------------------------------------------------------------------- #
# Presence heads as streaming side information
# --------------------------------------------------------------------------- #


def _heads_frame_model():
    from puresound.streaming import load_streaming_dpcrn_model

    return load_streaming_dpcrn_model(
        _REPO_ROOT / "egs/voice_isolate/config/infer_dpcrn_heads.yaml",
        _archivable("dpcrn_v11_ep19.ckpt"),
    ).eval()


def _tone(seconds=6):
    L = seconds * 16000
    t = torch.arange(L, dtype=torch.float32) / 16000.0
    torch.manual_seed(0)
    return (0.3 * torch.sin(2 * np.pi * 220 * t) + 0.1 * torch.randn(L)).unsqueeze(0)


@pytest.mark.skipif(
    not (_archivable("dpcrn_v11_ep19.ckpt")).is_file(),
    reason="needs the v11 checkpoint",
)
def test_head_logits_stream_bit_exactly_at_the_algorithmic_delay():
    """The streamed head must equal the offline head, at `streaming_delay` lead.

    The heads read the bottleneck, which streaming computes `streaming_delay`
    frames AHEAD of the audio it emits -- so a logit describes a frame the
    output has not reached yet. Without the warm-up gate below this read 1.198
    and was still growing at frame 400, because the 4 s EMA integrates the
    causal down path's phantom startup frames forever.
    """
    fm = _heads_frame_model()
    sm = fm.system_model.eval()
    wav = _tone()
    with torch.no_grad():
        tf = sm.encoder(wav)
        feats_out, _ = sm.feats(tf)
        sm.backbone(feats_out)
        off_v = sm.backbone.last_vad_logits[0].numpy()
        off_b = sm.backbone.last_background_vad_logits[0].numpy()

        state = fm.initial_state(batch_size=1)
        sv, sb = [], []
        for i in range(tf.shape[2]):
            _, extras, state = fm.forward_frame(tf[:, :, i, :], state)
            sv.append(float(extras[0].reshape(-1)[0]))
            sb.append(float(extras[1].reshape(-1)[0]))
    d = fm.streaming_delay
    sv, sb = np.array(sv), np.array(sb)
    n = min(len(off_v), len(sv) - d)
    assert np.abs(off_v[:n] - sv[d : d + n]).max() < 1e-4
    assert np.abs(off_b[:n] - sb[d : d + n]).max() < 1e-4


@pytest.mark.skipif(
    not (_archivable("dpcrn_v11_ep19.ckpt")).is_file(),
    reason="needs the v11 checkpoint",
)
def test_enabling_the_heads_does_not_touch_the_audio():
    """Side information means side information: the enhanced frames must be
    byte-identical to the heads-disabled graph, or the export has quietly
    changed the system every benchmark measured."""
    from puresound.streaming import load_streaming_dpcrn_model

    ckpt = _archivable("dpcrn_v11_ep19.ckpt")
    wav = _tone(3)
    outs = []
    for cfg in ("config/infer_dpcrn.yaml", "config/infer_dpcrn_heads.yaml"):
        fm = load_streaming_dpcrn_model(
            _REPO_ROOT / "egs/voice_isolate" / cfg, ckpt).eval()
        sm = fm.system_model.eval()
        with torch.no_grad():
            tf = sm.encoder(wav)
            state = fm.initial_state(batch_size=1)
            frames = []
            for i in range(tf.shape[2]):
                r = fm.forward_frame(tf[:, :, i, :], state)
                frames.append(r[0])
                state = r[-1]
        outs.append(torch.cat(frames, dim=0))
    assert torch.equal(outs[0], outs[1]), \
        float((outs[0] - outs[1]).abs().max())
