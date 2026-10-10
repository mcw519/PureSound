"""`puresound.streaming.dpcrn`: the per-frame DPCRN against the offline model.

The streaming path reimplements the backbone's forward rather than calling it,
so every block variant needs its own offline-vs-streaming check: a branch added
to one and not the other exports a quietly different model. Parity is a
mathematical property of the port, so random weights (no checkpoint) suffice.
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from puresound.nnet.masker import Masker
from puresound.streaming import load_streaming_dpcrn_model, validate_streaming_dpcrn_config
from puresound.utils import load_hparam

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPES = REPO_ROOT / "test/fixtures/recipes"
INFER_CONFIG = REPO_ROOT / "egs/voice_isolate/config/infer_dpcrn.yaml"
HEADS_CONFIG = REPO_ROOT / "egs/voice_isolate/config/infer_dpcrn_heads.yaml"
CAUSAL_CONFIG = RECIPES / "dpcrn_causal.yaml"          # delay [0, 0, 0]
LOOKAHEAD_CONFIG = INFER_CONFIG                         # delay [1, 1, 1], the released model
MAMBA_CONFIG = RECIPES / "dpcrn_mamba_inter.yaml"      # inter_type mamba, look-ahead


def derived_config(tmp_path, name, **backbone_args):
    """The released model with its presence heads, backbone arguments replaced,
    written to disk."""
    config = copy.deepcopy(yaml.safe_load(HEADS_CONFIG.read_text()))
    config["model"]["backbone"]["backbone_args"].update(backbone_args)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def randomise_df_head(frame_model):
    """The deep-filter head's last layer starts at zero, which makes parity
    trivial -- the residual is zero on both sides. Give it real taps, and check
    they reach the output, or parity would pass for a broken port."""
    system = frame_model.system_model.eval()
    wav = 0.1 * torch.randn(1, 8000, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        before = system(wav)
        out = frame_model.backbone.df_head.out
        out.weight.normal_(0.0, 0.3)
        out.bias.normal_(0.0, 0.3)
        after = system(wav)
    assert (after - before).abs().max() > 1e-3, "the deep-filter residual does nothing"


def _pack_frame(bf_frame: torch.Tensor) -> torch.Tensor:
    enhanced = bf_frame.squeeze(-1).permute(0, 2, 1).contiguous()
    real, imag = torch.chunk(enhanced, chunks=2, dim=-1)
    return torch.cat([real, imag], dim=-1).reshape(1, -1, 2)


def _test_signal(seconds):
    length = int(seconds * 16000)
    t = torch.arange(length, dtype=torch.float32) / 16000.0
    return (
        0.3 * torch.sin(2 * np.pi * 220 * t)
        + 0.2 * torch.sin(2 * np.pi * 700 * t)
        + 0.1 * torch.randn(length)
    ).unsqueeze(0)


def offline_vs_streaming(frame_model, seconds):
    """(best_delay, relative_error) between the full-utterance offline forward
    and the per-frame streaming forward, over a warmed-up middle span."""
    system_model = frame_model.system_model.eval()
    wav = _test_signal(seconds)
    with torch.no_grad():
        tf = system_model.encoder(wav)
        feats_out, feats_enh = system_model.feats(tf)
        mask = system_model.backbone(feats_out)
        enh = Masker.apply_complex_mask_with_df(
            feats_enh, mask, getattr(system_model.backbone, "last_df_coefs", None)
        )
        enh_bf = system_model.feats.back_forward(enh)
        n_frames = enh_bf.shape[-1]
        offline = torch.cat(
            [_pack_frame(enh_bf[..., i : i + 1]) for i in range(n_frames)], dim=0
        ).numpy()

        state = frame_model.initial_state(batch_size=1)
        streamed = []
        for i in range(tf.shape[2]):
            # with heads the frame model returns (wav, head_logits, state)
            result = frame_model.forward_frame(tf[:, :, i, :], state)
            streamed.append(result[0])
            state = result[-1]
        streaming = torch.cat(streamed, dim=0).numpy()

    warmup, tail = 120, 5
    best = None
    for delay in range(0, 9):
        n = n_frames - delay
        a = streaming[delay : delay + n][warmup : n - tail]
        b = offline[:n][warmup : n - tail]
        rel = float(np.max(np.abs(a - b))) / (float(np.max(np.abs(b))) + 1e-9)
        if best is None or rel < best[1]:
            best = (delay, rel)
    return best


def test_the_config_validates_and_a_frame_returns_a_frame_and_state():
    manifest = validate_streaming_dpcrn_config(load_hparam(str(CAUSAL_CONFIG)))
    assert manifest["sample_rate"] == 16000
    assert manifest["fft_length"] == 512
    assert manifest["hop_length"] == 160
    assert manifest["freq_bins"] == 257
    assert manifest["feature_bins"] == 256

    model = load_streaming_dpcrn_model(CAUSAL_CONFIG)
    frame = torch.randn(1, 257, 2)
    enhanced, next_state = model.forward_frame(frame, model.initial_state(batch_size=1))
    assert enhanced.shape == frame.shape
    assert len(next_state.down_caches) == 3
    assert len(next_state.up_caches) == 3
    assert len(next_state.h_states) == 2
    assert len(next_state.c_states) == 2
    assert next_state.down_caches[0].shape == (1, 2, 256, 1)


def _causal(model):
    # zero net delay; guards the block-step port and the transpose-conv bias
    assert not model.is_lookahead and model.bottleneck_delay == 0


def _lookahead(model):
    # future buffering: RNN warm-up gate, U-Net skip delay lines, and the
    # noisy-spectrum delay for mask application
    assert model.is_lookahead and model.bottleneck_delay == 3


def _banded(model):
    # the recurrent state is one per frequency position the blocks see
    assert model.backbone.band_bottleneck is not None
    assert model.backbone.band_bottleneck.n_bands == 32


def _mamba_context(model):
    # only Mamba is banded; the intra path and decoder stay full-grid
    context = model.backbone.dprnn_block1.context_bottleneck
    assert context.n_units == 64 and context.n_bands == 8


def _attention_intra(model):
    assert model.backbone.dprnn_block1.intra_type == "attention"
    assert not hasattr(model.backbone.dprnn_block1, "intra_rnn")


def _deep_filter(model):
    # the filter's K - 1 frame history is streaming state; offline zero-pads it
    assert model.df_head is not None
    assert "df_cache" in model.state_input_names


DF_HEAD = {"bins": 128, "order": 5, "hidden": 32}

PARITY = [
    pytest.param(lambda tmp: CAUSAL_CONFIG, 4.0, None, _causal,
                 id="causal", marks=pytest.mark.slow),
    pytest.param(lambda tmp: LOOKAHEAD_CONFIG, 4.0, None, _lookahead,
                 id="lookahead", marks=pytest.mark.slow),
    # MambaInter.step() rides the (h, c) state ports, so the manifest layout
    # is unchanged
    pytest.param(lambda tmp: MAMBA_CONFIG, 4.0, None, _lookahead,
                 id="mamba-inter", marks=pytest.mark.slow),
    pytest.param(lambda tmp: derived_config(
        tmp, "banded", inter_type="lstm", stride_f=[2, 2, 1],
        band_bottleneck={"n_bands": 32, "scale": "erb", "sample_rate": 16000}),
        4.0, None, _banded, id="perceptual-banding", marks=pytest.mark.slow),
    pytest.param(lambda tmp: derived_config(
        tmp, "mamba_context", inter_type="mamba_context",
        mamba_args={"d_state": 8, "d_conv": 3, "expand": 1},
        mamba_context={"n_bands": 8, "scale": "erb", "sample_rate": 16000}),
        1.5, None, _mamba_context, id="mamba-context"),
    pytest.param(lambda tmp: derived_config(
        tmp, "attention", inter_type="lstm", intra_type="attention", intra_nhead=4),
        4.0, None, _attention_intra, id="attention-intra", marks=pytest.mark.slow),
    pytest.param(lambda tmp: derived_config(
        tmp, "df_causal", inter_type="lstm", delay=[0, 0, 0], df_head=DF_HEAD),
        1.5, randomise_df_head, _deep_filter, id="deep-filter-causal"),
    # across a look-ahead warm-up the filter history has to be built from the
    # delay line's aligned frames
    pytest.param(lambda tmp: derived_config(
        tmp, "df_lookahead", inter_type="lstm", delay=[1, 1, 1], df_head=DF_HEAD),
        1.5, randomise_df_head, _deep_filter, id="deep-filter-lookahead"),
]


@pytest.mark.parametrize("config,seconds,prepare,check", PARITY)
def test_streaming_matches_offline_at_the_bottleneck_delay(
    tmp_path, config, seconds, prepare, check
):
    torch.manual_seed(0)
    model = load_streaming_dpcrn_model(str(config(tmp_path))).eval()
    check(model)
    if prepare is not None:
        prepare(model)
    delay, rel = offline_vs_streaming(model, seconds)
    assert delay == model.bottleneck_delay, (
        f"expected delay {model.bottleneck_delay}, got {delay}"
    )
    assert rel < 1e-3, f"streaming != offline (rel={rel:.3e})"


@pytest.mark.slow
def test_the_presence_heads_stream_as_side_information(tmp_path):
    """The streamed head must equal the offline head at `streaming_delay` lead,
    and enabling the heads must leave the audio byte-identical.

    The heads read the bottleneck, which streaming computes `streaming_delay`
    frames AHEAD of the audio it emits, so a logit describes a frame the output
    has not reached yet. The causal down path emits phantom startup frames that
    a 4 s EMA would integrate forever without the warm-up gate. And side
    information means side information: an export whose audio moved with the
    heads has quietly changed the system every benchmark measured.
    """
    torch.manual_seed(0)
    heads = load_streaming_dpcrn_model(HEADS_CONFIG).eval()
    checkpoint = tmp_path / "weights.ckpt"
    torch.save({"state_dict": heads.system_model.state_dict()}, checkpoint)
    plain = load_streaming_dpcrn_model(INFER_CONFIG, checkpoint).eval()

    length = 4 * 16000
    t = torch.arange(length, dtype=torch.float32) / 16000.0
    wav = (0.3 * torch.sin(2 * np.pi * 220 * t) + 0.1 * torch.randn(length)).unsqueeze(0)

    system = heads.system_model.eval()
    with torch.no_grad():
        tf = system.encoder(wav)
        feats_out, _ = system.feats(tf)
        system.backbone(feats_out)
        offline_near = system.backbone.last_vad_logits[0].numpy()
        offline_bg = system.backbone.last_background_vad_logits[0].numpy()

        audio = {}
        near, bg = [], []
        for name, model in (("heads", heads), ("plain", plain)):
            state = model.initial_state(batch_size=1)
            frames = []
            for i in range(tf.shape[2]):
                result = model.forward_frame(tf[:, :, i, :], state)
                frames.append(result[0])
                state = result[-1]
                if name == "heads":
                    near.append(float(result[1][0].reshape(-1)[0]))
                    bg.append(float(result[1][1].reshape(-1)[0]))
            audio[name] = torch.cat(frames, dim=0)

    d = heads.streaming_delay
    near, bg = np.array(near), np.array(bg)
    n = min(len(offline_near), len(near) - d)
    assert n > heads.warmup_frames + 100
    assert np.abs(offline_near[:n] - near[d : d + n]).max() < 1e-4
    assert np.abs(offline_bg[:n] - bg[d : d + n]).max() < 1e-4
    assert torch.equal(audio["heads"], audio["plain"]), float(
        (audio["heads"] - audio["plain"]).abs().max()
    )
