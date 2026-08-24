import math

import pytest
import torch

from puresound.nnet.dpcrn import DPCRN, DPRNNblock2D
from puresound.nnet.lobe.rnn import SingleRNN
from puresound.nnet.lobe.ssm import MambaInter


def _block(**kw):
    torch.manual_seed(0)
    return MambaInter(d_model=32, d_state=8, d_conv=4, expand=2, **kw).eval()


def test_shape_contract_matches_single_rnn():
    m = _block()
    x = torch.randn(3, 32, 40)
    y = m(x)
    assert y.shape == x.shape


def test_causal_future_cannot_touch_the_past():
    m = _block()
    x = torch.randn(2, 32, 60)
    y0 = m(x)
    x2 = x.clone()
    x2[..., 30:] += torch.randn_like(x2[..., 30:])
    y2 = m(x2)
    assert torch.allclose(y0[..., :30], y2[..., :30], atol=1e-6)
    assert not torch.allclose(y0[..., 30:], y2[..., 30:], atol=1e-3)


def test_step_matches_forward_bitwise_enough():
    """The deployment path is step(); it must reproduce the training-time scan."""
    m = _block()
    x = torch.randn(2, 32, 50)
    with torch.no_grad():
        ref = m(x)
        state = m.initial_stream_state(2)
        outs = []
        for t in range(50):
            y, state = m.step(x[:, :, t], state)
            outs.append(y)
        stream = torch.stack(outs, dim=-1)
    assert torch.allclose(ref, stream, atol=1e-4), float((ref - stream).abs().max())


def test_state_stays_fp32_under_bf16_autocast():
    """Same failure mode the VADHead EMA hit: tiny dt increments underflow in
    bf16. The scan must hold its state in fp32 regardless of autocast."""
    m = _block()
    x = torch.randn(1, 32, 8)
    state = m.initial_stream_state(1)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y, (conv_c, h) = m.step(x[:, :, 0], state)
    assert h.dtype == torch.float32
    assert torch.isfinite(y).all()


def test_parameter_budget_is_a_swap_not_an_upgrade():
    mamba = MambaInter(d_model=128, d_state=16, d_conv=4, expand=2)
    lstm = SingleRNN("LSTM", 128, 96, bidirectional=False)
    p_m = sum(p.numel() for p in mamba.parameters())
    p_l = sum(p.numel() for p in lstm.parameters())
    assert p_m < 1.5 * p_l, (p_m, p_l)


def test_dt_bias_init_lands_in_the_configured_range():
    m = _block(dt_min=0.001, dt_max=0.1)
    dt = torch.nn.functional.softplus(m.dt_proj.bias)
    assert float(dt.min()) >= 0.001 * 0.5
    assert float(dt.max()) <= 0.1 * 2.0


def test_block2d_flag_builds_both_and_rejects_junk():
    kw = dict(input_size=32, hidden_size=24, fused_type="film")
    b_l = DPRNNblock2D(**kw)
    b_m = DPRNNblock2D(**kw, inter_type="mamba", mamba_args={"d_state": 8})
    assert isinstance(b_l.inter_rnn, SingleRNN)
    assert isinstance(b_m.inter_rnn, MambaInter)
    with pytest.raises(ValueError, match="inter_type"):
        DPRNNblock2D(**kw, inter_type="s4")
    x = torch.randn(2, 32, 16, 20)   # [N, CH, C, T]
    assert b_m.eval()(x).shape == x.shape


def test_dpcrn_end_to_end_with_mamba_inter_via_recipe():
    """The real integration path: recipe dict -> init_siso_model -> forward.
    Same model block as the shipped v11b recipe, only inter_type flipped."""
    from puresound.config import load_recipe
    from puresound.recipes import init_siso_model
    recipe = load_recipe(
        "egs/voice_isolate/config/exp/train_dpcrn_v11b_compinv.yaml",
        expected_task="voice_isolation", expected_purpose="train",
    )
    model_dict = recipe.model
    model_dict["backbone"]["backbone_args"]["inter_type"] = "mamba"
    model_dict["backbone"]["backbone_args"]["mamba_args"] = {"d_state": 16}
    torch.manual_seed(0)
    m = init_siso_model(model_dict).eval()
    assert isinstance(m.backbone.dprnn_block1.inter_rnn, MambaInter)
    wav = torch.randn(1, 16000)
    with torch.no_grad():
        out = m(wav)
    # STFT framing truncates the tail (< one window) for lstm and mamba alike;
    # behavioural parity with the shipped variant is the honest contract.
    assert wav.shape[-1] - out.shape[-1] < 512
    assert torch.isfinite(out).all()


def test_training_backward_fits_in_memory_via_checkpointing():
    """The failure that killed the first v13 launch: a 6 s training batch
    stores every scan step for backward. The chunked+checkpointed scan must
    survive a realistic [batch*bins, T] shape with grad enabled."""
    m = MambaInter(d_model=64, d_state=16).train()
    x = torch.randn(66, 64, 300, requires_grad=True)   # scaled-down but same regime
    y = m(x)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_checkpointed_grads_match_plain_grads():
    torch.manual_seed(1)
    m = MambaInter(d_model=32, d_state=8)
    x = torch.randn(3, 32, 130)
    m.train()
    m(x).sum().backward()
    g_ckpt = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    m.eval()                       # eval path skips checkpointing
    with torch.enable_grad():
        m(x).sum().backward()
    g_plain = [p.grad.clone() for p in m.parameters()]
    for a, b in zip(g_ckpt, g_plain):
        assert torch.allclose(a, b, atol=1e-5), float((a - b).abs().max())
