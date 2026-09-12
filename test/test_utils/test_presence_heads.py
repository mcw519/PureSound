"""What must stay true about the presence heads (VADHead + EMA, background head).

Each guard was reverse-verified: the implementation was broken in the way the
test names and the failure confirmed.
"""
import io
import math

import pytest
import torch
from pydantic import ValidationError

from puresound.nnet.dpcrn import DPCRN
from puresound.nnet.lobe.heads import VADHead, VADHeadConfig


def head(**kw):
    kw.setdefault("enc_channels", 8)
    kw.setdefault("hidden", 8)
    kw.setdefault("kernel_t", 5)
    return VADHead(**kw)


def tiny_dpcrn(**kw):
    return DPCRN(input_dim=64, channels=(1, 8, 16), kernel_t=(2, 2), stride_t=(1, 1),
                 dilation_t=(1, 1), kernel_f=(5, 3), stride_f=(2, 2),
                 dilation_f=(1, 1), delay=(0, 0), rnn_hidden=16, **kw)


# ------------------------------------------------------------------ config


@pytest.mark.parametrize("kw", [
    dict(ema_taus_s=()),                # empty: say what you mean
    dict(ema_taus_s=(0.0,)),
    dict(ema_taus_s=(0.5, -1.0)),
    dict(frame_rate=0.0),
])
def test_config_rejects_incoherent_ema_settings(kw):
    with pytest.raises(ValidationError):
        VADHeadConfig(enabled=True, **kw)


def test_ema_off_is_the_legacy_architecture():
    """ema_taus_s None must build the exact pre-EMA module: same shapes, so a
    v6_gate-era checkpoint keeps loading bit-identically."""
    legacy = head()
    assert legacy.proj.in_features == 8
    again = head()
    buf = io.BytesIO()
    torch.save(legacy.state_dict(), buf)
    buf.seek(0)
    again.load_state_dict(torch.load(buf))


# ------------------------------------------------------------------ the EMA bank


def test_debiased_ema_of_a_constant_is_that_constant_from_frame_zero():
    """The zero initial state must not leak in as a warm-up dip."""
    h = head(ema_taus_s=(0.05, 4.0))
    x = torch.full((1, 8, 30), 3.0)
    bank = h._ema_bank(x)
    assert torch.allclose(bank, torch.full_like(bank, 3.0), atol=1e-5)


def test_ema_time_constants_are_seconds_not_frames():
    """A step reaches 1-1/e of its height after one tau at ANY frame rate."""
    target = 1.0 - math.exp(-1.0)
    for fps in (50.0, 100.0, 200.0):
        h = head(ema_taus_s=(0.5,), frame_rate=fps)
        n = int(2.0 * fps)
        x = torch.ones(1, 8, n)
        y = h._ema_bank(x)[0, 8, :]      # first EMA block, first channel
        # Debiasing rescales the step response; measure it as the ratio of the
        # biased response, which the closed form gives exactly.
        k = int(0.5 * fps) - 1
        biased = y[k] * (1.0 - (1.0 - h._alphas[0]) ** (k + 1))
        assert abs(float(biased) - target) < 0.02, (fps, float(biased))


def test_ema_is_not_clamped_to_one():
    """lfilter clamps to [-1, 1] BY DEFAULT and bottleneck features are not
    bounded; a reintroduced clamp silently flattens every loud frame."""
    h = head(ema_taus_s=(0.05,))
    x = torch.full((1, 8, 40), 5.0)
    bank = h._ema_bank(x)
    assert float(bank[:, 8:, :].max()) > 4.9


def test_ema_recurrence_survives_bf16_input():
    """The recurrence must run in float32: at tau 4 s the step is a = 0.0025,
    and a bf16 state swallows increments that small."""
    h = head(ema_taus_s=(4.0,))
    x = torch.linspace(0, 1, 600).view(1, 1, 600).expand(1, 8, 600)
    ref = h._ema_bank(x)[:, 8:, :]
    got = h._ema_bank(x.bfloat16())[:, 8:, :].float()
    assert torch.allclose(got, ref, atol=0.02), float((got - ref).abs().max())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reproduces a CUDA-only kernel assertion")
def test_ema_survives_bf16_autocast_on_cuda():
    """bf16-mixed training wraps the forward in autocast, which re-casts
    lfilter's internals to bf16 -- the CUDA kernel asserts fp32/fp64 and the
    whole run dies on step one. The bank must disable autocast around itself;
    a bare .float() is silently undone by the context."""
    h = head(ema_taus_s=(0.05, 4.0)).cuda()
    x = torch.randn(2, 8, 4, 60, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = h(x)
    assert torch.isfinite(y).all()


def test_head_is_causal():
    """Changing the future must not change the past, EMA bank included."""
    h = head(ema_taus_s=(0.05, 1.0))
    h.eval()
    x = torch.randn(1, 8, 4, 60)
    y1 = h(x)
    x2 = x.clone()
    x2[..., 40:] += 10.0
    y2 = h(x2)
    assert torch.allclose(y1[:, :40], y2[:, :40], atol=1e-5)
    assert not torch.allclose(y1[:, 40:], y2[:, 40:], atol=1e-3)


def test_gradient_flows_through_the_ema_bank():
    """The whole point of training-time heads is bottleneck pressure -- and the
    EMA path is the only route from a late output back to an early frame (the
    instant path reaches kernel_t-1 = 4 frames), so the long-range gradient is
    specifically the EMA's gradient. A detached bank zeroes it.
    """
    h = head(ema_taus_s=(0.25,))
    x = torch.randn(1, 8, 4, 60, requires_grad=True)
    h(x)[:, -1].sum().backward()
    early = float(x.grad[..., :10].abs().sum())
    assert early > 0.0, "no gradient reaches frames only the EMA connects"


# ------------------------------------------------------------------ backbone


def test_background_head_is_populated_when_enabled_and_none_when_off():
    m = tiny_dpcrn(vad_head={"enabled": True},
                   background_vad_head={"enabled": True})
    x = torch.randn(2, 1, 64, 12)
    m(x)
    assert m.last_vad_logits.shape == m.last_background_vad_logits.shape == (2, 12)
    m2 = tiny_dpcrn()
    m2(x)
    assert m2.last_vad_logits is None
    assert m2.last_background_vad_logits is None


def test_the_two_heads_are_separate_weights():
    """One module answering both questions would tie "user talking" to
    "bystander talking" -- the four-state table needs independent outputs."""
    m = tiny_dpcrn(vad_head={"enabled": True},
                   background_vad_head={"enabled": True})
    assert m.vad_head is not m.background_vad_head
    near_keys = {k for k, _ in m.vad_head.named_parameters()}
    bg_keys = {k for k, _ in m.background_vad_head.named_parameters()}
    assert near_keys == bg_keys                       # same shape family
    with torch.no_grad():
        for p in m.background_vad_head.parameters():
            p.add_(1.0)
    x = torch.randn(1, 1, 64, 12)
    m(x)
    assert not torch.allclose(m.last_vad_logits, m.last_background_vad_logits)


def test_background_loss_accepts_the_new_logits():
    from puresound.nnet.loss.vad import BackgroundVADHeadBCELoss

    m = tiny_dpcrn(background_vad_head={"enabled": True, "ema_taus_s": (0.05,)})
    x = torch.randn(2, 1, 64, 12)
    m(x)
    loss = BackgroundVADHeadBCELoss()(
        m.last_background_vad_logits,
        torch.randint(0, 2, (2, 12)).float(),
    )
    assert torch.isfinite(loss)


# ------------------------------------------------------------------ streaming


@pytest.mark.parametrize("taus", [None, (0.05,), (0.05, 0.25, 1.0, 4.0)])
def test_step_matches_forward_frame_by_frame(taus):
    """The streaming head must BE the offline head, not approximate it.

    This is the property the export rests on: a graph whose head drifts from the
    offline one produces a presence signal nothing has been benchmarked against.
    """
    torch.manual_seed(0)
    h = head(enc_channels=16, hidden=12, ema_taus_s=taus).eval()
    x = torch.randn(2, 16, 5, 60)
    with torch.no_grad():
        offline = h(x)
        state = h.initial_stream_state(2)
        got = []
        for t in range(x.shape[-1]):
            logit, state = h.step(x[..., t : t + 1], state)
            got.append(logit)
        streamed = torch.cat(got, dim=-1)
    assert torch.allclose(offline, streamed, atol=1e-6), \
        float((offline - streamed).abs().max())


def test_step_debias_needs_the_frame_count():
    """A frozen or missing count makes every EMA column the wrong size -- a
    level error the shapes cannot catch."""
    h = head(ema_taus_s=(1.0,)).eval()
    x = torch.randn(1, 8, 4, 1)
    s0 = h.initial_stream_state(1)
    with torch.no_grad():
        _, s1 = h.step(x, s0)
    assert float(s1[2]) == 1.0
    with torch.no_grad():
        _, s2 = h.step(x, s1)
    assert float(s2[2]) == 2.0


def test_step_ema_state_survives_a_bf16_graph():
    """tau 4 s steps by 0.0025, which a bf16 accumulator swallows entirely.

    Asserted on the VALUE, not the dtype: a dtype assertion passes as soon as
    something upstream casts, while the slow averages quietly freeze. Feeding
    bf16 frames must still track the float32 reference.
    """
    h = head(enc_channels=16, hidden=12, ema_taus_s=(4.0,)).eval()
    x = torch.randn(1, 16, 4, 400)

    def run(dtype):
        state = h.initial_stream_state(1)
        out = []
        with torch.no_grad():
            for t in range(x.shape[-1]):
                logit, state = h.step(x[..., t : t + 1].to(dtype), state)
                out.append(logit.float())
        return torch.cat(out, dim=-1), state[0].float()

    ref, ref_ema = run(torch.float32)
    got, got_ema = run(torch.bfloat16)
    # The EMA state itself is what a low-precision accumulator destroys.
    assert torch.allclose(got_ema, ref_ema, atol=0.05), \
        float((got_ema - ref_ema).abs().max())


def test_step_ema_stays_float32_even_in_a_bf16_model():
    """The dangerous case is a bf16-cast MODEL, not a bf16 caller.

    With bf16 weights the whole head runs bf16, and a state that follows the
    parameters would accumulate a 0.0025 step in bf16 -- the slow averages stop
    moving. The recurrence has to hold float32 whatever the module is cast to.
    """
    h = head(enc_channels=16, hidden=12, ema_taus_s=(4.0,)).eval().to(torch.bfloat16)
    state = h.initial_stream_state(1, dtype=torch.bfloat16)
    assert state[0].dtype is torch.float32, "EMA state followed the module dtype"
    x = torch.randn(1, 16, 4, 300).to(torch.bfloat16)
    with torch.no_grad():
        for t in range(x.shape[-1]):
            _, state = h.step(x[..., t : t + 1], state)
    assert state[0].dtype is torch.float32
    # And it must actually have integrated: a frozen bf16 state stays ~0.
    assert float(state[0].abs().max()) > 1e-3, float(state[0].abs().max())


def test_step_conv_cache_carries_left_context():
    """kernel_t-1 frames of context. A cache that is dropped turns the dwconv
    into a 1-frame conv and silently shortens the receptive field."""
    h = head(kernel_t=5, ema_taus_s=None).eval()
    s = h.initial_stream_state(1)
    assert s[1].shape == (1, 8, 4)
    with torch.no_grad():
        _, s2 = h.step(torch.randn(1, 8, 4, 1), s)
    assert s2[1].shape == (1, 8, 4)
