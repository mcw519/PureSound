"""The auxiliary heads on the DPCRN bottleneck (`puresound.nnet.lobe.heads`) and
their losses: the presence head with its EMA bank, the background head, and the
distance head."""

import math

import pytest
import torch
from pydantic import ValidationError

from puresound.nnet.dpcrn import DPCRN
from puresound.nnet.lobe.heads import VADHead, VADHeadConfig
from puresound.nnet.loss import DistHeadRegressionLoss

BOTTLENECK = 16  # `tiny_dpcrn`'s channels[-1]


def head(**kw):
    kw.setdefault("enc_channels", 8)
    kw.setdefault("hidden", 8)
    kw.setdefault("kernel_t", 5)
    return VADHead(**kw)


def tiny_dpcrn(**kw):
    return DPCRN(input_dim=64, channels=(1, 8, BOTTLENECK), kernel_t=(2, 2),
                 stride_t=(1, 1), dilation_t=(1, 1), kernel_f=(5, 3), stride_f=(2, 2),
                 dilation_f=(1, 1), delay=(0, 0), rnn_hidden=16, **kw)


# ------------------------------------------------------------------ config


@pytest.mark.parametrize(
    "build,match",
    [
        (lambda: VADHeadConfig(enabled=True, ema_taus_s=()), None),  # say what you mean
        (lambda: VADHeadConfig(enabled=True, ema_taus_s=(0.0,)), None),
        (lambda: VADHeadConfig(enabled=True, ema_taus_s=(0.5, -1.0)), None),
        (lambda: VADHeadConfig(enabled=True, frame_rate=0.0), None),
        # `.get(key, default)` would answer a typo with the default: a misspelt
        # size trains a head of the wrong width, a misspelt gate leaves it off
        (lambda: tiny_dpcrn(vad_head={"enabled": True, "hiden": 64}), "hiden"),
        (lambda: tiny_dpcrn(vad_head={"enabld": True}), "enabld"),
        (lambda: tiny_dpcrn(vad_head={"enabled": True, "kernel_t": 0}), "kernel_t"),
    ],
)
def test_a_head_block_is_validated_not_defaulted(build, match):
    with pytest.raises(ValidationError, match=match):
        build()


def test_the_head_width_defaults_to_the_bottleneck():
    """The one default the config model cannot hold: only the backbone knows
    how wide its own bottleneck is, so the model says None and it decides."""
    assert tiny_dpcrn(vad_head={"enabled": True}).vad_head.proj.out_features == BOTTLENECK
    assert tiny_dpcrn(vad_head={"enabled": True, "hidden": 64}).vad_head.proj.out_features == 64
    assert head().proj.in_features == 8  # no EMA bank, no extra features


def test_a_disabled_or_absent_block_attaches_no_head():
    model = tiny_dpcrn(vad_head={"enabled": False})
    model(torch.randn(1, 1, 64, 12))
    assert model.vad_head is None and model.last_vad_logits is None
    model = tiny_dpcrn()
    model(torch.randn(1, 1, 64, 12))
    assert model.vad_head is None and model.background_vad_head is None
    assert model.dist_head is None and model.last_dist_preds is None


# ------------------------------------------------------------------ the EMA bank


def test_the_debiased_ema_of_a_constant_is_that_constant_unclamped():
    """The zero initial state must not leak in as a warm-up dip, and lfilter's
    default [-1, 1] clamp must not flatten a loud frame: bottleneck features
    are not bounded."""
    h = head(ema_taus_s=(0.05, 4.0))
    bank = h._ema_bank(torch.full((1, 8, 40), 5.0))
    assert torch.allclose(bank, torch.full_like(bank, 5.0), atol=1e-5)


def test_ema_time_constants_are_seconds_not_frames():
    """A step reaches 1-1/e of its height after one tau at ANY frame rate."""
    target = 1.0 - math.exp(-1.0)
    for fps in (50.0, 100.0, 200.0):
        h = head(ema_taus_s=(0.5,), frame_rate=fps)
        y = h._ema_bank(torch.ones(1, 8, int(2.0 * fps)))[0, 8, :]  # first EMA block
        # Debiasing rescales the step response; undo it with the closed form.
        k = int(0.5 * fps) - 1
        biased = y[k] * (1.0 - (1.0 - h._alphas[0]) ** (k + 1))
        assert abs(float(biased) - target) < 0.02, (fps, float(biased))


def test_the_ema_recurrence_runs_in_float32_whatever_the_dtype_around_it():
    """At tau 4 s the step is a = 0.0025, which a bf16 accumulator swallows.

    Asserted on the VALUE, not the dtype: a dtype assertion passes as soon as
    something upstream casts, while the slow averages quietly freeze. The
    dangerous streaming case is a bf16-cast MODEL, whose state would otherwise
    follow the parameters.
    """
    h = head(ema_taus_s=(4.0,))
    x = torch.linspace(0, 1, 600).view(1, 1, 600).expand(1, 8, 600)
    ref = h._ema_bank(x)[:, 8:, :]
    got = h._ema_bank(x.bfloat16())[:, 8:, :].float()
    assert torch.allclose(got, ref, atol=0.02), float((got - ref).abs().max())

    h = head(enc_channels=16, hidden=12, ema_taus_s=(4.0,)).eval()
    frames = torch.randn(1, 16, 4, 400)

    def run(module, dtype, state_dtype=torch.float32):
        state = module.initial_stream_state(1, dtype=state_dtype)
        with torch.no_grad():
            for t in range(frames.shape[-1]):
                _, state = module.step(frames[..., t : t + 1].to(dtype), state)
        return state[0]

    ref_ema = run(h, torch.float32)
    assert torch.allclose(run(h, torch.bfloat16).float(), ref_ema, atol=0.05)

    bf16_model = head(enc_channels=16, hidden=12, ema_taus_s=(4.0,)).eval().to(torch.bfloat16)
    state = run(bf16_model, torch.bfloat16, state_dtype=torch.bfloat16)
    assert state.dtype is torch.float32, "EMA state followed the module dtype"
    assert float(state.abs().max()) > 1e-3, "a frozen bf16 state stays ~0"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reproduces a CUDA-only kernel assertion")
def test_ema_survives_bf16_autocast_on_cuda():
    """bf16-mixed training wraps the forward in autocast, which re-casts
    lfilter's internals to bf16 -- the CUDA kernel asserts fp32/fp64. The bank
    must disable autocast around itself; a bare .float() is undone by the
    context."""
    h = head(ema_taus_s=(0.05, 4.0)).cuda()
    x = torch.randn(2, 8, 4, 60, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = h(x)
    assert torch.isfinite(y).all()


def test_the_head_is_causal_and_the_ema_carries_gradient_back():
    """Changing the future must not change the past, EMA bank included. And the
    EMA is the only route from a late output back to an early frame (the
    instant path reaches kernel_t-1 = 4 frames), so a detached bank would
    zero the long-range gradient the training-time head exists to provide."""
    h = head(ema_taus_s=(0.05, 1.0)).eval()
    x = torch.randn(1, 8, 4, 60)
    y1 = h(x)
    x2 = x.clone()
    x2[..., 40:] += 10.0
    y2 = h(x2)
    assert torch.allclose(y1[:, :40], y2[:, :40], atol=1e-5)
    assert not torch.allclose(y1[:, 40:], y2[:, 40:], atol=1e-3)

    h = head(ema_taus_s=(0.25,))
    x = torch.randn(1, 8, 4, 60, requires_grad=True)
    h(x)[:, -1].sum().backward()
    assert float(x.grad[..., :10].abs().sum()) > 0.0


@pytest.mark.parametrize("taus", [None, (0.05,), (0.05, 0.25, 1.0, 4.0)])
def test_step_matches_forward_frame_by_frame(taus):
    """The streaming head must BE the offline head, not approximate it -- the
    property the export rests on. Covers the conv cache (kernel_t - 1 frames of
    left context) and the frame count the EMA debias needs."""
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
    assert torch.allclose(offline, streamed, atol=1e-6), float((offline - streamed).abs().max())


# ------------------------------------------------------------------ on the backbone


def test_the_near_and_background_heads_are_separate_outputs_with_a_loss():
    """One module answering both questions would tie "user talking" to
    "bystander talking"; the four-state table needs independent outputs."""
    from puresound.nnet.loss.vad import BackgroundVADHeadBCELoss

    m = tiny_dpcrn(vad_head={"enabled": True},
                   background_vad_head={"enabled": True, "ema_taus_s": (0.05,)})
    assert m.vad_head is not m.background_vad_head
    with torch.no_grad():
        for p in m.background_vad_head.parameters():
            p.add_(1.0)
    m(torch.randn(2, 1, 64, 12))
    assert m.last_vad_logits.shape == m.last_background_vad_logits.shape == (2, 12)
    assert not torch.allclose(m.last_vad_logits, m.last_background_vad_logits)

    loss = BackgroundVADHeadBCELoss()(
        m.last_background_vad_logits, torch.randint(0, 2, (2, 12)).float()
    )
    assert torch.isfinite(loss)


def test_the_backbone_stashes_the_bottleneck_only_when_asked():
    """A reference held every training step would keep the graph alive; the
    presence gate is the only consumer, so it is off unless turned on."""
    model = tiny_dpcrn()
    x = torch.randn(2, 1, 64, 12)
    model(x)
    assert model.last_bottleneck is None
    model.stash_bottleneck = True
    model(x)
    assert model.last_bottleneck is not None
    assert model.last_bottleneck.dim() == 4
    assert not model.last_bottleneck.requires_grad  # detached, no graph held


def test_the_distance_head_trains_through_its_loss():
    model = tiny_dpcrn(dist_head={"enabled": True, "hidden": 8})
    y = model(torch.randn(2, 1, 64, 12))
    assert y.shape[0] == 2
    assert model.last_dist_preds.shape == (2, 3)

    loss = DistHeadRegressionLoss()(model.last_dist_preds, {
        "foreground_drr": torch.tensor([float("nan"), 5.0]),
        "foreground_distance": torch.tensor([0.74, 0.5]),
        "nearest_interferer_distance": torch.tensor([2.0, float("nan")]),
    })
    assert torch.isfinite(loss)
    loss.backward()
    assert model.dist_head.net[0].weight.grad.abs().sum() > 0


def test_the_distance_loss_masks_what_a_row_does_not_know():
    """NaN labels and missing keys contribute nothing; an all-unknown batch is a
    zero that still carries a graph, so DDP sees every parameter."""
    loss_fn = DistHeadRegressionLoss()

    # only the interferer distance is present: smooth_l1(0, log10(2)), beta=1
    loss = loss_fn(torch.zeros(2, 3), {"nearest_interferer_distance": torch.tensor([2.0, 2.0])})
    assert abs(float(loss) - 0.5 * math.log10(2.0) ** 2) < 1e-6

    nan = torch.full((3,), float("nan"))
    loss = loss_fn(torch.randn(3, 3, requires_grad=True), {
        "foreground_drr": nan, "foreground_distance": nan,
        "nearest_interferer_distance": nan,
    })
    assert float(loss) == 0.0 and loss.requires_grad

    with pytest.raises(ValueError):
        loss_fn(None, {})
