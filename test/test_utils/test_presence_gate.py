"""What must stay true about the inference-only near-presence gain.

Each test here guards a property the mechanism's safety argument rests on, and
each one was checked by breaking the implementation and confirming it fails --
a test that cannot fail is not a guard.
"""
import math

import pytest
import torch

from puresound.system.presence_gate import PresenceGate


def gate(**kw):
    kw.setdefault("weight", torch.zeros(8))
    kw.setdefault("bias", 0.0)
    return PresenceGate(**kw)


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "kw",
    [
        dict(b_lo=0.6, b_hi=0.5),          # lo above hi
        dict(b_lo=0.5, b_hi=0.5),          # no ramp at all
        dict(b_lo=0.0, b_hi=0.5),          # lo at zero -> gain never reaches floor
        dict(b_hi=1.2),                    # above the range of b
        dict(gain_floor_db=0.0),           # a gain that does not attenuate
        dict(gain_floor_db=3.0),           # a gain that amplifies
        dict(tau_up_s=0.0),
        dict(tau_dn_s=-1.0),
        dict(b_init=1.5),
    ],
)
def test_rejects_incoherent_operating_points(kw):
    with pytest.raises(ValueError):
        gate(**kw)


def test_rejects_a_readout_that_is_not_a_vector():
    with pytest.raises(ValueError, match="1-D"):
        PresenceGate(weight=torch.zeros(2, 8), bias=0.0)


def test_rejects_a_readout_of_the_wrong_width():
    """A readout fitted on a different checkpoint must not silently broadcast."""
    g = gate(weight=torch.zeros(7))
    with pytest.raises(ValueError, match="wrong checkpoint"):
        g.presence(torch.zeros(1, 8, 4, 10))


def test_rejects_a_bottleneck_of_the_wrong_rank():
    with pytest.raises(ValueError, match=r"\[N, C, F, T\]"):
        gate().presence(torch.zeros(1, 8, 10))


# ---------------------------------------------------------------- dead zone


def test_dead_zone_gain_is_exactly_one():
    """THE safety property: above b_hi the output is bit-identical to not gating.

    Not approximately -- a gain of 0.999 would make every keep span on record
    differ from the number it was measured at.
    """
    g = gate(b_hi=0.5, b_lo=0.1)
    b = torch.tensor([[0.5, 0.5000001, 0.7, 1.0]])
    assert torch.equal(g.gain(b), torch.ones_like(b))


def test_apply_is_bit_identical_when_presence_is_saturated():
    """The same property end to end, through the waveform path."""
    g = gate(weight=torch.zeros(8), bias=20.0)      # sigmoid(20) ~ 1.0
    wav = torch.randn(1, 1600) * 0.1
    out = g.apply(wav, torch.randn(1, 8, 4, 10), hop=160)
    assert torch.equal(out, wav)


def test_gain_reaches_the_floor_and_no_further():
    g = gate(b_hi=0.5, b_lo=0.1, gain_floor_db=-26.0)
    floor = 10.0 ** (-26.0 / 20.0)
    b = torch.tensor([[0.1, 0.05, 0.0]])
    assert torch.allclose(g.gain(b), torch.full_like(b, floor))


def test_gain_is_monotone_in_presence():
    g = gate(b_hi=0.6, b_lo=0.2)
    b = torch.linspace(0, 1, 51).view(1, -1)
    out = g.gain(b)[0]
    assert torch.all(out[1:] >= out[:-1] - 1e-7)
    assert out[0] < out[-1]


# ---------------------------------------------------------------- integrator


def test_integrator_starts_from_its_prior():
    """b_init is what the mechanism believes before any evidence, and cold start
    is exactly the case where that matters."""
    g = gate(b_init=1.0, tau_dn_s=1.0)
    s = torch.zeros(1, 1)
    assert g.trajectory(s, 100.0)[0, 0] < 1.0            # it moved
    assert g.trajectory(s, 100.0)[0, 0] > 0.98           # only a little, in one frame


def test_integrator_rises_faster_than_it_falls():
    """The measured asymmetry, and the one that protects the user rather than
    cutting them off. A symmetric integrator must fail this."""
    g = gate(tau_up_s=0.05, tau_dn_s=1.0)
    up = g.trajectory(torch.ones(1, 20), 100.0)[0, -1] - g.b_init
    dn = g.trajectory(torch.zeros(1, 20), 100.0)[0, -1] - g.b_init
    assert up == 0.0                                      # already at 1.0
    g2 = gate(tau_up_s=0.05, tau_dn_s=1.0, b_init=0.5)
    rise = g2.trajectory(torch.ones(1, 10), 100.0)[0, -1] - 0.5
    fall = 0.5 - g2.trajectory(torch.zeros(1, 10), 100.0)[0, -1]
    assert rise > 4 * fall, (float(rise), float(fall))


def test_time_constants_are_in_seconds_not_frames():
    """One time constant should reach 1-1/e of the step regardless of frame rate."""
    g = gate(tau_dn_s=0.5, b_init=1.0)
    target = math.exp(-1.0)                               # 1 -> 0 after one tau
    for fps in (50.0, 100.0, 200.0):
        n = int(0.5 * fps)
        end = float(g.trajectory(torch.zeros(1, n), fps)[0, -1])
        assert abs(end - target) < 0.02, (fps, end)


def test_trajectory_is_per_row():
    """A batch must not leak state between items."""
    g = gate(b_init=1.0, tau_dn_s=0.1)
    s = torch.stack([torch.ones(30), torch.zeros(30)]).view(2, 30)
    b = g.trajectory(s, 100.0)
    assert b[0, -1] > 0.99 and b[1, -1] < 0.1


def test_rejects_a_nonpositive_frame_rate():
    with pytest.raises(ValueError, match="frame_rate"):
        gate().trajectory(torch.zeros(1, 4), 0.0)


# ---------------------------------------------------------------- apply


def test_apply_attenuates_when_presence_is_absent():
    g = gate(weight=torch.zeros(8), bias=-20.0, tau_dn_s=0.01, gain_floor_db=-26.0)
    wav = torch.ones(1, 1600) * 0.5
    out = g.apply(wav, torch.randn(1, 8, 4, 10), hop=160)
    # last frame is well past a 10 ms time constant, so it is at the floor
    assert float(out[0, -160:].abs().mean()) < 0.5 * 10 ** (-20 / 20)


def test_apply_covers_every_sample():
    """A frame grid shorter than the waveform must not leave a tail ungated."""
    g = gate(weight=torch.zeros(8), bias=-20.0, tau_dn_s=0.001)
    wav = torch.ones(1, 2000)                              # 12.5 frames of 160
    out = g.apply(wav, torch.randn(1, 8, 4, 10), hop=160)   # only 10 frames of gain
    assert out.shape == wav.shape
    assert float(out[0, -1].abs()) < 0.5


def test_apply_clamps():
    g = gate(weight=torch.zeros(8), bias=20.0)
    wav = torch.full((1, 320), 3.0)
    assert float(g.apply(wav, torch.randn(1, 8, 4, 2), hop=160).max()) == 1.0


def test_apply_consumes_no_randomness():
    """Every eval script seeds; a gate that drew would shift what came after."""
    g = gate(weight=torch.randn(8), bias=0.0)
    wav, bn = torch.randn(1, 1600), torch.randn(1, 8, 4, 10)
    torch.manual_seed(0)
    g.apply(wav, bn, hop=160)
    after = torch.rand(3)
    torch.manual_seed(0)
    assert torch.equal(torch.rand(3), after)


def test_manifest_records_the_operating_point():
    m = gate(b_hi=0.5, b_lo=0.1, gain_floor_db=-26.0).as_manifest()
    assert m["b_hi"] == 0.5 and m["b_lo"] == 0.1
    assert m["gain_floor_db"] == -26.0 and m["readout_dim"] == 8


# ------------------------------------------------- where it sits in the chain


def test_backbone_stashes_the_bottleneck_only_when_asked():
    """A reference held every training step would keep the graph alive; the gate
    is the only consumer, so it is off unless something turns it on."""
    from puresound.nnet.dpcrn import DPCRN

    model = DPCRN(input_dim=64, channels=(1, 8, 16), kernel_t=(2, 2), stride_t=(1, 1),
                  dilation_t=(1, 1), kernel_f=(5, 3), stride_f=(2, 2),
                  dilation_f=(1, 1), delay=(0, 0), rnn_hidden=16)
    x = torch.randn(2, 1, 64, 12)
    model(x)
    assert model.last_bottleneck is None
    model.stash_bottleneck = True
    model(x)
    assert model.last_bottleneck is not None
    assert model.last_bottleneck.dim() == 4
    assert not model.last_bottleneck.requires_grad     # detached, no graph held


def test_gate_is_applied_after_the_blend_so_it_can_beat_the_ceiling():
    """The ordering IS the mechanism.

    `dry_blend` 0.9 keeps 10% of the input, an arithmetic floor at -20 dB. Gate
    first and the blend puts that floor straight back:

        wrong:  0.9 * (g * enh) + 0.1 * mix   ->  |out| >= 0.1 * |mix|
        right:  g * (0.9 * enh + 0.1 * mix)   ->  |out| -> 0 as g -> 0

    Being able to go past -20 dB is the entire reason this is a gain and not a
    deeper blend, so a swap here silently reverts the mechanism to the one that
    was measured at a median 0.03 dB.
    """
    import numpy as np

    from puresound.config import load_recipe
    from puresound.recipes import init_siso_model

    recipe = load_recipe("egs/voice_isolate/config/infer_dpcrn.yaml",
                         expected_task="voice_isolation")
    model = init_siso_model(recipe.model).eval()        # random weights are fine
    rng = np.random.default_rng(0)
    wav = torch.from_numpy(rng.standard_normal((1, 16000)).astype("float32")) * 0.05

    # bias -30 => sigmoid ~ 0, so b falls and the gain reaches its floor
    g = PresenceGate(weight=torch.zeros(recipe.model["backbone"]["backbone_args"]["channels"][-1]),
                     bias=-30.0, b_hi=0.5, b_lo=0.1, gain_floor_db=-40.0,
                     tau_up_s=0.05, tau_dn_s=0.05)
    with torch.no_grad():
        out = model(wav.clone(), dry_blend=0.9, presence_gate=g)

    # The flag is a loan, not a setting: leave it on and every later forward --
    # including a training step -- keeps stashing a tensor nothing reads.
    assert model.backbone.stash_bottleneck is False

    n = min(out.shape[-1], wav.shape[-1])
    tail = slice(n // 2, n)                            # past the integrator settling
    level = 10 * math.log10(
        float(out[..., tail].square().mean()) / float(wav[..., tail].square().mean()))
    assert level < -21.0, (
        f"output is {level:.1f} dB below input; gate-before-blend would floor it at -20"
    )


def test_forward_without_a_gate_is_unchanged():
    """Every stage that does not pass the flag must be bit-identical to before."""
    import numpy as np

    from puresound.config import load_recipe
    from puresound.recipes import init_siso_model

    model = init_siso_model(load_recipe("egs/voice_isolate/config/infer_dpcrn.yaml",
                                        expected_task="voice_isolation").model).eval()
    rng = np.random.default_rng(1)
    wav = torch.from_numpy(rng.standard_normal((1, 8000)).astype("float32")) * 0.05
    with torch.no_grad():
        a = model(wav.clone(), dry_blend=0.9)
        b = model(wav.clone(), dry_blend=0.9, presence_gate=None)
    assert torch.equal(a, b)


# ------------------------------------------------- head-driven mode


def test_head_driven_gate_needs_no_readout():
    g = PresenceGate(b_hi=0.5, b_lo=0.1)
    assert g.head_driven is True
    assert g.as_manifest()["source"] == "head"
    assert g.as_manifest()["readout_dim"] is None


def test_head_driven_gate_refuses_a_bottleneck():
    """Silently returning 0.5 for everything would look like a working gate."""
    g = PresenceGate(b_hi=0.5, b_lo=0.1)
    with pytest.raises(ValueError, match="head-driven"):
        g.presence(torch.zeros(1, 8, 4, 10))


def test_apply_demands_exactly_one_source():
    g = gate()
    wav, bn = torch.randn(1, 1600), torch.randn(1, 8, 4, 10)
    with pytest.raises(ValueError, match="exactly one"):
        g.apply(wav, bn, hop=160, logits=torch.zeros(1, 10))
    with pytest.raises(ValueError, match="exactly one"):
        g.apply(wav, hop=160)


def test_head_logits_are_bit_identical_while_present():
    """The dead zone must survive the head path too -- that is the whole
    safety argument, and it cannot depend on which estimator feeds it.

    In-range audio on purpose: `apply` clamps to [-1, 1] like the module's own
    output does, so a fixture with samples past full scale would be changed by
    the clamp and not by the gate. See the companion test below.
    """
    g = PresenceGate(b_hi=0.5, b_lo=0.1, tau_up_s=0.05, tau_dn_s=1.0)
    wav = torch.randn(1, 100 * 160) * 0.1
    out = g.apply(wav, hop=160, logits=torch.full((1, 100), 8.0))
    assert torch.equal(out, wav)


def test_the_dead_zone_still_clamps_out_of_range_input():
    """Bit-identity is a claim about the GAIN, not about the clamp.

    Pinned because it is the obvious way to misread a failing bit-identity
    assertion: an over-full-scale fixture will differ, and that is the clamp
    doing its documented job, not the gate attenuating.
    """
    g = PresenceGate(b_hi=0.5, b_lo=0.1, tau_dn_s=1.0)
    hot = torch.full((1, 320), 3.0)
    out = g.apply(hot, hop=160, logits=torch.full((1, 2), 8.0))
    assert float(out.max()) == 1.0
    assert not torch.equal(out, hot)


def test_head_logits_reach_the_floor_given_time():
    """tau_dn is in seconds: absent evidence must actually close the gate, and
    a per-frame reading of tau would close it 100x too fast."""
    g = PresenceGate(b_hi=0.5, b_lo=0.1, gain_floor_db=-26.0, tau_dn_s=1.0)
    wav = torch.ones(1, 400 * 160)
    out = g.apply(wav, hop=160, logits=torch.full((1, 400), -8.0))
    tail_db = 20 * math.log10(float(out[0, -1].abs()))
    assert abs(tail_db - (-26.0)) < 0.5, tail_db
    # and it must NOT have closed within the first 100 ms
    early = 20 * math.log10(float(out[0, 10 * 160].abs()))
    assert early > -1.0, early


def test_head_and_readout_paths_agree_on_the_same_s():
    """Same presence sequence in, same gain out -- the estimator is swappable
    and the actuator is not doing anything different for it."""
    w = torch.zeros(8)
    g_read = PresenceGate(weight=w, bias=-8.0, b_hi=0.5, b_lo=0.1, tau_dn_s=1.0)
    g_head = PresenceGate(b_hi=0.5, b_lo=0.1, tau_dn_s=1.0)
    wav = torch.ones(1, 300 * 160)
    a = g_read.apply(wav, torch.zeros(1, 8, 4, 300), hop=160)
    b = g_head.apply(wav, hop=160, logits=torch.full((1, 300), -8.0))
    assert torch.allclose(a, b, atol=1e-6)
