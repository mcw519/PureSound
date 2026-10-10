"""What must stay true about the inference-only near-presence gain.

Each test here guards a property the mechanism's safety argument rests on. Where
it sits in the model's forward is pinned in `test_siso.py`.
"""
import math

import pytest
import torch

from puresound.system.presence_gate import PresenceGate


def gate(**kw):
    kw.setdefault("weight", torch.zeros(8))
    kw.setdefault("bias", 0.0)
    return PresenceGate(**kw)


BOTTLENECK = torch.zeros(1, 8, 4, 10)


@pytest.mark.parametrize(
    "call,match",
    [
        (lambda: gate(b_lo=0.6, b_hi=0.5), None),         # lo above hi
        (lambda: gate(b_lo=0.5, b_hi=0.5), None),         # no ramp at all
        (lambda: gate(b_lo=0.0, b_hi=0.5), None),         # gain never reaches the floor
        (lambda: gate(b_hi=1.2), None),                   # above the range of b
        (lambda: gate(gain_floor_db=0.0), None),          # does not attenuate
        (lambda: gate(gain_floor_db=3.0), None),          # amplifies
        (lambda: gate(tau_up_s=0.0), None),
        (lambda: gate(tau_dn_s=-1.0), None),
        (lambda: gate(b_init=1.5), None),
        (lambda: PresenceGate(weight=torch.zeros(2, 8), bias=0.0), "1-D"),
        # a readout fitted on a different checkpoint must not silently broadcast
        (lambda: gate(weight=torch.zeros(7)).presence(BOTTLENECK), "wrong checkpoint"),
        (lambda: gate().presence(torch.zeros(1, 8, 10)), r"\[N, C, F, T\]"),
        (lambda: gate().trajectory(torch.zeros(1, 4), 0.0), "frame_rate"),
        # silently returning 0.5 for everything would look like a working gate
        (lambda: PresenceGate(b_hi=0.5, b_lo=0.1).presence(BOTTLENECK), "head-driven"),
        (lambda: gate().apply(torch.randn(1, 1600), BOTTLENECK, hop=160,
                              logits=torch.zeros(1, 10)), "exactly one"),
        (lambda: gate().apply(torch.randn(1, 1600), hop=160), "exactly one"),
    ],
)
def test_incoherent_settings_and_inputs_are_refused(call, match):
    with pytest.raises(ValueError, match=match):
        call()


# ---------------------------------------------------------------- dead zone


def test_the_dead_zone_is_bit_identical_to_not_gating():
    """THE safety property: above b_hi the output is the input.

    Not approximately -- a gain of 0.999 would make every keep span on record
    differ from the number it was measured at. It must hold on the gain curve,
    through the readout path and through the head path alike. In-range audio on
    purpose: `apply` clamps to [-1, 1], which is a separate claim.
    """
    g = gate(b_hi=0.5, b_lo=0.1)
    b = torch.tensor([[0.5, 0.5000001, 0.7, 1.0]])
    assert torch.equal(g.gain(b), torch.ones_like(b))

    wav = torch.randn(1, 100 * 160) * 0.1
    saturated = gate(weight=torch.zeros(8), bias=20.0)  # sigmoid(20) ~ 1.0
    assert torch.equal(saturated.apply(wav, torch.randn(1, 8, 4, 100), hop=160), wav)

    head = PresenceGate(b_hi=0.5, b_lo=0.1, tau_up_s=0.05, tau_dn_s=1.0)
    assert torch.equal(head.apply(wav, hop=160, logits=torch.full((1, 100), 8.0)), wav)


def test_the_output_is_clamped_to_full_scale_even_in_the_dead_zone():
    """Bit-identity is a claim about the GAIN, not about the clamp: an
    over-full-scale fixture differs, and that is the clamp's documented job."""
    hot = torch.full((1, 320), 3.0)
    readout = gate(weight=torch.zeros(8), bias=20.0).apply(hot, torch.randn(1, 8, 4, 2), hop=160)
    head = PresenceGate(b_hi=0.5, b_lo=0.1, tau_dn_s=1.0).apply(
        hot, hop=160, logits=torch.full((1, 2), 8.0)
    )
    for out in (readout, head):
        assert float(out.max()) == 1.0
        assert not torch.equal(out, hot)


def test_the_gain_is_monotone_and_stops_at_the_floor():
    g = gate(b_hi=0.6, b_lo=0.2)
    out = g.gain(torch.linspace(0, 1, 51).view(1, -1))[0]
    assert torch.all(out[1:] >= out[:-1] - 1e-7)
    assert out[0] < out[-1]

    g = gate(b_hi=0.5, b_lo=0.1, gain_floor_db=-26.0)
    b = torch.tensor([[0.1, 0.05, 0.0]])
    assert torch.allclose(g.gain(b), torch.full_like(b, 10.0 ** (-26.0 / 20.0)))


# ---------------------------------------------------------------- integrator


def test_the_integrator_starts_at_its_prior_and_rises_faster_than_it_falls():
    """b_init is what the mechanism believes before any evidence, which is
    exactly the cold-start case. The asymmetry protects the user rather than
    cutting them off; a symmetric integrator fails the ratio."""
    g = gate(b_init=1.0, tau_dn_s=1.0)
    first = g.trajectory(torch.zeros(1, 1), 100.0)[0, 0]
    assert 0.98 < first < 1.0  # it moved, but only a little in one frame

    g = gate(tau_up_s=0.05, tau_dn_s=1.0, b_init=0.5)
    rise = g.trajectory(torch.ones(1, 10), 100.0)[0, -1] - 0.5
    fall = 0.5 - g.trajectory(torch.zeros(1, 10), 100.0)[0, -1]
    assert rise > 4 * fall, (float(rise), float(fall))


def test_the_integrator_runs_in_seconds_and_per_row():
    """One time constant reaches 1/e of the step whatever the frame rate, and a
    batch does not leak state between items."""
    g = gate(tau_dn_s=0.5, b_init=1.0)
    for fps in (50.0, 100.0, 200.0):
        end = float(g.trajectory(torch.zeros(1, int(0.5 * fps)), fps)[0, -1])
        assert abs(end - math.exp(-1.0)) < 0.02, (fps, end)

    g = gate(b_init=1.0, tau_dn_s=0.1)
    b = g.trajectory(torch.stack([torch.ones(30), torch.zeros(30)]).view(2, 30), 100.0)
    assert b[0, -1] > 0.99 and b[1, -1] < 0.1


# ---------------------------------------------------------------- apply


def test_absent_presence_closes_the_gate_over_every_sample():
    """A frame grid shorter than the waveform must not leave a tail ungated."""
    g = gate(weight=torch.zeros(8), bias=-20.0, tau_dn_s=0.01, gain_floor_db=-26.0)
    wav = torch.ones(1, 2000) * 0.5                          # 12.5 frames of 160
    out = g.apply(wav, torch.randn(1, 8, 4, 10), hop=160)   # only 10 frames of gain
    assert out.shape == wav.shape
    # well past a 10 ms time constant, so the tail is at the floor
    assert float(out[0, -160:].abs().mean()) < 0.5 * 10 ** (-20 / 20)


def test_head_logits_close_the_gate_on_the_seconds_scale():
    """Absent evidence must actually close the gate, and a per-frame reading of
    tau would close it 100x too fast."""
    g = PresenceGate(b_hi=0.5, b_lo=0.1, gain_floor_db=-26.0, tau_dn_s=1.0)
    out = g.apply(torch.ones(1, 400 * 160), hop=160, logits=torch.full((1, 400), -8.0))
    assert abs(20 * math.log10(float(out[0, -1].abs())) - (-26.0)) < 0.5
    assert 20 * math.log10(float(out[0, 10 * 160].abs())) > -1.0  # open at 100 ms


def test_head_and_readout_paths_agree_on_the_same_presence():
    """The estimator is swappable; the actuator does nothing different for it."""
    g_read = PresenceGate(weight=torch.zeros(8), bias=-8.0, b_hi=0.5, b_lo=0.1, tau_dn_s=1.0)
    g_head = PresenceGate(b_hi=0.5, b_lo=0.1, tau_dn_s=1.0)
    wav = torch.ones(1, 300 * 160)
    a = g_read.apply(wav, torch.zeros(1, 8, 4, 300), hop=160)
    b = g_head.apply(wav, hop=160, logits=torch.full((1, 300), -8.0))
    assert torch.allclose(a, b, atol=1e-6)


def test_apply_consumes_no_randomness():
    """Every eval script seeds; a gate that drew would shift what came after."""
    g = gate(weight=torch.randn(8), bias=0.0)
    wav, bn = torch.randn(1, 1600), torch.randn(1, 8, 4, 10)
    torch.manual_seed(0)
    g.apply(wav, bn, hop=160)
    after = torch.rand(3)
    torch.manual_seed(0)
    assert torch.equal(torch.rand(3), after)


def test_the_manifest_records_the_operating_point_and_its_source():
    m = gate(b_hi=0.5, b_lo=0.1, gain_floor_db=-26.0).as_manifest()
    assert m["b_hi"] == 0.5 and m["b_lo"] == 0.1
    assert m["gain_floor_db"] == -26.0 and m["readout_dim"] == 8

    head = PresenceGate(b_hi=0.5, b_lo=0.1)
    assert head.head_driven is True
    assert head.as_manifest()["source"] == "head"
    assert head.as_manifest()["readout_dim"] is None
