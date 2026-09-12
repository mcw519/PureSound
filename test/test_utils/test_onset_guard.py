"""What must stay true about the inference-only onset guard.

Each test guards a property the mechanism's argument rests on. The two that
matter most are the ones a refactor would quietly break: the time constants are
in SECONDS (a per-frame reading closes the guard 100x too fast and the field
numbers would look the same), and the offline and streaming faces are the SAME
arithmetic (a runtime that drifts from the benchmark is not the system that was
measured).
"""
import math

import numpy as np
import pytest
import torch

from puresound.system.onset_guard import OnsetGuard

SR = 16000.0
HOP = 160


def quiet(n_frames: int, hop: int = HOP, seed: int = 0, amp: float = 1e-3) -> np.ndarray:
    """Stationary floor: never 8 dB over its own running minimum."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n_frames * hop) * amp).astype(np.float64)


def burst(n_frames: int, hop: int = HOP, seed: int = 1, over_db: float = 30.0,
          amp: float = 1e-3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n_frames * hop) * amp * 10.0 ** (over_db / 20.0))


def arm_frame(guard: OnsetGuard, first_burst_frame: int, fps: float) -> int:
    """The frame the guard must hand over on, from the knobs alone.

    Frame energy spans 20 ms, so the frame BEFORE the burst's first hop already
    sees half of it: raw activity starts at ``first_burst_frame - 1``. Then
    ``min_run_s`` is a confirmation delay and ``t_arm_s`` a continuous run.
    """
    n_minr = max(1, int(guard.min_run_s * fps))
    n_arm = max(1, int(round(guard.t_arm_s * fps)))
    return (first_burst_frame - 1) + (n_minr - 1) + (n_arm - 1)


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "kw",
    [
        dict(t_arm_s=0.0),                       # arms instantly = no guard
        dict(t_arm_s=-1.0),
        dict(t_arm_s=math.inf),                  # never hands over = no model
        dict(t_forget_s=0.0),                    # re-arms every frame
        dict(t_forget_s=-1.0),
        dict(tau_up_s=0.0),
        dict(tau_dn_s=0.0),
        dict(tau_up_s=1.0, tau_dn_s=0.5),        # protects slower than it releases
        dict(margin_db=0.0),                     # every frame is speech
        dict(margin_db=-3.0),
        dict(hangover_s=-0.1),
        dict(min_run_s=0.0),
        dict(snap=0.0),                          # nothing is ever bit-identical
        dict(floor_win_s=0.0),
        dict(floor_rise_db_per_s=0.0),           # a floor that can never recover
        dict(init_s=3.0, floor_win_s=2.0),       # init longer than the window
        dict(init_s=1.5, t_arm_s=1.0),           # init longer than arming
    ],
)
def test_rejects_incoherent_operating_points(kw):
    with pytest.raises(ValueError):
        OnsetGuard(**kw)


def test_defaults_are_the_measured_operating_point():
    g = OnsetGuard()
    assert (g.t_arm_s, g.t_forget_s, g.tau_dn_s, g.margin_db) == (1.0, 5.0, 2.0, 8.0)
    assert (g.floor_win_s, g.floor_rise_db_per_s, g.init_s) == (2.0, 3.0, 0.2)
    assert (g.hangover_s, g.min_run_s, g.tau_up_s) == (0.2, 0.10, 0.05)


def test_streaming_state_rejects_a_frame_rate_the_priming_shortcut_cannot_survive():
    """The shortcut is exact only because arming cannot happen during priming.

    At 1 fps the initialisation window is as long as the arming window, so the
    streaming form would return dry over a frame the offline form had handed
    over. Caught rather than silently divergent.
    """
    g = OnsetGuard(init_s=0.2, t_arm_s=0.3, min_run_s=0.01, tau_dn_s=2.0)
    with pytest.raises(ValueError, match="priming shortcut"):
        g.streaming_state(hop=16000)


def test_step_rejects_a_frame_that_is_not_one_hop():
    g = OnsetGuard()
    st = g.streaming_state(hop=HOP)
    with pytest.raises(ValueError, match="one hop"):
        g.step(st, np.zeros(HOP - 1))


# ------------------------------------------------------- the protected regime


def test_no_activity_ever_is_bit_identical_to_dry():
    """THE safety property: with nothing confirmed the model is not consulted.

    Not approximately -- a gain of 0.999 would mean every keep span on record
    differs from the number it was measured at.
    """
    g = OnsetGuard()
    dry_np = quiet(400)
    assert np.all(g.frame_gain(dry_np, hop=HOP, sr=SR) == 1.0)

    dry = torch.from_numpy(dry_np).float().view(1, -1)
    enh = torch.randn(1, dry.shape[-1]) * 0.2
    assert torch.equal(g.apply(enh, dry, hop=HOP, sr=SR), dry)


def test_the_protected_regime_still_clamps_out_of_range_input():
    """Bit-identity is a claim about the GAIN, not about the clamp."""
    g = OnsetGuard()
    dry = torch.full((1, 400 * HOP), 3.0)
    out = g.apply(torch.zeros_like(dry), dry, hop=HOP, sr=SR)
    assert float(out.max()) == 1.0
    assert not torch.equal(out, dry)


def test_apply_covers_every_sample():
    """A frame grid shorter than the waveform must not leave a tail ungated."""
    g = OnsetGuard()
    n = 400 * HOP + 37
    dry = torch.from_numpy(quiet(401)[:n]).float().view(1, -1)
    enh = torch.randn(1, n) * 0.2
    out = g.apply(enh, dry, hop=HOP, sr=SR)
    assert out.shape == dry.shape
    assert torch.equal(out, dry)


def test_apply_consumes_no_randomness():
    """Every eval script seeds; a guard that drew would shift what came after."""
    g = OnsetGuard()
    dry = torch.from_numpy(quiet(150)).float().view(1, -1)
    enh = torch.randn(1, dry.shape[-1]) * 0.2
    torch.manual_seed(0)
    g.apply(enh, dry, hop=HOP, sr=SR)
    after = torch.rand(3)
    torch.manual_seed(0)
    assert torch.equal(torch.rand(3), after)


# ------------------------------------------------------------------- arming


def test_arms_after_exactly_t_arm_plus_the_confirmation_delay():
    g = OnsetGuard()
    f0 = 300                                       # burst starts on a frame boundary
    dry = np.concatenate([quiet(f0), burst(400)])
    gain = g.frame_gain(dry, hop=HOP, sr=SR)
    k = arm_frame(g, f0, SR / HOP)
    assert gain[k - 1] == 1.0, (k, gain[k - 5:k + 3])
    assert gain[k] < 1.0
    # and it is the first frame that moves at all
    assert np.all(gain[:k] == 1.0)


def test_a_shorter_talker_than_t_arm_never_hands_over():
    """0.5 s of speech does nothing to the model's own anchor
    (`reference_matrix_README` §3), so it must not open the guard either."""
    g = OnsetGuard(t_arm_s=1.0)
    dry = np.concatenate([quiet(300), burst(60), quiet(300)])   # 0.6 s of speech
    assert np.all(g.frame_gain(dry, hop=HOP, sr=SR) == 1.0)


def test_time_constants_are_in_seconds_not_frames():
    """One tau must reach 1/e of the step whatever the frame rate.

    A per-frame reading of tau_dn would hand over in 20 ms instead of 2 s, which
    is the whole mechanism gone while every knob still reads 2.0.
    """
    g = OnsetGuard(tau_dn_s=0.5)
    for hop in (80, 160, 320):
        fps = SR / hop
        f0 = int(round(3.0 * fps))
        dry = np.concatenate([quiet(f0, hop), burst(int(round(6.0 * fps)), hop)])
        gain = g.frame_gain(dry, hop=hop, sr=SR)
        k = arm_frame(g, f0, fps)
        n_tau = int(round(g.tau_dn_s * fps))
        assert gain[k] == pytest.approx(math.exp(-1.0 / (g.tau_dn_s * fps)), abs=1e-9)
        end = float(gain[k + n_tau - 1])
        assert abs(end - math.exp(-1.0)) < 0.01, (hop, end)


def test_the_gain_snaps_to_exactly_zero_so_the_model_runs_untouched():
    """The other end of the dead zone: once handed over, the model's output is
    passed through bit for bit rather than at 0.999 of itself."""
    g = OnsetGuard()
    # 80 dB over the floor, so the floor tracker cannot climb into it before the
    # release has run its course -- see the test below on why that matters.
    dry = np.concatenate([quiet(300, amp=1e-4), burst(2000, over_db=80.0, amp=1e-4)])
    gain = g.frame_gain(dry, hop=HOP, sr=SR)
    assert gain[-1] == 0.0
    dryt = torch.from_numpy(dry).float().view(1, -1)
    enh = torch.randn(1, dryt.shape[-1]) * 0.2
    out = g.apply(enh, dryt, hop=HOP, sr=SR)
    tail = slice(-100 * HOP, None)
    assert torch.equal(out[..., tail], enh[..., tail])


def test_stationary_broadband_noise_is_absorbed_into_the_floor():
    """A property of the minimum-statistics tracker, pinned because it looks
    like a bug and is not.

    The floor may climb ``floor_rise_db_per_s``, so a *stationary* source only
    ``margin_db + 3*t`` dB over the floor stops being 'speech' after ~t seconds
    and, ``t_forget_s`` later, the guard protects again. That is the right answer
    for a noise-floor tracker -- sustained unmodulated broadband energy IS the
    floor -- and it is why the fixtures above use either a short burst or a very
    loud one. Real speech is modulated and stays over the floor.
    """
    g = OnsetGuard()
    gain = g.frame_gain(np.concatenate([quiet(300), burst(2500)]), hop=HOP, sr=SR)
    assert gain.min() == 0.0                 # it did hand over
    assert gain[-1] == 1.0                   # and the floor caught up
    mn = int(gain.argmin())
    back = (mn + int(np.flatnonzero(gain[mn:] == 1.0)[0])) / 100.0
    # earliest the mechanism allows: the climb eats the margin, then t_forget.
    # It runs later than that because the noise flickers over the threshold
    # while the floor is inside its own fluctuation band, which resets the
    # silence counter -- so this is a window, not a point.
    earliest = 3.0 + (30.0 - g.margin_db) / g.floor_rise_db_per_s + g.t_forget_s
    assert earliest <= back <= earliest + 5.0, back


# ------------------------------------------------------------------ re-arming


def test_after_t_forget_of_floor_the_next_onset_is_protected_again():
    g = OnsetGuard(t_forget_s=5.0)
    f0, gap = 300, 700                      # 7 s of floor > t_forget + release
    dry = np.concatenate([quiet(f0), burst(400), quiet(gap), burst(400)])
    gain = g.frame_gain(dry, hop=HOP, sr=SR)
    assert gain[arm_frame(g, f0, SR / HOP)] < 1.0            # armed on the first
    f1 = f0 + 400 + gap
    assert gain[f1 - 1] == 1.0                               # forgotten by then
    k1 = arm_frame(g, f1, SR / HOP)
    assert np.all(gain[f1:k1] == 1.0), "the second onset was not protected"
    assert gain[k1] < 1.0                                    # and re-arms on time


def test_t_forget_inf_reproduces_the_reference_sim_and_does_not_re_arm():
    """The contrast is the point: with re-arming disabled the second talker is
    attenuated from their first word, which is the deletion this exists for."""
    g = OnsetGuard(t_forget_s=math.inf)
    f0, gap = 300, 700
    dry = np.concatenate([quiet(f0), burst(400), quiet(gap), burst(400)])
    gain = g.frame_gain(dry, hop=HOP, sr=SR)
    f1 = f0 + 400 + gap
    # still handed over to the model when the second talker starts, where the
    # finite-t_forget guard above was back at exactly 1.0
    assert gain[f1] < 0.05, gain[f1]
    assert OnsetGuard(t_forget_s=5.0).frame_gain(dry, hop=HOP, sr=SR)[f1] == 1.0
    assert g.as_manifest()["t_forget_s"] == math.inf


# ------------------------------------------------- offline == streaming


def test_apply_and_the_step_loop_are_bit_identical():
    """One arithmetic, two faces. A runtime that drifts from the benchmark is
    not the system the benchmark measured.
    """
    g = OnsetGuard()
    rng = np.random.default_rng(7)
    dry = np.concatenate([quiet(250), burst(300), quiet(700), burst(300)])
    n_frames = int(math.ceil(dry.size / HOP))
    offline = g.frame_gain(dry, hop=HOP, sr=SR, n_frames=n_frames)

    padded = np.concatenate([dry, np.zeros((n_frames + 1) * HOP - dry.size)])
    st = g.streaming_state(hop=HOP, sr=SR)
    streamed = []
    for t in range(n_frames + 1):
        gain, st = g.step(st, padded[t * HOP:(t + 1) * HOP])
        if t >= 1:
            streamed.append(gain)
    streamed = np.array(streamed)
    assert streamed.shape == offline.shape
    assert np.array_equal(streamed, offline), np.abs(streamed - offline).max()

    # and through the waveform, on a random enh/dry pair
    dryt = torch.from_numpy(dry).float().view(1, -1)
    enh = torch.from_numpy(rng.standard_normal(dry.size) * 0.2).float().view(1, -1)
    gs = torch.from_numpy(streamed).float().repeat_interleave(HOP)[:dry.size].view(1, -1)
    want = torch.clamp(gs * dryt + (1.0 - gs) * enh, -1.0, 1.0)
    assert torch.equal(g.apply(enh, dryt, hop=HOP, sr=SR), want)


def test_the_streaming_face_holds_dry_while_it_primes_the_floor():
    """The one non-causal part is the 0.2 s floor initialisation; the streaming
    form pays for it by staying dry, which is exact by construction."""
    g = OnsetGuard()
    st = g.streaming_state(hop=HOP, sr=SR)
    dry = burst(50)                                # loud from sample zero
    for t in range(st.n_init):                     # priming call + n_init - 1
        gain, st = g.step(st, dry[t * HOP:(t + 1) * HOP])
        assert gain == 1.0, t


# ---------------------------------------------------------------- manifest


def test_manifest_round_trips():
    g = OnsetGuard(t_arm_s=1.5, t_forget_s=3.0, tau_dn_s=1.0, margin_db=6.0)
    m = g.as_manifest()
    assert m["t_arm_s"] == 1.5 and m["t_forget_s"] == 3.0
    assert m["tau_dn_s"] == 1.0 and m["margin_db"] == 6.0
    assert OnsetGuard.from_manifest(m) == g
    assert OnsetGuard.MANIFEST_KEY == "onset_guard"


# ------------------------------------------------- where it sits in the chain


def _infer_model():
    from puresound.config import load_recipe
    from puresound.recipes import init_siso_model

    return init_siso_model(
        load_recipe("egs/voice_isolate/config/infer_dpcrn.yaml",
                    expected_task="voice_isolation").model
    ).eval()


def test_forward_without_a_guard_is_unchanged():
    """Every stage that does not pass the flag must be bit-identical to before."""
    model = _infer_model()
    rng = np.random.default_rng(1)
    wav = torch.from_numpy(rng.standard_normal((1, 8000)).astype("float32")) * 0.05
    with torch.no_grad():
        a = model(wav.clone(), dry_blend=0.9)
        b = model(wav.clone(), dry_blend=0.9, onset_guard=None)
    assert torch.equal(a, b)


def test_the_guard_runs_after_the_presence_gate_so_it_can_restore_the_dry_input():
    """The ordering IS the mechanism.

    `PresenceGate` is a multiplicative gain, so it does not commute with a blend
    back to dry:

        wrong:  p * (g * mix + (1 - g) * enh)  ->  |out| = p * |mix| at g = 1
        right:  g * mix + (1 - g) * (p * enh)  ->  |out| = |mix| at g = 1

    Guard first and the presence gain attenuates the very signal the guard just
    restored -- here by 40 dB -- so the protection the field numbers were
    measured with would silently be worth nothing.
    """
    from puresound.system.presence_gate import PresenceGate

    model = _infer_model()
    rng = np.random.default_rng(0)
    # stationary floor: nothing for the guard to confirm, so it must stay dry.
    # Level matched to the presence-gate test's fixture so the gate's own
    # arithmetic is the one that was measured there.
    wav = torch.from_numpy((rng.standard_normal((1, 32000)) * 0.05).astype("float32"))
    channels = 128
    gate = PresenceGate(weight=torch.zeros(channels), bias=-30.0, b_hi=0.5,
                        b_lo=0.1, gain_floor_db=-40.0, tau_up_s=0.05, tau_dn_s=0.05)
    with torch.no_grad():
        gated = model(wav.clone(), dry_blend=0.9, presence_gate=gate)
        guarded = model(wav.clone(), dry_blend=0.9, presence_gate=gate,
                        onset_guard=OnsetGuard())
    assert model.backbone.stash_bottleneck is False
    n = min(guarded.shape[-1], wav.shape[-1])
    tail = slice(n // 2, n)

    def level(x):
        return 10 * math.log10(float(x[..., tail].square().mean())
                               / float(wav[..., tail].square().mean()))

    assert level(gated) < -30.0, level(gated)          # the gate really is closed
    assert torch.equal(guarded[..., :n], wav[..., :n])  # and the guard undoes it
    assert abs(level(guarded)) < 1e-6, level(guarded)
