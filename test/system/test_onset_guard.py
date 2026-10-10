"""What must stay true about the inference-only onset guard.

Each test guards a property the mechanism's argument rests on. The two that
matter most are the ones a refactor would quietly break: the time constants are
in SECONDS (a per-frame reading closes the guard 100x too fast), and the offline
and streaming faces are the SAME arithmetic (a runtime that drifts from the
benchmark is not the system that was measured). Where it sits in the model's
forward is pinned in `test_siso.py`; the manifest round trip in the SDK tests.
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
    "call,match",
    [
        (lambda: OnsetGuard(t_arm_s=0.0), None),                  # arms instantly = no guard
        (lambda: OnsetGuard(t_arm_s=-1.0), None),
        (lambda: OnsetGuard(t_arm_s=math.inf), None),             # never hands over = no model
        (lambda: OnsetGuard(t_forget_s=0.0), None),               # re-arms every frame
        (lambda: OnsetGuard(t_forget_s=-1.0), None),
        (lambda: OnsetGuard(tau_up_s=0.0), None),
        (lambda: OnsetGuard(tau_dn_s=0.0), None),
        (lambda: OnsetGuard(tau_up_s=1.0, tau_dn_s=0.5), None),   # protects slower than it releases
        (lambda: OnsetGuard(margin_db=0.0), None),                # every frame is speech
        (lambda: OnsetGuard(margin_db=-3.0), None),
        (lambda: OnsetGuard(hangover_s=-0.1), None),
        (lambda: OnsetGuard(min_run_s=0.0), None),
        (lambda: OnsetGuard(snap=0.0), None),                     # nothing is ever bit-identical
        (lambda: OnsetGuard(floor_win_s=0.0), None),
        (lambda: OnsetGuard(floor_rise_db_per_s=0.0), None),      # a floor that never recovers
        (lambda: OnsetGuard(init_s=3.0, floor_win_s=2.0), None),  # init longer than the window
        (lambda: OnsetGuard(init_s=1.5, t_arm_s=1.0), None),      # init longer than arming
        # The streaming shortcut is exact only because arming cannot happen
        # during priming; at 1 fps the initialisation window is as long as the
        # arming window, so the streaming form would diverge from the offline one.
        (lambda: OnsetGuard(init_s=0.2, t_arm_s=0.3, min_run_s=0.01, tau_dn_s=2.0)
         .streaming_state(hop=16000), "priming shortcut"),
        (lambda: OnsetGuard().step(OnsetGuard().streaming_state(hop=HOP), np.zeros(HOP - 1)),
         "one hop"),
    ],
)
def test_incoherent_operating_points_and_frames_are_refused(call, match):
    with pytest.raises(ValueError, match=match):
        call()


# ------------------------------------------------------- the protected regime


def test_with_nothing_confirmed_the_output_is_the_input():
    """THE safety property: with nothing confirmed the model is not consulted.

    Not approximately -- a gain of 0.999 would mean every keep span on record
    differs from the number it was measured at -- and over every sample, even
    when the frame grid is shorter than the waveform. Bit-identity is a claim
    about the GAIN: out-of-range input is still clamped.
    """
    g = OnsetGuard()
    n = 400 * HOP + 37
    dry_np = quiet(401)[:n]
    assert np.all(g.frame_gain(dry_np, hop=HOP, sr=SR) == 1.0)

    dry = torch.from_numpy(dry_np).float().view(1, -1)
    out = g.apply(torch.randn(1, n) * 0.2, dry, hop=HOP, sr=SR)
    assert out.shape == dry.shape
    assert torch.equal(out, dry)

    hot = torch.full((1, 400 * HOP), 3.0)
    clamped = g.apply(torch.zeros_like(hot), hot, hop=HOP, sr=SR)
    assert float(clamped.max()) == 1.0
    assert not torch.equal(clamped, hot)


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


def test_it_arms_after_exactly_t_arm_plus_the_confirmation_delay():
    """And a talker shorter than t_arm never hands over at all."""
    g = OnsetGuard()
    f0 = 300                                       # burst starts on a frame boundary
    gain = g.frame_gain(np.concatenate([quiet(f0), burst(400)]), hop=HOP, sr=SR)
    k = arm_frame(g, f0, SR / HOP)
    assert np.all(gain[:k] == 1.0), (k, gain[k - 5:k + 3])
    assert gain[k] < 1.0

    short = np.concatenate([quiet(300), burst(60), quiet(300)])   # 0.6 s of speech
    assert np.all(OnsetGuard(t_arm_s=1.0).frame_gain(short, hop=HOP, sr=SR) == 1.0)

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
    """The contrast is the point: with re-arming disabled the second talker is
    attenuated from their first word, which is the deletion this exists for."""
    f0, gap = 300, 700                      # 7 s of floor > t_forget + release
    dry = np.concatenate([quiet(f0), burst(400), quiet(gap), burst(400)])
    f1 = f0 + 400 + gap

    g = OnsetGuard(t_forget_s=5.0)
    gain = g.frame_gain(dry, hop=HOP, sr=SR)
    assert gain[arm_frame(g, f0, SR / HOP)] < 1.0            # armed on the first
    assert gain[f1 - 1] == 1.0                               # forgotten by then
    k1 = arm_frame(g, f1, SR / HOP)
    assert np.all(gain[f1:k1] == 1.0), "the second onset was not protected"
    assert gain[k1] < 1.0                                    # and re-arms on time

    never = OnsetGuard(t_forget_s=math.inf)
    assert never.frame_gain(dry, hop=HOP, sr=SR)[f1] < 0.05  # still handed over
    assert never.as_manifest()["t_forget_s"] == math.inf


# ------------------------------------------------- offline == streaming


def test_apply_and_the_step_loop_are_bit_identical():
    """One arithmetic, two faces. A runtime that drifts from the benchmark is
    not the system the benchmark measured. The one non-causal part is the floor
    initialisation, which the streaming face pays for by staying dry -- exact by
    construction.
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

    # priming: loud from sample zero, and still dry until the floor is primed
    st = g.streaming_state(hop=HOP, sr=SR)
    loud = burst(50)
    for t in range(st.n_init):                     # priming call + n_init - 1
        gain, st = g.step(st, loud[t * HOP:(t + 1) * HOP])
        assert gain == 1.0, t
