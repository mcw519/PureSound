"""What `OverlapGating` promises, pinned one contract at a time.

The fingerprint proves the extraction from `ns.__getitem__` changed nothing.
These pin why it is allowed to change: which signals each regime may touch,
what a row that gates nothing costs, and that the number it reports is the
overlap that actually happened rather than the one that was drawn.
"""

import math
import random

import pytest
import torch

from puresound.config.augmentation import OverlapControlConfig
from puresound.task.overlap_gating import OverlapGating, _Envelope

SR = 16000
HOP = 160


class _Labeler:
    """Stands in for the energy VAD: the real one is a threshold on frame power
    and every test here needs to state the activity pattern, not discover it."""

    hop_length = HOP

    def __init__(self, active):
        self.active = active

    def __call__(self, wav, sample_rate):
        return self.active.float()


def _speech(n_frames=100):
    length = n_frames * HOP
    time = torch.arange(length, dtype=torch.float32) / SR
    return (0.3 * torch.sin(2 * torch.pi * 220.0 * time)).view(1, -1)


def _cfg(**overrides):
    base = dict(
        used=True,
        no_overlap_prob=0.0,
        high_overlap_prob=0.0,
        mid_overlap_range=[0.5, 0.5],
        high_overlap_range=[0.9, 1.0],
        fill_on_silent_range=[0.4, 0.4],
        fade_samples=64,
        turn_taking_prob=0.0,
        turn_near_seconds=[0.2, 0.2],
        turn_far_seconds=[0.2, 0.2],
        turn_gap_seconds=[0.0, 0.0],
        turn_overlap_seconds=[0.0, 0.0],
        far_first_prob=0.5,
    )
    base.update(overrides)
    return OverlapControlConfig(**base)


def _gating(active_frames=100, **overrides):
    active = torch.ones(active_frames, dtype=torch.bool)
    return OverlapGating(_cfg(**overrides), _Labeler(active))


def _draws(build, *, n_interferers=1):
    """The next value off the shared stream after one `apply`.

    Counted this way rather than by comparing audio: the point of the
    pass-through contract is that a row which gates nothing costs *nothing*,
    not merely that it produces the same signal.
    """
    torch.manual_seed(0)
    random.seed(0)
    build().apply(
        _speech(), [_speech() for _ in range(n_interferers)], sr=SR, target_mix=_speech()
    )
    return float(torch.rand(1))


def _nothing_at_all():
    torch.manual_seed(0)
    random.seed(0)
    return float(torch.rand(1))


# --------------------------------------------------------------------------- #
# What a row that gates nothing costs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,build,n_interferers",
    [
        ("block disabled", lambda: _gating(used=False), 1),
        ("no block at all", lambda: OverlapGating(None, _Labeler(torch.ones(8))), 1),
        ("no labeler", lambda: OverlapGating(_cfg(), None), 1),
        ("no interferers", lambda: _gating(), 0),
        (
            "silent target",
            lambda: OverlapGating(_cfg(), _Labeler(torch.zeros(100, dtype=torch.bool))),
            1,
        ),
    ],
)
def test_a_row_that_gates_nothing_consumes_no_randomness(name, build, n_interferers):
    """Otherwise turning gating off would shift every later stage's draw and an
    old recipe would stop regenerating what it used to."""
    assert _draws(build, n_interferers=n_interferers) == _nothing_at_all(), name


def test_a_row_that_gates_nothing_says_so_rather_than_reporting_zero():
    """NaN, not 0.0: eval skips a missing measurement and averages a zero."""
    result = _gating(used=False).apply(_speech(), [_speech()], sr=SR)
    assert math.isnan(result.overlap_fraction)
    assert result.turn_taking == 0.0


# --------------------------------------------------------------------------- #
# Which signals each regime may touch
# --------------------------------------------------------------------------- #


def test_bernoulli_gates_the_interferer_and_leaves_the_target_alone():
    """Only who talks *over* the target changes; the target is the reference."""
    target, mix, interferer = _speech(), _speech(), _speech()
    result = _gating().apply(
        target.clone(), [interferer.clone()], sr=SR, target_mix=mix.clone()
    )
    assert torch.equal(result.target, target)
    assert torch.equal(result.target_mix, mix)
    assert not torch.equal(result.interferers[0], interferer)
    assert result.turn_taking == 0.0


def test_turn_taking_gates_the_target_and_its_mix_with_the_same_envelope():
    """Gate the label without the mixture and the row claims the near speaker
    was talking through frames the mixture has silence in."""
    speech = _speech()
    result = _gating(turn_taking_prob=1.0).apply(
        speech.clone(), [speech.clone()], sr=SR, target_mix=speech.clone()
    )
    assert result.turn_taking == 1.0
    assert not torch.equal(result.target, speech)
    assert torch.equal(result.target, result.target_mix)


def test_a_target_absent_row_is_never_turn_taking_gated():
    """The foreground is subtracted from the mixture later using a pre-gating
    snapshot, so gating it here leaves a residual of the voice the row claims
    is not there."""
    target = _speech()
    result = _gating(turn_taking_prob=1.0).apply(
        target.clone(), [_speech()], sr=SR, allow_turn_taking=False
    )
    assert torch.equal(result.target, target)
    assert result.turn_taking == 0.0


def test_the_per_row_override_beats_the_block_rate():
    """Row types that want more far-solo stretches ask here, not in the config,
    so every row type that does not override stays bit-identical."""
    forced = _gating(turn_taking_prob=0.0).apply(
        _speech(), [_speech()], sr=SR, turn_taking_prob=1.0
    )
    suppressed = _gating(turn_taking_prob=1.0).apply(
        _speech(), [_speech()], sr=SR, turn_taking_prob=0.0
    )
    assert forced.turn_taking == 1.0
    assert suppressed.turn_taking == 0.0


# --------------------------------------------------------------------------- #
# The number the row reports about itself
# --------------------------------------------------------------------------- #


def test_the_reported_overlap_is_the_realized_one_not_the_drawn_one():
    """The drawn probability is not the answer: the target's own silences move
    the outcome, and eval buckets on this number."""
    # Half the frames active, and an interferer that fires on every frame it is
    # offered. Overlap is measured against the ACTIVE half only, so it is 1.0 --
    # a metric that counted the whole row would report 0.5.
    active = torch.zeros(100, dtype=torch.bool)
    active[:50] = True
    gating = OverlapGating(
        _cfg(mid_overlap_range=[1.0, 1.0], fill_on_silent_range=[1.0, 1.0]),
        _Labeler(active),
    )
    result = gating.apply(_speech(), [_speech()], sr=SR)
    assert result.overlap_fraction == pytest.approx(1.0)

    silent_interferer = OverlapGating(
        _cfg(mid_overlap_range=[0.0, 0.0], fill_on_silent_range=[1.0, 1.0]),
        _Labeler(active),
    )
    # Fires only in the target's silences: full activity, zero overlap.
    assert silent_interferer.apply(
        _speech(), [_speech()], sr=SR
    ).overlap_fraction == pytest.approx(0.0)


def test_the_gate_fades_rather_than_switching():
    """A frame mask flips in one sample, and a step in a speech waveform is a
    click. Every gate boundary is tapered instead."""
    fade = 64
    mask = torch.zeros(100, dtype=torch.bool)
    mask[50:] = True
    curve = _Envelope(HOP, fade).curve(mask, 100 * HOP)

    boundary = 50 * HOP
    edge = curve[boundary - fade : boundary + fade]
    # A hard switch has exactly two distinct values across the boundary; a
    # taper walks between them, monotonically.
    assert len(torch.unique(edge)) > 10
    assert bool((edge[1:] >= edge[:-1] - 1e-6).all())
    # ...and it is still a gate: fully closed before the fade, fully open after.
    # The last `fade` samples are excluded because the convolution zero-pads,
    # so every gated signal also tapers out over its own final 4 ms.
    assert float(curve[: boundary - fade].abs().max()) == pytest.approx(0.0)
    assert float(curve[boundary + fade : -fade].min()) == pytest.approx(1.0)


def test_the_turn_taking_coin_is_inside_the_short_circuit():
    """A row that cannot take the turn-taking branch must not pay for the coin.

    The three pass-through cases above return before the branch entirely, so
    none of them can see this: it is the *default* Bernoulli row, where gating
    runs but turn-taking is off, that would silently shift if the draw were
    hoisted above the `allow_turn_taking and prob > 0.0` guard. Pinned as an
    absolute count, because "off" and "did not trigger" both draw nothing and so
    cannot be told apart by comparing them to each other.

    With `no_overlap_prob` at 1.0 the Bernoulli path's consumption is exactly:
    the regime roll, the fill rate, and one value per frame.
    """
    n_frames, fill = 100, 0.4
    gating = OverlapGating(
        _cfg(no_overlap_prob=1.0, fill_on_silent_range=[fill, fill], turn_taking_prob=0.0),
        _Labeler(torch.ones(n_frames, dtype=torch.bool)),
    )

    torch.manual_seed(0)
    gating.apply(_speech(n_frames), [_speech(n_frames)], sr=SR)
    after_apply = float(torch.rand(1))

    torch.manual_seed(0)
    torch.rand(1)  # which overlap regime
    torch.empty(1).uniform_(fill, fill)  # rate for the target's silences
    torch.rand(n_frames)  # per-frame coin flips
    assert after_apply == float(torch.rand(1))
