"""What `AnchorInheritanceLoss` must be, independent of how it is written.

Five properties, each pinned because an earlier round in this repo lost time to
its absence:

1. The floored envelope `P(a)`. It is the definition of "deletion" for this
   round, so its values are a table, not an implementation detail. Included: the
   -30 dB floor (unfloored, the statistic is a phase/anti-correlation tail
   detector -- mean 14.89 against median 2.90 on the diagnosis's own rows) and
   the dead zone the softplus leaves behind.
2. Only the *difference* `Pbar_on - Qbar_pre` may appear. Absolute dB
   calibrations died twice here (across capture chains and across checkpoints of
   one run); relative forms survived twice.
3. Zero on a row that already keeps the new talker; positive on a row that hands
   the new talker the gain it had decided on for the previous one.
4. A batch with no eligible row returns a graph-carrying zero. A `None` or a NaN
   here severs a DDP job, as the background-VAD crash already did once.
5. The onset index the loss reads is `vad_target`'s own, on a *speed-perturbed*
   row -- the grid `vad_target` lives on is post-speed-perturbation, and a loss
   that framed on a pre-speed grid would be measuring the wrong 0.5 s.
"""

import math

import pytest
import torch

from puresound.audio.augmentation import AudioEffectAugmentor
from puresound.audio.vad import EnergyVADLabeler
from puresound.config.recipe import LossConfig
from puresound.nnet.loss import AnchorInheritanceLoss
from puresound.recipes import init_loss_func

SR = 16000
FRAME = 400
HOP = 160


# --------------------------------------------------------------------------- #
# 1. the floored envelope
# --------------------------------------------------------------------------- #


#: (achieved gain a, floored P in dB). Verified against the round's own numbers:
#: max|dP/da| on the *unclamped* envelope is 138.87 (not the 100.2 first
#: claimed, which is the value at a = 0), and the clamp -- not the softplus --
#: is what zeroes the gradient, from a <= -0.0261 down.
ENVELOPE = [
    (2.0, 0.0),           # ceiling: over-estimation is other losses' business
    (1.0, 0.0),
    (0.5, -6.02024),
    (0.31622777, -9.98914),
    (0.1, -19.05499),
    (0.0, -27.26589),     # a = 0 is NOT the floor: softplus(0)/16 = 0.0433
    (-0.05, -30.0),
    (-0.85, -30.0),       # the softplus dead zone, already floored
    (-1.0, -30.0),
    (-2.0, -30.0),
]


@pytest.mark.parametrize("ratio,expected_db", ENVELOPE)
def test_floored_envelope_matches_the_pre_registered_table(ratio, expected_db):
    loss = AnchorInheritanceLoss()
    got = loss.floored_gain_db(torch.tensor([ratio], dtype=torch.float64))
    assert got.item() == pytest.approx(expected_db, abs=1e-4)


def test_the_clamp_and_not_the_softplus_is_what_zeroes_the_gradient():
    """`softplus(16.)` moves the dead zone rather than removing it.

    Unclamped, dP/da at a = -1 is 0.0069 dB per unit a -- vanishing but not
    zero. The -30 dB clamp is the part that makes it exactly zero, and it bites
    at a = -0.0261, far earlier than the softplus does.
    """
    loss = AnchorInheritanceLoss()

    def slope(fn, a):
        x = torch.tensor([a], dtype=torch.float64, requires_grad=True)
        (grad,) = torch.autograd.grad(fn(x).sum(), x)
        return grad.item()

    def unclamped(x):
        pos = torch.nn.functional.softplus(x, beta=loss.softplus_beta)
        return 10.0 * torch.log10(pos.square() + loss.eps_db)

    assert slope(unclamped, -1.0) == pytest.approx(0.0069, abs=2e-4)

    grid = torch.linspace(-2.0, 2.0, 40001, dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(unclamped(grid).sum(), grid)
    assert grad.abs().max().item() == pytest.approx(138.87, abs=0.05)

    assert slope(loss.floored_gain_db, -0.05) == 0.0
    assert slope(loss.floored_gain_db, -1.0) == 0.0
    assert slope(loss.floored_gain_db, 1.0) == 0.0        # the 0 dB ceiling
    assert slope(loss.floored_gain_db, -0.026) != 0.0     # just inside the floor


# --------------------------------------------------------------------------- #
# a synthetic row whose P and Q are exactly what we asked for
# --------------------------------------------------------------------------- #


def _inverse_envelope(p_db: float) -> float:
    """The `a` whose floored P is `p_db` (valid strictly inside the clamps)."""
    pos = 10.0 ** (p_db / 20.0)
    return math.log(math.expm1(16.0 * pos)) / 16.0


def _row(
    a_onset: float,
    q_pre: float,
    *,
    onset_frame: int = 150,
    n_frames: int = 260,
    n_interferers: float = 1.0,
    background_before_onset: int | None = None,
    seed: int = 0,
):
    """One batch row with a_t == `a_onset` over O and q_t == `q_pre` over B.

    The construction is frame-exact rather than approximate:

    * the user's reference starts at sample `onset_frame * HOP`, so every frame
      in O (which starts at `onset_frame`) lies wholly inside it;
    * the pre-onset noise *stops* three frames before the onset, so no frame
      that survives the 25 dB energy rule straddles the boundary;
    * the enhanced signal is `q_pre * noise` before the onset and
      `a_onset * reference` from it on, so both inner-product ratios are exact.
    """
    generator = torch.Generator().manual_seed(seed)
    length = FRAME + HOP * (n_frames - 1)
    onset_sample = onset_frame * HOP
    noise_end = max(0, (onset_frame - 3) * HOP)

    reference = torch.zeros(1, length)
    reference[:, onset_sample:] = torch.randn(
        1, length - onset_sample, generator=generator
    ) * 0.1
    noise = torch.zeros(1, length)
    noise[:, :noise_end] = torch.randn(1, noise_end, generator=generator) * 0.05

    enhanced = torch.zeros(1, length)
    enhanced[:, :onset_sample] = q_pre * noise[:, :onset_sample]
    enhanced[:, onset_sample:] = a_onset * reference[:, onset_sample:]

    vad = torch.zeros(1, n_frames)
    vad[:, onset_frame:] = 1.0
    n_bg = onset_frame if background_before_onset is None else background_before_onset
    background = torch.zeros(1, n_frames)
    background[:, :n_bg] = 1.0

    batch = {
        "consistency_noise": noise,
        "background_vad_target": background,
        "n_interferers": torch.tensor([n_interferers]),
    }
    return enhanced, reference, batch, vad


def test_the_synthetic_row_delivers_the_scores_it_was_built_for():
    """The fixture is only useful if P and Q come out where they were placed."""
    loss = AnchorInheritanceLoss(margin_db=10.0)
    enhanced, reference, batch, vad = _row(0.5, 0.1)
    s = loss.row_scores(enhanced, reference, batch, vad)

    assert bool(s["eligible"][0])
    assert int(s["onset_frame"][0]) == 150
    assert int(s["n_onset_frames"][0]) == 50
    assert int(s["n_pre_frames"][0]) == 147          # frames 0..146 keep their noise
    assert s["pbar_on"][0].item() == pytest.approx(-6.02024, abs=1e-3)
    assert s["qbar_pre"][0].item() == pytest.approx(-19.05499, abs=1e-3)
    assert not s["qbar_pre"].requires_grad          # detached, by construction


# --------------------------------------------------------------------------- #
# 2. only the difference appears
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("offset_db", [2.0, 4.0, -3.0])
def test_hinge_is_invariant_under_a_common_db_offset(offset_db):
    """`P, Q -> P + b, Q + b` must leave the hinge untouched.

    This is the property that makes the term chain- and checkpoint-relative: an
    absolute dB anywhere in it would move with the offset.
    """
    loss = AnchorInheritanceLoss(margin_db=30.0)
    p_db, q_db = -8.0, -20.0

    def hinge(p, q):
        enhanced, reference, batch, vad = _row(
            _inverse_envelope(p), _inverse_envelope(q)
        )
        s = loss.row_scores(enhanced, reference, batch, vad)
        assert bool(s["eligible"][0])
        return s["hinge"][0].item()

    base = hinge(p_db, q_db)
    shifted = hinge(p_db + offset_db, q_db + offset_db)
    assert base > 0.0
    assert shifted == pytest.approx(base, abs=1e-5)


# --------------------------------------------------------------------------- #
# 3. the sign on the two rows that matter
# --------------------------------------------------------------------------- #


def test_zero_on_a_row_that_keeps_the_new_talker():
    """User kept at unit gain, previous background suppressed 19 dB: nothing to
    ask for, and the term must be silent rather than merely small."""
    loss = AnchorInheritanceLoss(margin_db=10.0)
    enhanced, reference, batch, vad = _row(1.0, 0.1)
    s = loss.row_scores(enhanced, reference, batch, vad)
    assert s["contrast"][0].item() == pytest.approx(19.05499, abs=1e-3)
    assert loss(enhanced, reference, batch, vad).item() == 0.0


def test_positive_on_an_inherited_deletion_row():
    """The new talker gets the gain the model had decided on for the previous
    one: same P as Q, so the contrast is zero and the hinge is the full margin."""
    loss = AnchorInheritanceLoss(margin_db=10.0)
    enhanced, reference, batch, vad = _row(0.1, 0.1)
    enhanced = enhanced.clone().requires_grad_(True)
    value = loss(enhanced, reference, batch, vad)
    assert value.item() == pytest.approx(10.0, abs=1e-3)
    assert value.requires_grad


def test_the_gradient_pushes_the_onset_up_and_never_the_background():
    """`Qbar_pre` is detached, so no gradient may reach the pre-onset frames --
    otherwise the hinge could be satisfied by suppressing the bystander less,
    which is the trade this term exists to avoid."""
    loss = AnchorInheritanceLoss(margin_db=10.0)
    enhanced, reference, batch, vad = _row(0.1, 0.1)
    enhanced = enhanced.clone().requires_grad_(True)
    loss(enhanced, reference, batch, vad).backward()

    onset_sample = 150 * HOP
    grad = enhanced.grad
    assert grad[:, :onset_sample].abs().max().item() == 0.0
    assert grad[:, onset_sample:].abs().max().item() > 0.0


# --------------------------------------------------------------------------- #
# 4. DDP safety
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(dict(onset_frame=10), id="onset_too_early"),
        pytest.param(dict(n_interferers=0.0), id="no_interferer"),
        pytest.param(dict(background_before_onset=0), id="no_interferer_speech_before"),
        pytest.param(dict(background_before_onset=40), id="only_0.4s_before"),
    ],
)
def test_a_batch_with_no_eligible_row_returns_a_graph_carrying_zero(kwargs):
    loss = AnchorInheritanceLoss(margin_db=10.0)
    enhanced, reference, batch, vad = _row(0.1, 0.1, **kwargs)
    enhanced = enhanced.clone().requires_grad_(True)

    s = loss.row_scores(enhanced, reference, batch, vad)
    assert not bool(s["eligible"].any())

    value = loss(enhanced, reference, batch, vad)
    assert value.item() == 0.0
    assert value.requires_grad and value.grad_fn is not None
    value.backward()
    assert enhanced.grad is not None
    assert torch.equal(enhanced.grad, torch.zeros_like(enhanced.grad))


def test_the_fallback_rule_is_reported_separately_and_never_used_by_forward():
    """0.5 s of interferer-before-onset is the pre-registered fallback. It is a
    *report*, not a silent widening: the trained rule stays the configured one
    (the anchor knee is at 1 s -- 0.5 s prefixes reach only -3.67/-4.14 dB
    against 1 s's -12.61/-9.72)."""
    loss = AnchorInheritanceLoss(margin_db=10.0)
    enhanced, reference, batch, vad = _row(0.1, 0.1, background_before_onset=70)
    s = loss.row_scores(enhanced, reference, batch, vad)
    assert not bool(s["eligible"][0])
    assert bool(s["eligible_fallback"][0])
    assert loss(enhanced, reference, batch, vad).item() == 0.0


def test_a_missing_background_target_is_all_silent_not_a_crash():
    """When no row in a batch carries background speech the collate emits no
    `background_vad_target` at all. That batch has no eligible row; it must not
    take the job down."""
    loss = AnchorInheritanceLoss(margin_db=10.0)
    enhanced, reference, batch, vad = _row(0.1, 0.1)
    enhanced = enhanced.clone().requires_grad_(True)
    batch.pop("background_vad_target")
    value = loss(enhanced, reference, batch, vad)
    assert value.item() == 0.0
    assert value.grad_fn is not None


def test_it_refuses_to_run_without_the_inputs_it_needs():
    loss = AnchorInheritanceLoss()
    enhanced, reference, batch, vad = _row(0.1, 0.1)
    with pytest.raises(ValueError, match="vad_target"):
        loss(enhanced, reference, batch, None)
    with pytest.raises(KeyError, match="consistency_noise"):
        loss(enhanced, reference, {k: v for k, v in batch.items()
                                   if k != "consistency_noise"}, vad)
    with pytest.raises(KeyError, match="n_interferers"):
        loss(enhanced, reference, {k: v for k, v in batch.items()
                                   if k != "n_interferers"}, vad)


def test_a_recipe_can_name_it():
    losses, weights = init_loss_func(
        [LossConfig(type="AnchorInheritanceLoss", weighted=0.25,
                    args={"margin_db": 6.0})]
    )
    assert isinstance(losses[0], AnchorInheritanceLoss)
    assert losses[0].margin_db == 6.0
    assert weights == [0.25]


# --------------------------------------------------------------------------- #
# 5. alignment on a speed-perturbed 30 s row
# --------------------------------------------------------------------------- #


def test_onset_index_agrees_with_vad_target_on_a_speed_perturbed_30s_row():
    """The label grid is post-speed-perturbation (`ns.py` clones `vad_reference`
    after the speed block), so the loss has to frame on the perturbed waveform
    and read the perturbed label. Both are checked: the loss's onset is the
    label's, and the label's onset is where the speed factor puts it.
    """
    speed = 1.05
    onset_s = 5.0
    length = int(30.0 * SR)
    generator = torch.Generator().manual_seed(3)

    reference = torch.zeros(1, length)
    onset_sample = int(onset_s * SR)
    # Modulated noise: a flat carrier would let a -40 dB relative threshold mark
    # the whole row active.
    body = torch.randn(1, length - onset_sample, generator=generator) * 0.1
    envelope = 0.5 + 0.5 * torch.sin(
        torch.linspace(0.0, 200.0, body.shape[-1])
    ).abs()
    reference[:, onset_sample:] = body * envelope
    noise = torch.randn(1, length, generator=generator) * 0.02

    augmentor = AudioEffectAugmentor()
    reference, _ = augmentor.sox_speed_perturbed(wav=reference, speed=speed, sr=SR)
    noise, _ = augmentor.sox_speed_perturbed(wav=noise, speed=speed, sr=SR)
    n = min(reference.shape[-1], noise.shape[-1])
    reference, noise = reference[..., :n], noise[..., :n]

    labeler = EnergyVADLabeler(frame_length=FRAME, hop_length=HOP)
    vad = labeler(reference).view(1, -1)
    label_onset = int(vad.argmax(-1).item())

    n_frames = vad.shape[-1]
    background = torch.zeros(1, n_frames)
    background[:, :label_onset] = 1.0
    batch = {
        "consistency_noise": noise,
        "background_vad_target": background,
        "n_interferers": torch.tensor([1.0]),
    }
    loss = AnchorInheritanceLoss(margin_db=10.0)
    s = loss.row_scores(reference * 0.5, reference, batch, vad)

    assert bool(s["eligible"][0])
    assert abs(int(s["onset_frame"][0]) - label_onset) <= 2
    # and the label itself moved with the speed change: 5 s / 1.05 = 4.762 s.
    expected = onset_s / speed * SR / HOP
    assert abs(label_onset - expected) <= 4
