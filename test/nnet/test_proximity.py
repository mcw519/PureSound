"""What `RelativeProximityLoss` must be doing.

Two terms and one prohibition. The ordering term wants the user's turns to read
above a bystander's by a margin; the consistency term wants that *contrast* --
not the values -- to survive a change of capture chain. Nothing here may pin an
absolute value: fixed thresholds and metre readouts do not survive chain
offsets or checkpoint drift, so the head's scale is its own business and only
differences are supervised.

The per-frame readout is fed in directly rather than run through the head: what
is under test is the pooling, the pairing and the margin, and a synthetic
readout is the only way to state the expected number exactly.
"""

import pytest
import torch

from puresound.nnet.loss import RelativeProximityLoss

T = 30
SPANS = ((0, 10), (12, 20), (22, 30))  # turn 1, 2, 3
K = len(SPANS)
ROLES = (1, 2, 1)  # user, bystander, user


def readout(rows) -> torch.Tensor:
    """[B, T] per-frame proximity from a per-(row, turn) level."""
    out = torch.zeros(len(rows), T)
    for b, levels in enumerate(rows):
        for k, (start, end) in enumerate(SPANS):
            out[b, start:end] = levels[k]
    return out.requires_grad_(True)


def batch_of(rows, chains=None, source_ids=None, roles=ROLES, overlap=None) -> dict:
    n = len(rows)
    turn_id = torch.zeros(n, T, dtype=torch.long)
    for k, (start, end) in enumerate(SPANS):
        turn_id[:, start:end] = k + 1
    chains = list(range(n)) if chains is None else chains
    out = {
        "turn_id": turn_id,
        "turn_role": torch.tensor([list(roles)] * n),
        "turn_distance": torch.tensor([[0.5, 2.0, 0.5]] * n),
        "turn_speaker": torch.tensor([[1, 2, 1]] * n),
        "turn_chain": torch.tensor([[c] * K for c in chains]),
        "user_active": torch.zeros(n, T),
        "bystander_active": torch.zeros(n, T) if overlap is None else overlap,
    }
    if overlap is not None:
        out["user_active"] = torch.ones(n, T)
    if source_ids is not None:
        out["row_source_id"] = torch.tensor(source_ids)
    return out


# --------------------------------------------------------------------------- #
# the ordering term
# --------------------------------------------------------------------------- #


def test_the_ordering_term_rewards_the_user_above_the_bystander():
    """softplus is never exactly zero, so "correct ordering" means the term has
    decayed away: at 20 units past a margin of 1 it is 5.6e-9."""
    assert float(RelativeProximityLoss()(readout([[20.0, 0.0, 20.0]]), batch_of([None]))) < 1e-6
    assert float(RelativeProximityLoss()(readout([[0.0, 20.0, 0.0]]), batch_of([None]))) > 10.0


@pytest.mark.parametrize("margin", [1.0, 6.0])
def test_the_term_is_exactly_softplus_of_the_margin_minus_the_gap(margin):
    """Stated in closed form so a change to the pairing shows up as a number,
    not as a direction."""
    gap = 0.25
    value = RelativeProximityLoss(margin=margin, consistency_weight=0.0)(
        readout([[gap, 0.0, gap]]), batch_of([None])
    )
    expected = torch.nn.functional.softplus(torch.tensor(margin - gap))
    assert torch.allclose(value, expected, atol=1e-6)


def test_every_user_bystander_pair_is_scored_not_just_the_easiest():
    """Turn 3 is the user's quiet return. If only one pair were scored, raising
    turn 1 alone would buy the row out of the loss."""
    loud_first = RelativeProximityLoss(consistency_weight=0.0)(
        readout([[50.0, 0.0, 0.0]]), batch_of([None])
    )
    both = RelativeProximityLoss(consistency_weight=0.0)(
        readout([[50.0, 0.0, 50.0]]), batch_of([None])
    )
    assert float(loud_first) > 0.3
    assert float(both) < 1e-6


def test_the_gradient_reaches_the_frames_of_both_turns():
    prox = readout([[0.0, 0.0, 0.0]])
    RelativeProximityLoss(consistency_weight=0.0)(prox, batch_of([None])).backward()
    assert float(prox.grad[0, 0:10].abs().sum()) > 0.0     # user turn
    assert float(prox.grad[0, 12:20].abs().sum()) > 0.0    # bystander turn
    assert float(prox.grad[0, 10:12].abs().sum()) == 0.0   # the gap between them


# --------------------------------------------------------------------------- #
# the cross-chain consistency term
# --------------------------------------------------------------------------- #


AGREE = [[20.0, 0.0, 20.0], [23.0, 3.0, 23.0]]      # contrast 20 in both
DISAGREE = [[20.0, 0.0, 20.0], [8.0, 0.0, 8.0]]     # contrasts 20 and 8


@pytest.mark.parametrize(
    "rows, chains, source_ids, extra",
    [
        pytest.param(AGREE, [0, 1], [7, 7], 0.0, id="chains-agree"),
        pytest.param(DISAGREE, [0, 1], [7, 7], 12.0, id="chains-disagree"),
        # Two rows on one chain say nothing about a chain change; pairing them
        # would penalise ordinary between-row variation.
        pytest.param(DISAGREE, [3, 3], [7, 7], 0.0, id="same-chain"),
        pytest.param(DISAGREE, [0, 1], [7, 9], 0.0, id="different-source"),
        pytest.param(DISAGREE, [0, 1], [-1, -1], 0.0, id="no-source"),
        pytest.param(DISAGREE, [0, 1], [-1, 7], 0.0, id="one-without-source"),
        # The key is optional; its absence must not pair unrelated rows.
        pytest.param(DISAGREE, [0, 1], None, 0.0, id="no-row-source-id"),
    ],
)
def test_consistency_pays_the_contrast_difference_only_across_chains_of_one_source(
    rows, chains, source_ids, extra
):
    batch = batch_of(rows, chains=chains, source_ids=source_ids)
    with_consistency = RelativeProximityLoss(consistency_weight=1.0)(readout(rows), batch)
    without = RelativeProximityLoss(consistency_weight=0.0)(readout(rows), batch)
    assert torch.allclose(with_consistency - without, torch.tensor(extra), atol=1e-4)


# --------------------------------------------------------------------------- #
# overlap, rows without turns, misconfiguration
# --------------------------------------------------------------------------- #


def test_overlap_frames_do_not_enter_the_turn_means():
    overlap = torch.zeros(1, T)
    overlap[:, 6:10] = 1.0  # the tail of turn 1
    prox = torch.zeros(1, T)
    prox[:, 0:6] = 20.0
    prox[:, 6:10] = -500.0  # would drag turn 1's mean under the bystander
    prox[:, 12:20] = 0.0
    prox[:, 22:30] = 20.0
    prox.requires_grad_(True)

    flagged = RelativeProximityLoss()(prox, batch_of([None], overlap=overlap))
    assert float(flagged) < 1e-6
    ignored = RelativeProximityLoss()(prox, batch_of([None]))
    assert float(ignored) > 1.0


def _no_turns():
    batch = batch_of([None])
    batch["turn_id"] = torch.zeros_like(batch["turn_id"])
    return batch


@pytest.mark.parametrize(
    "batch",
    [
        pytest.param(lambda: {}, id="no-session-keys"),
        pytest.param(_no_turns, id="no-eligible-turn"),
        pytest.param(lambda: batch_of([None], roles=(1, 1, 1)), id="no-bystander-turn"),
    ],
)
def test_a_batch_with_nothing_to_order_is_a_graph_carrying_zero(batch):
    prox = readout([[1.0, 0.0, 1.0]])
    value = RelativeProximityLoss()(prox, batch())
    assert float(value) == 0.0
    assert value.requires_grad
    value.backward()
    assert prox.grad is not None


def test_the_inputs_it_declares_and_the_error_when_the_head_is_off():
    assert RelativeProximityLoss.required_inputs == ("proximity", "batch")
    with pytest.raises(ValueError, match="proximity_head"):
        RelativeProximityLoss()(None, batch_of([None]))
