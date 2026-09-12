"""What `RelativeProximityLoss` must be doing (v20 R1a).

Two terms and one prohibition. The ordering term wants the user's turns to read
above a bystander's by a margin; the consistency term wants that *contrast* --
not the values -- to survive a change of capture chain. Nothing here may pin an
absolute value: this repo has three recorded deaths of fixed thresholds and
metres readouts on chain offsets and checkpoint drift, which is why the head's
scale is its own business and only differences are supervised.

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


def test_a_user_well_above_the_bystander_costs_nothing():
    """softplus is never exactly zero, so "correct ordering" means the term has
    decayed away: at 20 units past a margin of 1 it is 5.6e-9."""
    value = RelativeProximityLoss()(readout([[20.0, 0.0, 20.0]]), batch_of([None]))
    assert float(value) < 1e-6


def test_the_reversed_ordering_is_expensive():
    value = RelativeProximityLoss()(readout([[0.0, 20.0, 0.0]]), batch_of([None]))
    assert float(value) > 10.0


def test_the_term_is_exactly_softplus_of_the_margin_minus_the_gap():
    """Stated in closed form so a change to the pairing shows up as a number,
    not as a direction."""
    gap = 0.25
    value = RelativeProximityLoss(margin=1.0, consistency_weight=0.0)(
        readout([[gap, 0.0, gap]]), batch_of([None])
    )
    expected = torch.nn.functional.softplus(torch.tensor(1.0 - gap))
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


def test_the_margin_is_the_knob():
    prox = readout([[2.0, 0.0, 2.0]])
    tight = RelativeProximityLoss(margin=1.0, consistency_weight=0.0)(prox, batch_of([None]))
    wide = RelativeProximityLoss(margin=6.0, consistency_weight=0.0)(prox, batch_of([None]))
    assert float(wide) > float(tight)


def test_a_row_with_no_bystander_turn_contributes_nothing():
    single = RelativeProximityLoss()(
        readout([[0.0, 0.0, 0.0]]), batch_of([None], roles=(1, 1, 1))
    )
    assert float(single) == 0.0
    assert single.requires_grad


# --------------------------------------------------------------------------- #
# the cross-chain consistency term
# --------------------------------------------------------------------------- #


def test_two_chains_that_agree_on_the_contrast_pay_nothing_for_it():
    """Same source material, two different device chains, same contrast: the
    consistency term is exactly zero and the total is the ordering term alone."""
    rows = [[20.0, 0.0, 20.0], [23.0, 3.0, 23.0]]  # contrast 20 in both
    batch = batch_of(rows, chains=[0, 1], source_ids=[7, 7])
    with_consistency = RelativeProximityLoss(consistency_weight=1.0)(readout(rows), batch)
    without = RelativeProximityLoss(consistency_weight=0.0)(readout(rows), batch)
    assert torch.allclose(with_consistency, without, atol=1e-9)


def test_two_chains_that_disagree_pay_the_difference():
    rows = [[20.0, 0.0, 20.0], [8.0, 0.0, 8.0]]  # contrasts 20 and 8
    batch = batch_of(rows, chains=[0, 1], source_ids=[7, 7])
    without = RelativeProximityLoss(consistency_weight=0.0)(readout(rows), batch)
    with_consistency = RelativeProximityLoss(consistency_weight=1.0)(readout(rows), batch)
    assert torch.allclose(with_consistency - without, torch.tensor(12.0), atol=1e-4)


def test_the_same_chain_twice_is_not_a_cross_chain_pair():
    """The term's whole content is "a different chain must not change the
    contrast". Two rows on the same chain say nothing about that, and pairing
    them would penalise ordinary between-row variation."""
    rows = [[20.0, 0.0, 20.0], [8.0, 0.0, 8.0]]
    same_chain = batch_of(rows, chains=[3, 3], source_ids=[7, 7])
    value = RelativeProximityLoss(consistency_weight=1.0)(readout(rows), same_chain)
    without = RelativeProximityLoss(consistency_weight=0.0)(readout(rows), same_chain)
    assert torch.allclose(value, without, atol=1e-9)


def test_rows_from_different_source_material_are_not_paired():
    rows = [[20.0, 0.0, 20.0], [8.0, 0.0, 8.0]]
    for ids in ([7, 9], [-1, -1], [-1, 7]):
        batch = batch_of(rows, chains=[0, 1], source_ids=ids)
        value = RelativeProximityLoss(consistency_weight=1.0)(readout(rows), batch)
        without = RelativeProximityLoss(consistency_weight=0.0)(readout(rows), batch)
        assert torch.allclose(value, without, atol=1e-9), ids


def test_without_row_source_id_there_is_no_consistency_term():
    """The key is optional; its absence must not pair unrelated rows."""
    rows = [[20.0, 0.0, 20.0], [8.0, 0.0, 8.0]]
    batch = batch_of(rows, chains=[0, 1])
    value = RelativeProximityLoss(consistency_weight=1.0)(readout(rows), batch)
    without = RelativeProximityLoss(consistency_weight=0.0)(readout(rows), batch)
    assert torch.allclose(value, without, atol=1e-9)


# --------------------------------------------------------------------------- #
# overlap, today's rows, misconfiguration
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


def test_a_batch_without_the_session_keys_is_a_graph_carrying_zero():
    prox = readout([[1.0, 0.0, 1.0]])
    value = RelativeProximityLoss()(prox, {})
    assert float(value) == 0.0
    assert value.requires_grad
    value.backward()
    assert prox.grad is not None


def test_no_eligible_turn_is_a_graph_carrying_zero():
    batch = batch_of([None])
    batch["turn_id"] = torch.zeros_like(batch["turn_id"])
    value = RelativeProximityLoss()(readout([[1.0, 0.0, 1.0]]), batch)
    assert float(value) == 0.0
    assert value.requires_grad


def test_the_gradient_reaches_the_frames_of_both_turns():
    prox = readout([[0.0, 0.0, 0.0]])
    RelativeProximityLoss(consistency_weight=0.0)(prox, batch_of([None])).backward()
    assert float(prox.grad[0, 0:10].abs().sum()) > 0.0     # user turn
    assert float(prox.grad[0, 12:20].abs().sum()) > 0.0    # bystander turn
    assert float(prox.grad[0, 10:12].abs().sum()) == 0.0   # the gap between them


def test_the_error_names_the_config_key_that_is_missing():
    with pytest.raises(ValueError, match="proximity_head"):
        RelativeProximityLoss()(None, batch_of([None]))


def test_it_declares_the_inputs_the_siso_module_provides():
    assert RelativeProximityLoss.required_inputs == ("proximity", "batch")
