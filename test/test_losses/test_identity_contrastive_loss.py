"""What `IdentityContrastiveLoss` must be doing to be worth training (v20 R1a).

The loss is easy to write in a way that *looks* right and learns a near/far
scalar instead of an identity (v20 review item 4), or that collapses because
its targets co-adapt, or that trains on overlap frames where "whose voice is
this" has no answer. Each test below is one of those.

The batch is synthetic and constructed so that "same speaker" is a fact about
the features: every turn's frames carry that speaker's own vector plus a
row-level chain bias, so a correct label set has genuinely similar positives
and a permuted one does not.
"""

import pytest
import torch

from puresound.nnet.lobe.heads import IdentityHead
from puresound.nnet.loss import IdentityContrastiveLoss

C = 12                              # bottleneck width
T = 40                              # frames
SPANS = ((2, 18), (22, 38))         # one turn per span
K = len(SPANS)


def head(seed: int = 0) -> IdentityHead:
    torch.manual_seed(seed)
    return IdentityHead(enc_channels=C, dim=8, kernel_t=3)


def bottleneck(feature_speaker, chains, seed: int = 1) -> torch.Tensor:
    """[B, C, T] where each turn's frames carry its speaker's vector.

    ``feature_speaker[b][k]`` is the speaker whose voice turn k of row b really
    is; ``chains[b]`` adds a row-level capture-chain bias, which is the
    shortcut the loss must not key on.
    """
    generator = torch.Generator().manual_seed(seed)
    n_speakers = 1 + max(max(row) for row in feature_speaker)
    voices = torch.randn(n_speakers, C, generator=generator) * 3.0
    chain_bias = torch.randn(1 + max(chains), C, generator=generator) * 0.7

    rows = []
    for b, row in enumerate(feature_speaker):
        x = chain_bias[chains[b]].unsqueeze(-1).repeat(1, T)
        x = x + 0.05 * torch.randn(C, T, generator=generator)
        for k, (start, end) in enumerate(SPANS):
            x[:, start:end] += voices[row[k]].unsqueeze(-1)
        rows.append(x)
    return torch.stack(rows)


def turn_id_tensor(batch_size: int, spans=SPANS) -> torch.Tensor:
    ids = torch.zeros(batch_size, T, dtype=torch.long)
    for k, (start, end) in enumerate(spans):
        ids[:, start:end] = k + 1
    return ids


def session_batch(label_speaker, chains, spans=SPANS, **extra) -> dict:
    batch_size = len(label_speaker)
    return {
        "turn_id": turn_id_tensor(batch_size, spans),
        "turn_role": torch.tensor([[1, 2]] * batch_size),
        "turn_speaker": torch.tensor(label_speaker),
        "turn_chain": torch.tensor([[c] * K for c in chains]),
        "user_active": torch.zeros(batch_size, T),
        "bystander_active": torch.zeros(batch_size, T),
        **extra,
    }


# The features: rows alternate which of the two speakers takes which turn slot,
# and each row is a different capture chain.
FEATURE_SPEAKER = [[0, 1], [1, 0], [0, 1], [1, 0]]
CHAINS = [0, 1, 2, 3]
#: Labels that name the speakers correctly: every positive pair is the same
#: voice through a different chain.
TRUE_LABELS = [[100, 200], [200, 100], [100, 200], [200, 100]]
#: Same class sizes, but "speaker 100" is now two of each voice -- the control
#: for "did the loss actually use identity, or just count".
PERMUTED_LABELS = [[100, 200], [100, 200], [100, 200], [100, 200]]


def run(labels, chains=CHAINS, feature_speaker=FEATURE_SPEAKER, batch_extra=None,
        loss=None, bott=None, spans=SPANS, training=True):
    net = head()
    bott = bottleneck(feature_speaker, chains) if bott is None else bott
    emb = net(bott)
    loss = IdentityContrastiveLoss() if loss is None else loss
    loss.train(training)
    batch = session_batch(labels, chains, spans=spans, **(batch_extra or {}))
    return loss(emb, net, bott, batch), loss, net


# --------------------------------------------------------------------------- #
# the shortcut this loss exists to close
# --------------------------------------------------------------------------- #


def test_true_speaker_labels_score_better_than_a_permuted_control():
    """Positives are the same voice through a *different chain*; negatives are
    the other voice. If the loss can be satisfied without using identity, the
    permuted labels -- same class sizes, same rows, same features -- score the
    same. Measured: they do not.
    """
    correct, _, _ = run(TRUE_LABELS)
    control, _, _ = run(PERMUTED_LABELS)
    assert float(correct) < float(control), (float(correct), float(control))


def test_a_speaker_appearing_only_once_contributes_nothing():
    """No positive anywhere in the batch means no term, not a zero-similarity
    target pulled out of nowhere."""
    singletons = [[1, 2], [3, 4], [5, 6], [7, 8]]
    value, _, _ = run(singletons)
    assert float(value) == 0.0
    assert value.requires_grad


def test_pad_turns_and_pad_speakers_are_not_candidates():
    """`turn_speaker == -1` is the contract's pad. Four pads all carrying -1
    would otherwise read as four mutual same-speaker positives -- a loss term
    computed over padding.
    """
    net = head()
    bott = bottleneck(FEATURE_SPEAKER, CHAINS)
    padded = [[100, -1], [200, -1], [100, -1], [200, -1]]

    with_pads = session_batch(padded, CHAINS)
    without_pads = session_batch(padded, CHAINS)
    # The same batch with the pad turns' frames simply not claimed by any turn.
    without_pads["turn_id"] = torch.where(
        without_pads["turn_id"] == 2,
        torch.zeros_like(without_pads["turn_id"]),
        without_pads["turn_id"],
    )
    kept = IdentityContrastiveLoss()(net(bott), net, bott, with_pads)
    reference = IdentityContrastiveLoss()(net(bott), net, bott, without_pads)
    assert torch.allclose(kept, reference, atol=1e-6), (float(kept), float(reference))

    # Teeth: give those same turns a real shared id and they do enter the loss.
    real = session_batch([[100, 300], [200, 300], [100, 300], [200, 300]], CHAINS)
    assert not torch.allclose(
        IdentityContrastiveLoss()(net(bott), net, bott, real), reference, atol=1e-6
    )

    only_pads, _, _ = run([[-1, -1]] * 4)
    assert float(only_pads) == 0.0


# --------------------------------------------------------------------------- #
# overlap
# --------------------------------------------------------------------------- #


def _corrupted_tail(rows: int, amount: float = 50.0):
    """A bottleneck whose last 4 frames of every turn are wrecked, plus the
    activity tracks that flag exactly those frames as overlap.

    Row-dependent corruption so it destroys cross-row similarity rather than
    adding a constant every row shares. The wrecked frames sit at the END of
    each turn, so a causal head guarantees no kept frame can see them.
    """
    clean = bottleneck(FEATURE_SPEAKER, CHAINS)
    dirty = clean.clone()
    overlap = torch.zeros(rows, T)
    for _, end in SPANS:
        for b in range(rows):
            dirty[b, :, end - 4 : end] += amount * (b + 1)
        overlap[:, end - 4 : end] = 1.0
    flags = {"user_active": torch.ones(rows, T), "bystander_active": overlap}
    return clean, dirty, flags


def test_frames_where_both_talkers_are_active_are_excluded_from_the_turn_mean():
    """Two ways of saying "these frames are overlap": `turn_id = 0` (the
    contract) and the two activity tracks both at 1 (a generator that forgot).
    Pooling has to honour the second as well, or a mixed frame's embedding lands
    in a speaker's turn mean -- asserted on the mean itself, because the loss
    can be saturated enough to hide a change in it.
    """
    from puresound.nnet.loss.identity import align_turn_frames, pool_turn_means

    rows = len(FEATURE_SPEAKER)
    net = head()
    clean, dirty, flags = _corrupted_tail(rows)
    batch = session_batch(TRUE_LABELS, CHAINS, **flags)
    device = torch.device("cpu")

    def pooled(features, exclude_overlap):
        (values,), ids, exclude = align_turn_frames([net(features)], batch, device)
        return pool_turn_means(values, ids, K, exclude if exclude_overlap else None)[0]

    masked = pooled(dirty, True)
    reference = pooled(clean, True)
    assert torch.allclose(masked, reference, atol=1e-6), \
        float((masked - reference).abs().max())

    unmasked = pooled(dirty, False)
    assert float((unmasked - reference).abs().max()) > 0.05


def test_the_loss_is_unchanged_by_whatever_happens_on_overlap_frames():
    rows = len(FEATURE_SPEAKER)
    net = head()
    clean, dirty, flags = _corrupted_tail(rows)
    batch = session_batch(TRUE_LABELS, CHAINS, **flags)

    flagged = IdentityContrastiveLoss()(net(dirty), net, dirty, batch)
    reference = IdentityContrastiveLoss()(net(clean), net, clean, batch)
    assert torch.allclose(flagged, reference, atol=1e-8), (
        float(flagged), float(reference)
    )


# --------------------------------------------------------------------------- #
# the EMA teacher
# --------------------------------------------------------------------------- #


def test_the_teacher_takes_no_gradient_and_is_not_in_the_checkpoint():
    """Trainable targets collapse by co-adaptation; a teacher submodule created
    on step 1 changes `state_dict` after step 1 and breaks its own resume."""
    value, loss, net = run(TRUE_LABELS)
    value.backward()

    assert float(net.out.weight.grad.abs().sum()) > 0.0
    assert all(not p.requires_grad for p in loss.teacher.parameters())
    assert all(p.grad is None for p in loss.teacher.parameters())
    assert loss.state_dict() == {}
    assert not any(
        module is loss.teacher for module in loss.modules() if module is not loss
    )


def test_the_teacher_starts_as_a_copy_of_the_head_and_then_trails_it():
    net = head()
    bott = bottleneck(FEATURE_SPEAKER, CHAINS)
    loss = IdentityContrastiveLoss(momentum=0.9).train()

    assert loss.teacher is None
    loss(net(bott), net, bott, session_batch(TRUE_LABELS, CHAINS))
    start = loss.teacher.out.weight.detach().clone()
    assert torch.equal(start, net.out.weight.detach())

    with torch.no_grad():
        net.out.weight.add_(1.0)
    loss(net(bott), net, bott, session_batch(TRUE_LABELS, CHAINS))
    assert torch.allclose(loss.teacher.out.weight, start + 0.1, atol=1e-6)


def test_the_teacher_does_not_move_during_validation():
    """`validation_step` calls the same loss object. An EMA that advanced there
    would fold held-out batches into the training targets."""
    net = head()
    bott = bottleneck(FEATURE_SPEAKER, CHAINS)
    loss = IdentityContrastiveLoss(momentum=0.5).train()
    loss(net(bott), net, bott, session_batch(TRUE_LABELS, CHAINS))
    frozen = loss.teacher.out.weight.detach().clone()

    loss.eval()
    with torch.no_grad():
        net.out.weight.add_(5.0)
    loss(net(bott), net, bott, session_batch(TRUE_LABELS, CHAINS))
    assert torch.equal(loss.teacher.out.weight, frozen)


def test_a_head_of_another_shape_is_refused_rather_than_averaged():
    net = head()
    bott = bottleneck(FEATURE_SPEAKER, CHAINS)
    loss = IdentityContrastiveLoss().train()
    loss(net(bott), net, bott, session_batch(TRUE_LABELS, CHAINS))

    wider = IdentityHead(enc_channels=C, dim=16, kernel_t=3)
    with pytest.raises(ValueError, match="one loss instance tracks one head"):
        loss(wider(bott), wider, bott, session_batch(TRUE_LABELS, CHAINS))


# --------------------------------------------------------------------------- #
# today's rows, and misconfiguration
# --------------------------------------------------------------------------- #


def test_a_batch_without_the_session_keys_is_a_graph_carrying_zero():
    """Old recipes keep their numbers, and DDP never sees an unused head."""
    net = head()
    bott = bottleneck(FEATURE_SPEAKER, CHAINS)
    emb = net(bott)
    value = IdentityContrastiveLoss()(emb, net, bott, {})
    assert float(value) == 0.0
    assert value.requires_grad
    value.backward()
    assert net.out.weight.grad is not None


def test_no_eligible_turn_is_a_graph_carrying_zero():
    net = head()
    bott = bottleneck(FEATURE_SPEAKER, CHAINS)
    batch = session_batch(TRUE_LABELS, CHAINS)
    batch["turn_id"] = torch.zeros_like(batch["turn_id"])
    value = IdentityContrastiveLoss()(net(bott), net, bott, batch)
    assert float(value) == 0.0
    assert value.requires_grad


def test_the_errors_name_the_config_key_that_is_missing():
    net = head()
    bott = bottleneck(FEATURE_SPEAKER, CHAINS)
    batch = session_batch(TRUE_LABELS, CHAINS)
    loss = IdentityContrastiveLoss()
    with pytest.raises(ValueError, match="identity_head"):
        loss(None, None, bott, batch)
    with pytest.raises(ValueError, match="expose_bottleneck"):
        loss(net(bott), net, None, batch)


def test_it_declares_the_inputs_the_siso_module_provides():
    assert IdentityContrastiveLoss.required_inputs == (
        "identity_emb",
        "identity_head",
        "bottleneck",
        "batch",
    )
