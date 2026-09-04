"""The v20 R1a side outputs: what they are, and what they must not disturb.

Three properties, each one a way the round could quietly go wrong:

* the graph-carrying `bottleneck` provider is a *different* thing from the
  detached `last_bottleneck` the presence gate reads -- handing a gradient path
  to code that expected a probe input is the silent version of this change;
* with the heads off the backbone forward is bit-identical and the checkpoint
  keys are unchanged, so a v16-era warm start still loads 0 missing / 0
  unexpected;
* the streaming export gains no ports, because these heads are training-only.
"""

import io

import pytest
import torch

from puresound.nnet import ConvEncDec, FeatureEncoder
from puresound.nnet.dpcrn import DPCRN
from puresound.streaming import create_streaming_dpcrn_model
from puresound.system.siso import EncDecMaskBase

R1A_PROVIDERS = ("bottleneck", "identity_emb", "identity_head", "proximity")

#: Every provider the table had before R1a. Listed rather than derived: the
#: point is that adding to the table did not rename or drop anything.
PRE_R1A_PROVIDERS = (
    "enhanced",
    "target",
    "batch",
    "inactive_labels",
    "vad_target",
    "vad_logits",
    "background_vad_logits",
    "background_vad_target",
    "dist_preds",
)


def tiny_dpcrn(**kw):
    return DPCRN(
        input_dim=64, channels=(1, 8, 16), kernel_t=(2, 2), stride_t=(1, 1),
        dilation_t=(1, 1), kernel_f=(5, 3), stride_f=(2, 2), dilation_f=(1, 1),
        delay=(0, 0), rnn_hidden=16, **kw,
    )


def bare_module(backbone):
    """`_loss_providers` touches only the backbone, so skip Lightning's __init__."""
    module = object.__new__(EncDecMaskBase)
    torch.nn.Module.__init__(module)
    module.backbone = backbone
    return module


def providers_of(backbone):
    return bare_module(backbone)._loss_providers(
        enhanced=None, target=None, vad_target=None, batch=None, inactive_labels=None
    )


def streaming_system(**kw):
    """A real `EncDecMaskBase` at the shipped DPCRN time geometry, narrow."""
    encoder = ConvEncDec(
        fft_length=512, win_type="hann", win_length=512, hop_length=160,
        fmin=0, fmax=8000, sr=16000, trainable=False,
    )
    feats = FeatureEncoder(
        feats_type="complex", drop_stft_first_bin=True, trainable=False,
        include_specaug=False,
    )
    backbone = DPCRN(
        input_dim=256, channels=(2, 8, 16), kernel_t=(2, 2), stride_t=(1, 1),
        dilation_t=(1, 1), kernel_f=(5, 3), stride_f=(2, 2), dilation_f=(1, 1),
        delay=(0, 0), rnn_hidden=16, vad_head={"enabled": True}, **kw,
    )
    return EncDecMaskBase(encoder, feats, backbone, mask_type="complex")


# --------------------------------------------------------------------------- #
# the provider table
# --------------------------------------------------------------------------- #


def test_the_table_gained_the_r1a_names_and_lost_none_of_the_old_ones():
    names = set(providers_of(tiny_dpcrn()))
    assert set(PRE_R1A_PROVIDERS) <= names
    assert set(R1A_PROVIDERS) <= names


@pytest.mark.parametrize("name", R1A_PROVIDERS)
def test_each_r1a_provider_is_none_when_its_switch_is_off(name):
    """Off by default means the provider answers None, so a misconfigured recipe
    hits the loss's own "enable the head" error rather than a shape mismatch."""
    backbone = tiny_dpcrn()
    backbone(torch.randn(2, 1, 64, 20))
    assert providers_of(backbone)[name]() is None


def test_the_bottleneck_provider_is_pooled_carries_a_graph_and_is_not_the_stash():
    """`last_bottleneck` is [N, C, F, T] and detached (the presence gate fits a
    probe on it). The loss-side one is frequency-pooled [N, C, T] -- the
    convention `anchor_gate_cache.py` caches as `feat` and every per-frame head
    pools to internally -- and keeps its graph, which is the whole point: the
    identity loss runs an EMA copy of the head over it.
    """
    backbone = tiny_dpcrn(expose_bottleneck=True)
    backbone.stash_bottleneck = True
    x = torch.randn(2, 1, 64, 20)
    backbone(x)

    pooled = providers_of(backbone)["bottleneck"]()
    assert pooled.dim() == 3 and pooled.shape[0] == 2
    assert pooled.requires_grad

    stashed = backbone.last_bottleneck
    assert stashed.dim() == 4
    assert not stashed.requires_grad
    # Same tensor, one pooling apart: the two switches must not have drifted
    # into reading different things.
    assert torch.allclose(pooled.detach(), stashed.mean(dim=2), atol=1e-6)


def test_a_gradient_reaches_the_encoder_through_the_bottleneck_provider():
    """A detached side output would make the identity loss a no-op that still
    logs a plausible number."""
    backbone = tiny_dpcrn(expose_bottleneck=True)
    x = torch.randn(2, 1, 64, 20, requires_grad=True)
    backbone(x)
    providers_of(backbone)["bottleneck"]().sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0.0


def test_the_head_providers_hand_over_the_forward_s_side_outputs():
    backbone = tiny_dpcrn(
        identity_head={"enabled": True, "dim": 8, "kernel_t": 3},
        proximity_head={"enabled": True, "hidden": 8},
    )
    backbone(torch.randn(3, 1, 64, 24))
    table = providers_of(backbone)

    emb = table["identity_emb"]()
    assert emb.shape == (3, 24, 8)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(3, 24), atol=1e-5)
    assert table["proximity"]().shape == (3, 24)
    # The module itself, not a tensor: the identity loss keeps an EMA copy of it.
    assert table["identity_head"]() is backbone.identity_head


def test_both_new_heads_are_causal_in_time():
    """Changing the future must not change the past.

    Two reasons it is load-bearing here rather than only later: a turn mean that
    can see a later talker's frames is a leak the identity loss would happily
    exploit, and a non-causal readout could not become a streaming read at all.
    Tested on the head itself, so the backbone's own geometry is not the thing
    under test.
    """
    from puresound.nnet.lobe.heads import IdentityHead, ProximityHead

    torch.manual_seed(0)
    x = torch.randn(1, 8, 4, 60)
    future = x.clone()
    future[..., 40:] += 10.0
    for head in (
        IdentityHead(enc_channels=8, dim=8, kernel_t=5).eval(),
        ProximityHead(enc_channels=8, hidden=8, kernel_t=5).eval(),
    ):
        with torch.no_grad():
            before, after = head(x), head(future)
        assert torch.allclose(before[:, :40], after[:, :40], atol=1e-6), type(head)
        assert not torch.allclose(before[:, 40:], after[:, 40:], atol=1e-3), type(head)


def test_the_heads_read_the_provider_s_pooled_tensor_and_the_raw_one_alike():
    """The identity loss's EMA teacher is handed the `bottleneck` provider's
    pooled [N, C, T], while the live head is called on the raw [N, C, F, T]. If
    those two paths did not agree, the teacher's targets would come from a
    different feature than the student's embeddings and nothing would say so.
    """
    backbone = tiny_dpcrn(
        identity_head={"enabled": True, "dim": 8, "kernel_t": 3},
        proximity_head={"enabled": True, "hidden": 8},
        expose_bottleneck=True,
    ).eval()
    with torch.no_grad():
        backbone(torch.randn(2, 1, 64, 24))
        pooled = backbone.last_bottleneck_graph
        assert torch.allclose(
            backbone.identity_head(pooled), backbone.last_identity_emb, atol=1e-6
        )
        assert torch.allclose(
            backbone.proximity_head(pooled), backbone.last_proximity, atol=1e-6
        )


def test_the_inference_forward_does_not_turn_the_graph_side_output_on():
    """`EncDecMaskBase.forward` flips `stash_bottleneck` for a presence gate.
    `expose_bottleneck` is a build-time flag and must stay one -- an inference
    call that started retaining the graph would leak memory per frame."""
    model = streaming_system()
    assert model.backbone.expose_bottleneck is False
    with torch.no_grad():
        model(torch.randn(1, 16000))
    assert model.backbone.expose_bottleneck is False
    assert model.backbone.last_bottleneck_graph is None


# --------------------------------------------------------------------------- #
# off by default
# --------------------------------------------------------------------------- #


def test_heads_off_is_a_bit_identical_forward_and_the_same_checkpoint_keys():
    torch.manual_seed(0)
    plain = tiny_dpcrn().eval()
    torch.manual_seed(0)
    with_blocks = tiny_dpcrn(
        identity_head={"enabled": False},
        proximity_head={"enabled": False},
        expose_bottleneck=False,
    ).eval()

    x = torch.randn(2, 1, 64, 30)
    with torch.no_grad():
        assert torch.equal(plain(x), with_blocks(x))

    assert with_blocks.identity_head is None
    assert with_blocks.proximity_head is None
    assert with_blocks.last_identity_emb is None
    assert with_blocks.last_proximity is None

    buf = io.BytesIO()
    torch.save(plain.state_dict(), buf)
    buf.seek(0)
    report = with_blocks.load_state_dict(torch.load(buf), strict=True)
    assert list(report.missing_keys) == []
    assert list(report.unexpected_keys) == []


def test_enabling_the_heads_does_not_change_the_mask():
    """Training-only in the v11 pattern: the heads read the bottleneck, they do
    not write it, so the separator output is the same tensor either way."""
    torch.manual_seed(0)
    plain = tiny_dpcrn().eval()
    torch.manual_seed(0)
    headed = tiny_dpcrn(
        identity_head={"enabled": True},
        proximity_head={"enabled": True},
        expose_bottleneck=True,
    ).eval()
    # Same separator weights; the heads' own parameters are the extra draws, so
    # copy rather than trust the seed.
    headed.load_state_dict(plain.state_dict(), strict=False)

    x = torch.randn(2, 1, 64, 30)
    with torch.no_grad():
        assert torch.equal(plain(x), headed(x))


def test_the_new_head_blocks_reject_a_misspelled_key():
    """`extra="forbid"`, the same reason the other head configs are: a
    `identity_head: {enabled: true, dimm: 32}` that trained at the default width
    for a whole run is the failure this replaces."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        tiny_dpcrn(identity_head={"enabled": True, "dimm": 32})
    with pytest.raises(ValidationError):
        tiny_dpcrn(proximity_head={"enabled": True, "kernel_t": 7})


# --------------------------------------------------------------------------- #
# the streaming export is not in this round
# --------------------------------------------------------------------------- #


def test_both_r1a_losses_run_through_compute_loss_on_a_session_batch():
    """The provider contract end to end: `invoke_loss` has to be able to satisfy
    four declared inputs -- one of which is a module -- off the real table, and
    the gradient has to reach the encoder through the bottleneck."""
    from puresound.nnet.loss import IdentityContrastiveLoss, RelativeProximityLoss

    torch.manual_seed(0)
    model = streaming_system(
        identity_head={"enabled": True, "dim": 8, "kernel_t": 3},
        proximity_head={"enabled": True, "hidden": 8},
        expose_bottleneck=True,
    )
    model.register_loss_func(
        torch.nn.ModuleList([IdentityContrastiveLoss(), RelativeProximityLoss()]),
        [0.1, 0.1],
    )
    model.train()

    # 2 s at hop 160: 198 label frames against the head's 197 (see
    # test_turn_pooling_alignment), and three turns that all fit inside them.
    samples, frames = 32000, 198
    noisy = torch.randn(2, samples) * 0.1
    turn_id = torch.zeros(2, frames, dtype=torch.long)
    turn_id[:, 5:60] = 1     # user
    turn_id[:, 70:120] = 2   # bystander
    turn_id[:, 130:190] = 3  # the user's return
    batch = {
        "turn_id": turn_id,
        "turn_role": torch.tensor([[1, 2, 1], [1, 2, 1]]),
        # Speaker 11 is the user in both rows, rendered through two chains;
        # 22 and 33 are the bystanders, one per row.
        "turn_speaker": torch.tensor([[11, 22, 11], [11, 33, 11]]),
        "turn_chain": torch.tensor([[0, 0, 0], [1, 1, 1]]),
        "row_source_id": torch.tensor([5, 5]),
        "user_active": torch.zeros(2, frames),
        "bystander_active": torch.zeros(2, frames),
    }

    enhanced = model(noisy)
    total, values = model.compute_loss(
        enhanced=enhanced, target=torch.randn(2, samples) * 0.1, batch=batch
    )
    assert torch.isfinite(total)
    assert len(values) == 2 and all(v != 0.0 for v in values)

    total.backward()
    grads = [
        float(p.grad.abs().sum())
        for p in model.backbone.parameters()
        if p.grad is not None
    ]
    assert grads and max(grads) > 0.0


def test_todays_rows_leave_both_r1a_losses_at_exactly_zero():
    """A batch without the session keys must not change an old recipe's number,
    and must still carry a graph edge so DDP never reports an unused head."""
    from puresound.nnet.loss import IdentityContrastiveLoss, RelativeProximityLoss

    torch.manual_seed(0)
    model = streaming_system(
        identity_head={"enabled": True, "dim": 8, "kernel_t": 3},
        proximity_head={"enabled": True, "hidden": 8},
        expose_bottleneck=True,
    )
    model.register_loss_func(
        torch.nn.ModuleList([IdentityContrastiveLoss(), RelativeProximityLoss()]),
        [1.0, 1.0],
    )
    model.train()

    noisy = torch.randn(2, 8000) * 0.1
    total, values = model.compute_loss(
        enhanced=model(noisy), target=torch.randn(2, 8000) * 0.1, batch={}
    )
    assert values == [0.0, 0.0]
    assert float(total) == 0.0
    total.backward()  # a graph-carrying zero, so this is not an error


def test_the_streaming_export_gains_no_state_ports_from_the_new_heads():
    """R1a's success criterion is that inference is unchanged. The frame model
    enumerates exportable heads by name (`vad_head`, `background_vad_head`), so
    an identity or proximity head must add no port and no extra output -- and if
    a later round wants them streamed, this test is what has to be updated
    deliberately rather than discovered from a manifest mismatch.
    """
    off = create_streaming_dpcrn_model(streaming_system())
    on = create_streaming_dpcrn_model(
        streaming_system(
            identity_head={"enabled": True},
            proximity_head={"enabled": True},
            expose_bottleneck=True,
        )
    )
    assert on.state_input_names == off.state_input_names
    assert on.state_output_names == off.state_output_names
    assert on.extra_output_names == off.extra_output_names == ["vad_logit"]
    assert [name for name, _ in on.heads] == ["vad_head"]
