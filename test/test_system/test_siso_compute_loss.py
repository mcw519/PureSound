import torch

from puresound.nnet.loss import (
    BackgroundVADHeadBCELoss,
    ResidualReferenceLoss,
    VADHeadBCELoss,
)
from puresound.system.siso import EncDecMaskBase


class _BackboneStub:
    """Minimal backbone exposing only the side outputs ``compute_loss`` reads."""

    def __init__(self, vad_logits=None, background_vad_logits=None):
        self.last_vad_logits = vad_logits
        self.last_background_vad_logits = background_vad_logits
        self.last_aux_outputs = {}


def _make_module(backbone, loss_funcs, weights):
    # compute_loss only touches loss_func_list/_w and backbone, so bypass the
    # heavyweight Lightning __init__ (encoder/feats/optimizer machinery).
    module = object.__new__(EncDecMaskBase)
    torch.nn.Module.__init__(module)  # set up _modules/_parameters dicts only
    module.backbone = backbone
    module.loss_func_list = torch.nn.ModuleList(loss_funcs)
    module.loss_func_list_w = weights
    return module


def test_background_vad_loss_handles_all_silent_batch():
    """An all-silent batch carries no ``background_vad_target``; compute_loss
    must treat the missing target as all-zeros instead of crashing the loss.

    Regression test for the DDP-killing crash where a rare batch with no
    background speech in any row left ``background_vad_target`` absent and
    BackgroundVADHeadBCELoss raised on a None target.
    """
    backbone = _BackboneStub(background_vad_logits=torch.randn(2, 19))
    module = _make_module(backbone, [BackgroundVADHeadBCELoss()], [0.2])

    enhanced = torch.randn(2, 1600)
    target = torch.randn(2, 1600)

    # batch dict is present but has no "background_vad_target" key.
    overall, losses = module.compute_loss(
        enhanced, target, vad_target=None, batch={}
    )

    assert torch.isfinite(overall)
    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses[0]))


def test_background_vad_loss_matches_explicit_zero_target():
    """The synthesized all-zeros target must equal an explicitly-supplied one."""
    logits = torch.randn(2, 19)

    synth_module = _make_module(
        _BackboneStub(background_vad_logits=logits),
        [BackgroundVADHeadBCELoss()],
        [1.0],
    )
    explicit_module = _make_module(
        _BackboneStub(background_vad_logits=logits),
        [BackgroundVADHeadBCELoss()],
        [1.0],
    )

    enhanced = torch.randn(2, 1600)
    target = torch.randn(2, 1600)

    synth_overall, _ = synth_module.compute_loss(
        enhanced, target, vad_target=None, batch={}
    )
    explicit_overall, _ = explicit_module.compute_loss(
        enhanced,
        target,
        vad_target=None,
        batch={"background_vad_target": torch.zeros_like(logits)},
    )

    assert torch.allclose(synth_overall, explicit_overall)


def test_foreground_vad_loss_still_requires_vad_target():
    """The zero-fill is background-only; a missing foreground target should
    still surface the explicit configuration error rather than be masked."""
    backbone = _BackboneStub(vad_logits=torch.randn(2, 19))
    module = _make_module(backbone, [VADHeadBCELoss()], [0.5])

    enhanced = torch.randn(2, 1600)
    target = torch.randn(2, 1600)

    try:
        module.compute_loss(enhanced, target, vad_target=None, batch={})
    except ValueError as exc:
        assert "vad_target" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing foreground vad_target")


def test_residual_reference_loss_routes_batch_context():
    """Residual auxiliary loss should supervise noisy-enhanced against the
    dataloader's consistency residual without requiring a deployed far head."""
    enhanced = torch.tensor([[0.8, 0.1, -0.2, 0.0]])
    target = enhanced.clone()
    noisy = torch.tensor([[1.0, 0.0, 0.0, 0.5]])
    residual_ref = noisy - enhanced

    module = _make_module(
        _BackboneStub(),
        [ResidualReferenceLoss(loss="l1")],
        [0.2],
    )
    overall, losses = module.compute_loss(
        enhanced,
        target,
        batch={
            "noisy_speech": noisy,
            "consistency_noise": residual_ref,
            "target_present": torch.ones(1),
        },
    )

    assert torch.allclose(overall, torch.tensor(0.0))
    assert losses == [0.0]
