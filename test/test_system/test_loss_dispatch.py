"""How a loss gets its inputs.

`invoke_loss` replaced a chain of `if getattr(loss, "uses_X", False)` branches.
In a chain the order decides, and two things followed from that: a loss wanting
a later input had to switch every earlier flag off by hand, and a module that
could satisfy none of them fell through to `loss(enhanced, target)` -- a number
rather than an error. Both are pinned here.
"""

import pytest
import torch

from puresound.nnet import loss as loss_module
from puresound.nnet.loss import (
    BackgroundVADHeadBCELoss,
    VADHeadBCELoss,
)
from puresound.system.base import DEFAULT_LOSS_INPUTS, invoke_loss
from puresound.system.miso import EncDecCondMaskBase
from puresound.system.siso import EncDecMaskBase


class _Recorder(torch.nn.Module):
    """A loss that reports what it was handed instead of computing anything."""

    def __init__(self, required_inputs=None):
        super().__init__()
        if required_inputs is not None:
            self.required_inputs = required_inputs
        self.seen = None

    def forward(self, *args):
        self.seen = args
        return torch.zeros(())


class _Backbone:
    def __init__(self, **kw):
        self.last_vad_logits = kw.get("vad_logits")
        self.last_background_vad_logits = kw.get("background_vad_logits")
        self.last_dist_preds = kw.get("dist_preds")


def _module(cls, backbone=None, losses=(), weights=None):
    # compute_loss only touches loss_func_list/_w and backbone, so bypass the
    # Lightning __init__ (encoder / feats / optimizer machinery).
    module = object.__new__(cls)
    torch.nn.Module.__init__(module)
    module.backbone = backbone or _Backbone()
    module.loss_func_list = torch.nn.ModuleList(losses)
    module.loss_func_list_w = list(weights or [1.0] * len(losses))
    return module


# --------------------------------------------------------------------------- #
# invoke_loss itself
# --------------------------------------------------------------------------- #


def test_a_loss_is_called_with_exactly_what_it_declared_in_that_order():
    loss = _Recorder(("target", "enhanced"))  # deliberately reversed
    invoke_loss(loss, {"enhanced": lambda: "E", "target": lambda: "T"})
    assert loss.seen == ("T", "E")


def test_a_loss_that_declares_nothing_gets_the_waveform_pair():
    loss = _Recorder()
    assert not hasattr(loss, "required_inputs")
    invoke_loss(loss, {"enhanced": lambda: "E", "target": lambda: "T"})
    assert loss.seen == ("E", "T")
    assert DEFAULT_LOSS_INPUTS == ("enhanced", "target")


def test_an_input_this_module_cannot_provide_raises_and_names_it():
    """The chain ended in `loss(enhanced, target)`, so a module that could not
    satisfy a loss called it with the waveform pair. That is a trained-on number,
    not a crash."""
    loss = _Recorder(("vad_logits", "vad_target"))
    with pytest.raises(TypeError, match="vad_logits"):
        invoke_loss(loss, {"enhanced": lambda: "E", "target": lambda: "T"})
    assert loss.seen is None


def test_a_provider_nothing_asked_for_is_never_called():
    """Providers are callables so an unused side output is not read off the
    backbone, and a target nothing needs is not synthesized, on every batch."""
    calls = []

    def spy(name):
        def provide():
            calls.append(name)
            return name
        return provide

    invoke_loss(
        _Recorder(("enhanced",)),
        {"enhanced": spy("enhanced"), "dist_preds": spy("dist_preds")},
    )
    assert calls == ["enhanced"]


# --------------------------------------------------------------------------- #
# The subclass the ordered chain could get wrong
# --------------------------------------------------------------------------- #


def test_the_background_head_loss_gets_background_inputs_not_foreground_ones():
    """`BackgroundVADHeadBCELoss` subclasses `VADHeadBCELoss`.

    Under the ordered chain it inherited `uses_vad_logits = True` and had to
    carry `uses_vad_logits = False` purely to fall past the foreground branch.
    Drop that line and it trained against the foreground head's logits and the
    foreground target, silently. A declaration replaces rather than negates, so
    this is now structural -- but assert the routing anyway, because that is the
    property that mattered.
    """
    assert BackgroundVADHeadBCELoss.required_inputs == (
        "background_vad_logits",
        "background_vad_target",
    )
    assert VADHeadBCELoss.required_inputs == ("vad_logits", "vad_target")

    foreground = torch.full((2, 8), 3.0)
    background = torch.full((2, 8), -3.0)
    module = _module(
        EncDecMaskBase,
        _Backbone(vad_logits=foreground, background_vad_logits=background),
    )
    providers = module._loss_providers(
        enhanced=None, target=None, vad_target=None, batch={}, inactive_labels=None
    )
    loss = _Recorder(BackgroundVADHeadBCELoss.required_inputs)
    invoke_loss(loss, providers)
    assert torch.equal(loss.seen[0], background)
    # The synthesized all-zeros target, not the foreground vad_target.
    assert torch.equal(loss.seen[1], torch.zeros_like(background))


# --------------------------------------------------------------------------- #
# The two halves stay one list
# --------------------------------------------------------------------------- #


def _provider_names(cls, **kwargs):
    return set(_module(cls)._loss_providers(**kwargs))


SISO_ARGS = dict(
    enhanced=None, target=None, vad_target=None, batch=None, inactive_labels=None
)
MISO_ARGS = dict(enhanced=None, target=None, vad_target=None)


def test_every_shipped_loss_asks_only_for_things_a_module_provides():
    """A loss declaring a name no module offers is a `TypeError` on the first
    real batch. Checking the declarations against the provider table -- read off
    the module itself, not a copy of its names -- catches it here instead.
    """
    available = _provider_names(EncDecMaskBase, **SISO_ARGS)
    unsatisfiable = {}
    for name in loss_module.__all__:
        declared = getattr(getattr(loss_module, name), "required_inputs", None)
        if declared is None:
            continue  # class-level default, or set per instance
        missing = set(declared) - available
        if missing:
            unsatisfiable[name] = sorted(missing)
    assert unsatisfiable == {}, (
        f"these losses ask for inputs no module provides: {unsatisfiable}"
    )


def test_miso_provides_a_subset_of_what_siso_does():
    """MISO hosts no backbone side outputs, deliberately. It should be narrower
    than SISO rather than differently named -- a provider only MISO had would be
    a loss that works on one module and mysteriously not the other."""
    assert _provider_names(EncDecCondMaskBase, **MISO_ARGS) <= _provider_names(
        EncDecMaskBase, **SISO_ARGS
    )


def test_miso_refuses_a_side_output_loss_instead_of_mis_calling_it():
    """Measured on the chain this replaced: asking MISO for VADHeadBCELoss
    returned 0.8132, computed from the enhanced waveform read as logits and the
    clean waveform read as the target, against 0.7727 from the real ones."""
    module = _module(EncDecCondMaskBase, losses=[VADHeadBCELoss()])
    with pytest.raises(TypeError, match="vad_logits"):
        module.compute_loss(torch.randn(2, 1600), torch.randn(2, 1600))
