"""How a loss gets its inputs: ``invoke_loss`` and the provider tables it reads.

A loss declares ``required_inputs`` and is called with exactly those, in that
order. Two failure modes are pinned: a declaration that names something the
module cannot provide must raise rather than fall back to the waveform pair,
and a declaration no module can satisfy must be caught here rather than on the
first real batch.
"""

import lightning as L
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from puresound.nnet import loss as loss_module
from puresound.nnet.loss import VADHeadBCELoss
from puresound.system.base import DEFAULT_LOSS_INPUTS, BaseLightningModule, invoke_loss
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


@pytest.mark.parametrize(
    "declared, expected",
    [(("target", "enhanced"), ("T", "E")), (None, ("E", "T")), (("enhanced",), ("E",))],
    ids=["declared-order", "default-pair", "subset"],
)
def test_a_loss_is_called_with_exactly_what_it_declared_in_that_order(declared, expected):
    """Providers are callables, so one nothing asked for is never called: an
    unused side output is not read off the backbone, and a target nothing needs
    is not synthesized, on every batch."""
    calls = []

    def spy(name, value):
        def provide():
            calls.append(name)
            return value
        return provide

    loss = _Recorder(declared)
    invoke_loss(loss, {"enhanced": spy("enhanced", "E"), "target": spy("target", "T"),
                       "dist_preds": spy("dist_preds", "D")})
    assert loss.seen == expected
    assert "dist_preds" not in calls and len(calls) == len(expected)
    assert DEFAULT_LOSS_INPUTS == ("enhanced", "target")


def test_an_input_this_module_cannot_provide_raises_and_names_it():
    """Falling back to ``loss(enhanced, target)`` would train on a number, not crash."""
    loss = _Recorder(("vad_logits", "vad_target"))
    with pytest.raises(TypeError, match="vad_logits"):
        invoke_loss(loss, {"enhanced": lambda: "E", "target": lambda: "T"})
    assert loss.seen is None


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


# --------------------------------------------------------------------------- #
# Learning-rate warm-up across a resume
# --------------------------------------------------------------------------- #


class _Ramp(BaseLightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.rates = []
        optimizer = torch.optim.SGD(self.parameters(), lr=1.0)
        self.register_optimizer(optimizer)
        self.register_scheduler(torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000))
        self.register_warmup_step(10)

    def training_step(self, batch, batch_idx):
        loss = self.layer(batch).square().mean()
        self.puresound_logging.update({"epoch_train_loss": loss.item()})
        return loss

    def on_before_optimizer_step(self, optimizer):
        self.rates.append(optimizer.param_groups[0]["lr"])


def _fit_ramp(epochs, **fit_kwargs):
    model, saved = _Ramp(), ModelCheckpoint(save_top_k=-1, every_n_epochs=1, save_on_train_epoch_end=True)
    trainer = L.Trainer(accelerator="cpu", devices=1, max_epochs=epochs, limit_train_batches=4,
                        logger=False, enable_progress_bar=False, enable_model_summary=False,
                        callbacks=[saved], default_root_dir=fit_kwargs.pop("root"))
    trainer.fit(model, torch.utils.data.DataLoader(torch.ones(8, 1, 1), batch_size=None), **fit_kwargs)
    return model.rates, saved.best_model_path


def test_a_resume_inside_the_warmup_continues_the_ramp(tmp_path):
    """A warm-up longer than one epoch is resumed from a checkpoint that already
    holds scaled rates; the ramp has to carry on from its own base, not crash and
    not restart from the scaled value."""
    uninterrupted, _ = _fit_ramp(3, root=tmp_path / "whole")
    first, checkpoint = _fit_ramp(1, root=tmp_path / "first")
    resumed, _ = _fit_ramp(3, root=tmp_path / "resumed", ckpt_path=checkpoint)

    assert first == uninterrupted[:4]
    assert resumed == pytest.approx(uninterrupted[4:])
