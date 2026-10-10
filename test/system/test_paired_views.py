"""The generic paired-view dispatcher: one auxiliary forward on the rows that
carry a paired view, collated onto the primary rows' frame grid, and a
graph-carrying zero when a batch carries none. It does not know about session
rows."""

from types import SimpleNamespace

import pytest
import torch

from puresound.nnet.loss.proximity import RelativeProximityLoss
from puresound.system.paired_views import PairedViewConsistencyConfig, paired_view_loss
from puresound.task.paired_views import collate_paired_views


class _ToyModule:
    def __init__(self):
        self.loss_func_list = torch.nn.ModuleList([RelativeProximityLoss()])
        self.loss_func_list_w = [0.5]
        self.backbone = type("Backbone", (), {})()
        self.forward_calls = 0

    def forward(self, wav):
        self.forward_calls += 1
        self.backbone.last_proximity = wav
        self.last_mask = wav
        return wav

    def _loss_providers(self, *, enhanced, target, vad_target, batch, inactive_labels):
        del enhanced, target, vad_target, inactive_labels
        return {
            "proximity": lambda: self.backbone.last_proximity,
            "batch": lambda: batch,
        }

    def log(self, *args, **kwargs):
        del args, kwargs


def _batch():
    wav = torch.zeros(2, 4, requires_grad=True)
    labels = {
        "noisy_speech": wav,
        "clean_speech": wav.detach(),
        "turn_id": torch.tensor([[1, 1, 2, 2], [1, 1, 2, 2]]),
        "turn_role": torch.tensor([[1, 2], [1, 2]]),
        "turn_distance": torch.tensor([[.5, 2.], [.5, 2.]]),
        "turn_chain": torch.ones(2, 2, dtype=torch.long),
        "row_source_id": torch.tensor([7, -1]),
    }
    labels["paired_view"] = {
        "noisy_speech": wav[:1].detach().clone(),
        "clean_speech": wav[:1].detach().clone(),
        "source_indices": torch.tensor([0]),
        "row_source_id": torch.tensor([7]),
        "turn_chain": torch.tensor([[2, 2]]),
    }
    return wav, labels


@pytest.mark.parametrize("paired", [True, False], ids=["paired-rows", "no-paired-rows"])
def test_dispatch_runs_at_most_one_auxiliary_forward_and_keeps_the_graph(paired):
    """With no paired rows the helper still returns a graph-carrying zero (a
    disabled config is handled by ``EncDecMaskBase`` before this helper)."""
    module = _ToyModule()
    wav, batch = _batch()
    if not paired:
        batch.pop("paired_view")
    module.forward(wav)
    value = paired_view_loss(
        module, batch, wav, PairedViewConsistencyConfig(enabled=True, max_rows=1)
    )
    assert module.forward_calls == (2 if paired else 1)
    assert value.requires_grad
    if not paired:
        assert float(value) == 0.0
    value.backward()
    assert wav.grad is not None


def test_a_short_auxiliary_view_is_padded_onto_the_primary_grid_and_backpropagates():
    class Objective:
        paired_output = "enhanced"

        def paired_consistency(self, primary, secondary, batch, view):
            return (primary[view["source_indices"]] - secondary).square().mean(), 1, 1

    class Module:
        backbone = SimpleNamespace()
        loss_func_list = [Objective()]
        loss_func_list_w = [1.0]

        def forward(self, x):
            return x * 2

        def _loss_providers(self, *, enhanced, **kwargs):
            return {"enhanced": lambda: enhanced}

        def log(self, *args, **kwargs):
            pass

    x = torch.ones(2, 10, requires_grad=True)
    rows = [{"row_source_id": 1, "paired_view": {"row_source_id": 1,
             "noisy_speech": torch.ones(8), "clean_speech": torch.ones(8)}}, {}]
    batch = collate_paired_views(rows, {"noisy_speech": x, "clean_speech": x.detach()})
    assert batch["paired_view"]["noisy_speech"].shape == (1, 10)
    assert torch.equal(batch["paired_view"]["noisy_speech"][0, 8:], torch.zeros(2))
    value = paired_view_loss(Module(), batch, x, PairedViewConsistencyConfig(enabled=True))
    value.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
