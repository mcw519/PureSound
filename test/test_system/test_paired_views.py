"""The generic paired-view dispatcher does not know about session rows."""

import torch

from puresound.nnet.loss.proximity import RelativeProximityLoss
from puresound.system.paired_views import PairedViewConsistencyConfig, paired_view_loss


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


def test_generic_dispatch_runs_one_auxiliary_forward_and_keeps_primary_batch():
    module = _ToyModule()
    wav, batch = _batch()
    module.forward(wav)
    value = paired_view_loss(
        module, batch, wav,
        PairedViewConsistencyConfig(enabled=True, max_rows=1),
    )
    assert module.forward_calls == 2
    value.backward()
    assert wav.grad is not None
    assert value.requires_grad


def test_default_off_does_no_extra_forward_and_returns_graph_zero():
    module = _ToyModule()
    wav, batch = _batch()
    module.forward(wav)
    # A disabled config is handled by EncDecMaskBase before this helper. Calling
    # the helper directly with no paired rows still must be a graph zero.
    batch.pop("paired_view")
    value = paired_view_loss(
        module, batch, wav,
        PairedViewConsistencyConfig(enabled=True, max_rows=1),
    )
    assert module.forward_calls == 1
    assert float(value) == 0.0
    value.backward()
    assert wav.grad is not None
