"""Speaker-classification losses stay finite where their geometry degenerates."""

import pytest
import torch

from puresound.nnet.loss import SphereFace2, TripletLoss


@pytest.mark.parametrize("margin_type", ["A", "C"])
def test_sphereface2_is_finite_when_an_embedding_sits_on_its_class_weight(margin_type):
    """cos(theta) can round to just above 1 there; the arc-face branch takes the
    square root of 1 - cos^2."""
    torch.manual_seed(0)
    for _ in range(20):
        loss_fn = SphereFace2(192, 10, margin_type=margin_type)
        x = loss_fn.weight[:4].detach().clone().requires_grad_(True)
        loss = loss_fn(x, torch.arange(4))
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(x.grad).all()


def test_triplet_unreduced_loss_follows_the_input_device():
    """The hinge floor is built next to the distances, not on the default device
    (meta stands in for an accelerator the suite does not use)."""
    x = torch.empty(5, 3, 8, device="meta")
    assert TripletLoss()(x, reduction=False).device.type == "meta"
