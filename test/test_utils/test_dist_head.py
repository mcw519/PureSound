"""Tests for the DPCRN DistHead + DistHeadRegressionLoss auxiliary."""

import math

import pytest
import torch
from pydantic import ValidationError

from puresound.nnet.dpcrn import DPCRN
from puresound.nnet.loss import DistHeadRegressionLoss


def _tiny_dpcrn(**kw):
    return DPCRN(
        input_dim=64,
        channels=(1, 8, 16),
        kernel_t=(2, 2),
        stride_t=(1, 1),
        dilation_t=(1, 1),
        kernel_f=(5, 3),
        stride_f=(2, 2),
        dilation_f=(1, 1),
        delay=(0, 0),
        rnn_hidden=16,
        **kw,
    )


def test_dist_head_forward_and_grad():
    model = _tiny_dpcrn(dist_head={"enabled": True, "hidden": 8})
    x = torch.randn(2, 1, 64, 12)
    y = model(x)
    assert y.shape[0] == 2
    assert model.last_dist_preds is not None
    assert model.last_dist_preds.shape == (2, 3)

    loss_fn = DistHeadRegressionLoss()
    batch = {
        "foreground_drr": torch.tensor([float("nan"), 5.0]),
        "foreground_distance": torch.tensor([0.74, 0.5]),
        "nearest_interferer_distance": torch.tensor([2.0, float("nan")]),
    }
    loss = loss_fn(model.last_dist_preds, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.dist_head.net[0].weight.grad is not None
    assert model.dist_head.net[0].weight.grad.abs().sum() > 0


def test_dist_head_disabled_by_default():
    model = _tiny_dpcrn()
    model(torch.randn(1, 1, 64, 12))
    assert model.dist_head is None
    assert model.last_dist_preds is None


def test_dist_loss_all_nan_batch_gives_zero_with_graph():
    preds = torch.randn(3, 3, requires_grad=True)
    loss_fn = DistHeadRegressionLoss()
    nanv = torch.full((3,), float("nan"))
    loss = loss_fn(preds, {"foreground_drr": nanv, "foreground_distance": nanv,
                           "nearest_interferer_distance": nanv})
    assert float(loss) == 0.0 and loss.requires_grad


def test_dist_loss_missing_keys_and_targets():
    preds = torch.zeros(2, 3)
    loss_fn = DistHeadRegressionLoss()
    # Only interferer distance present; correct masked value = smooth_l1(0, log10(2))
    loss = loss_fn(preds, {"nearest_interferer_distance": torch.tensor([2.0, 2.0])})
    expected = 0.5 * math.log10(2.0) ** 2  # smooth_l1 quadratic region, beta=1
    assert abs(float(loss) - expected) < 1e-6


def test_dist_loss_requires_head():
    with pytest.raises(ValueError):
        DistHeadRegressionLoss()(None, {})


# --------------------------------------------------------------------------- #
# Head blocks are validated, not `.get`-ed
# --------------------------------------------------------------------------- #

BOTTLENECK = 16  # `_tiny_dpcrn`'s channels[-1]


@pytest.mark.parametrize(
    "block,expected",
    [
        ({"enabled": True, "hiden": 64}, "hiden"),  # misspelt size
        ({"enabld": True}, "enabld"),  # misspelt gate -- head silently absent
        ({"enabled": True, "kernel_t": 0}, "kernel_t"),  # nonsensical value
    ],
)
def test_a_misspelt_head_key_is_an_error_not_a_default(block, expected):
    """`.get(key, default)` answers a typo with the default.

    Not hypothetical: `vad_head: {enabled: true, hiden: 64}` used to build a
    head at the bottleneck width instead of 64 and train a whole run that way,
    silently. The gate was worse -- a typo on `enabled` left the head off
    entirely. Same reason the augmentation blocks are `extra="forbid"`.
    """
    with pytest.raises(ValidationError, match=expected):
        _tiny_dpcrn(vad_head=block)


def test_the_head_width_still_defaults_to_the_bottleneck():
    """The one default the config model cannot hold: only the backbone knows
    how wide its own bottleneck is, so the model says None and it decides."""
    assert _tiny_dpcrn(vad_head={"enabled": True}).vad_head.proj.out_features == BOTTLENECK
    assert _tiny_dpcrn(vad_head={"enabled": True, "hidden": 64}).vad_head.proj.out_features == 64


def test_a_disabled_or_absent_block_attaches_no_head():
    assert _tiny_dpcrn(vad_head={"enabled": False}).vad_head is None
    assert _tiny_dpcrn().vad_head is None
    assert _tiny_dpcrn().dist_head is None
