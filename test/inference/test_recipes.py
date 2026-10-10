"""``puresound.recipes``: shipped recipe configs build their model, and a loss is
built from its recipe entry by name with its weight and arguments."""

from pathlib import Path

import pytest
import torch

from puresound.config import load_recipe
from puresound.config.recipe import LossConfig
from puresound.nnet.loss import AnchorInheritanceLoss, VADActivityLoss
from puresound.recipes import init_loss_func, init_siso_model

REPO_ROOT = Path(__file__).resolve().parents[2]

NOISE_SUPPRESSION_CONFIGS = sorted(
    (REPO_ROOT / "egs" / "noise_suppression" / "config").glob("*.yaml")
)


@pytest.mark.parametrize(
    "config_path", NOISE_SUPPRESSION_CONFIGS, ids=lambda p: p.stem
)
def test_init_siso_model_from_recipe_config(config_path):
    model = init_siso_model(load_recipe(config_path).model)
    assert isinstance(model, torch.nn.Module)


@pytest.mark.parametrize(
    "cls, weight, args, attribute, value",
    [
        (VADActivityLoss, 0.1,
         {"frame_length": 160, "hop_length": 80, "activity_threshold_db": -35},
         None, None),
        (AnchorInheritanceLoss, 0.25, {"margin_db": 6.0}, "margin_db", 6.0),
    ],
    ids=lambda v: v.__name__ if isinstance(v, type) else None,
)
def test_a_recipe_names_a_loss_with_its_weight_and_arguments(cls, weight, args, attribute, value):
    losses, weights = init_loss_func([LossConfig(type=cls.__name__, weighted=weight, args=args)])
    assert isinstance(losses[0], cls)
    assert weights == [weight]
    if attribute is not None:
        assert getattr(losses[0], attribute) == value
