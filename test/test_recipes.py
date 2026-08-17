from pathlib import Path

import pytest
import torch

from puresound.config import load_recipe
from puresound.recipes import init_siso_model

REPO_ROOT = Path(__file__).resolve().parents[1]

NOISE_SUPPRESSION_CONFIGS = sorted(
    (REPO_ROOT / "egs" / "noise_suppression" / "config").glob("*.yaml")
)


@pytest.mark.parametrize(
    "config_path", NOISE_SUPPRESSION_CONFIGS, ids=lambda p: p.stem
)
def test_init_siso_model_from_recipe_config(config_path):
    model = init_siso_model(load_recipe(config_path).model)
    assert isinstance(model, torch.nn.Module)
