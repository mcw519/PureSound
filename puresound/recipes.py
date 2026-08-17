"""Building the objects a recipe names.

Config *loading* lives in ``puresound.config``; this module only turns an
already-validated recipe's ``model`` / ``loss_func`` sections into objects.
The twenty-tuple ``load_siso_recipe_config`` that used to live here is gone --
callers take a typed ``Recipe`` from ``puresound.config.load_recipe`` and read
it by name.
"""

from typing import Dict, List

import torch

from puresound import nnet, system
from puresound.config.recipe import LossConfig
from puresound.nnet import loss as ploss


def init_siso_model(model_dict: Dict):
    lightning_module = getattr(system, model_dict["lightning_module"]["type"])
    encoder = getattr(nnet, model_dict["encoder"]["type"])(
        **model_dict["encoder"]["encoder_args"]
    )

    feature_args = dict(model_dict["features"])
    if "freq_eq" in model_dict:
        peq_module = getattr(nnet, model_dict["freq_eq"]["type"])(
            **model_dict["freq_eq"]["eq_args"]
        )
        feature_args["peq_module"] = peq_module

    feature_encoder = nnet.FeatureEncoder(**feature_args)
    backbone = getattr(nnet, model_dict["backbone"]["type"])(
        **model_dict["backbone"]["backbone_args"]
    )
    return lightning_module(
        encoder,
        feature_encoder,
        backbone,
        **model_dict["lightning_module"]["module_args"],
    )


def init_loss_func(loss_configs: List[LossConfig]):
    loss_list = torch.nn.ModuleList([])
    loss_list_w = []
    for item in loss_configs:
        loss_func = getattr(ploss, item.type)(**item.args)
        loss_list.append(loss_func)
        loss_list_w.append(item.weighted)

    return loss_list, loss_list_w
