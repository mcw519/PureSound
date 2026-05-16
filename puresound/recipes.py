from typing import Dict, List, Tuple

import torch

from puresound import nnet, system
from puresound.nnet import loss as ploss
from puresound.utils import load_hparam


def _enabled_config(config: Dict, key: str):
    item = config.get(key)
    if item and item.get("used"):
        return item
    return None


def load_siso_recipe_config(f_path: str) -> Tuple:
    config = load_hparam(file_path=f_path)

    return (
        config["dataset"],
        config["trainer"],
        config["optimizer"],
        config["scheduler"],
        config["loss_func"],
        config["model"],
        _enabled_config(config, "augmentation_speech"),
        _enabled_config(config, "augmentation_noise"),
        _enabled_config(config, "augmentation_reverb"),
        _enabled_config(config, "augmentation_speed"),
        _enabled_config(config, "augmentation_ir_response"),
        _enabled_config(config, "augmentation_src"),
        config.get("augmentation_hpf"),
        config.get("augmentation_volume"),
        config.get("vad_label"),
    )


def init_siso_model(model_dict: Dict):
    lightning_module = getattr(system, model_dict["lighting_module"]["type"])
    encoder = getattr(nnet, model_dict["encoder"]["type"])(
        **model_dict["encoder"]["encoder_args"]
    )

    if "freq_eq" in model_dict:
        peq_module = getattr(nnet, model_dict["freq_eq"]["type"])(
            **model_dict["freq_eq"]["eq_args"]
        )
        model_dict["features"]["peq_module"] = peq_module

    feature_encoder = nnet.FeatureEncoder(**model_dict["features"])
    backbone = getattr(nnet, model_dict["backbone"]["type"])(
        **model_dict["backbone"]["backbone_args"]
    )
    return lightning_module(
        encoder,
        feature_encoder,
        backbone,
        **model_dict["lighting_module"]["module_args"],
    )


def init_loss_func(hparam_conf: List):
    loss_list = torch.nn.ModuleList([])
    loss_list_w = []
    for item in hparam_conf:
        loss_func = getattr(ploss, item["type"])(**item["args"])
        loss_list.append(loss_func)
        loss_list_w.append(item["weighted"])

    return loss_list, loss_list_w
