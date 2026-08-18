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


def init_miso_model(model_dict: Dict):
    """A conditioned (MISO) module: the mixture branch plus a conditioning branch.

    Same `model` block shape as `init_siso_model`, plus `c_encoder` / `c_features`
    / `c_backbone` for the branch that embeds the enrollment. `siamese_encoder`
    means the mixture's own encoder and features are reused for it, so those two
    sections are absent and the arguments go through as None.
    """
    lightning_module = getattr(system, model_dict["lightning_module"]["type"])

    encoder = getattr(nnet, model_dict["encoder"]["type"])(
        **model_dict["encoder"]["encoder_args"]
    )
    if not model_dict["lightning_module"]["module_args"]["siamese_encoder"]:
        c_encoder = getattr(nnet, model_dict["c_encoder"]["type"])(
            **model_dict["c_encoder"]["encoder_args"]
        )
    else:
        c_encoder = None

    feature_args = dict(model_dict["features"])
    if "freq_eq" in model_dict:
        peq_module = getattr(nnet, model_dict["freq_eq"]["type"])(
            **model_dict["freq_eq"]["eq_args"]
        )
        # register peq inside the feature module
        feature_args["peq_module"] = peq_module

    feature_encoder = nnet.FeatureEncoder(**feature_args)
    if not model_dict["lightning_module"]["module_args"]["siamese_encoder"]:
        c_feature_encoder = nnet.FeatureEncoder(**model_dict["c_features"])
    else:
        c_feature_encoder = None

    backbone = getattr(nnet, model_dict["backbone"]["type"])(
        **model_dict["backbone"]["backbone_args"]
    )
    c_backbone = getattr(nnet, model_dict["c_backbone"]["type"])(
        **model_dict["c_backbone"]["backbone_args"]
    )

    model = lightning_module(
        encoder,
        feature_encoder,
        backbone,
        c_backbone,
        c_encoder=c_encoder,
        c_feats=c_feature_encoder,
        **model_dict["lightning_module"]["module_args"],
    )
    return model

#: Which module shape a task's ``model`` block builds. Derived from the task
#: rather than passed in by each entry point: the two shapes take different
#: sections (`c_encoder` / `c_backbone` only exist for the conditioned one), so
#: a caller that picked the wrong one got a `KeyError` deep inside -- and only
#: once it had the corpus to get that far.
MODEL_FACTORY_FOR_TASK = {
    "noise_suppression": init_siso_model,
    "voice_isolation": init_siso_model,
    "speaker_embedding": init_siso_model,
    "target_speaker_extraction": init_miso_model,
}


def init_model_for_task(task: str):
    """The model factory a task's recipe needs."""
    try:
        return MODEL_FACTORY_FOR_TASK[task]
    except KeyError:
        raise KeyError(
            f"no model factory registered for task {task!r}; "
            f"known tasks are {sorted(MODEL_FACTORY_FOR_TASK)}"
        ) from None


def init_loss_func(loss_configs: List[LossConfig]):
    loss_list = torch.nn.ModuleList([])
    loss_list_w = []
    for item in loss_configs:
        loss_func = getattr(ploss, item.type)(**item.args)
        loss_list.append(loss_func)
        loss_list_w.append(item.weighted)

    return loss_list, loss_list_w
