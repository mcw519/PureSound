from typing import Dict

import torch

from puresound.config.recipe import OptimizerConfig, SchedulerConfig


def create_optimizer_and_scheduler(
    overall_params_and_lr_factor: Dict,
    optimizer_args: OptimizerConfig,
    scheduler_args: SchedulerConfig,
):
    overall_params = []
    for key in overall_params_and_lr_factor.keys():
        overall_params.append(
            {
                "params": overall_params_and_lr_factor[key]["params"],
                "lr": overall_params_and_lr_factor[key]["lr_factor"]
                * optimizer_args.learning_rate,
            }
        )

    optim_cls = getattr(torch.optim, optimizer_args.type)
    optimizer = optim_cls(
        overall_params, optimizer_args.learning_rate, **optimizer_args.args
    )

    scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_args.type)
    scheduler = scheduler_cls(optimizer, **scheduler_args.args)
    return optimizer, scheduler
