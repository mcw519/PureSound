import torch

from puresound.system.optim import create_optimizer_and_scheduler


def test_create_optimizer_and_scheduler_applies_lr_factors():
    layer_a = torch.nn.Linear(2, 2)
    layer_b = torch.nn.Linear(2, 1)

    optimizer, scheduler = create_optimizer_and_scheduler(
        {
            "a": {"params": layer_a.parameters(), "lr_factor": 1.0},
            "b": {"params": layer_b.parameters(), "lr_factor": 0.25},
        },
        optimizer_args={
            "type": "AdamW",
            "learning_rate": 0.01,
            "args": {"weight_decay": 0.001},
        },
        scheduler_args={"type": "StepLR", "args": {"step_size": 2, "gamma": 0.5}},
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert [group["lr"] for group in optimizer.param_groups] == [0.01, 0.0025]
