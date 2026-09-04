"""Peak GPU memory for <=30 optimiser steps of one length bucket.

Item 5 of the v20 review: session rows must be costed before they are trained,
because a session cuts the number of independent scenes per optimiser update and
because the shipped recipe already peaks at 19.0-19.2 GiB on a 23 GB card. R1a's
answer is "a session is a long row inside the existing schedule", so what has to
be shown is that the 30 s bucket is unchanged -- not that a new bucket fits.

This mirrors `runner.run_training` (same module, same seven losses including the
two SSL encoders, same channel-consistency second forward, same
accumulate_grad_batches and bf16-mixed) and differs from it only in what a smoke
test must differ in: ONE device, no DDP, no checkpointing, no validation, and
`limit_train_batches` steps. Read it as a lower bound on the DDP peak the header
numbers were measured at, not as the same number.

    cd egs/voice_isolate
    uv run python benchmarks/probes/v20_session_rows/memory_smoke.py \
        --config config/exp/train_dpcrn_v20_r1a.yaml --seconds 30 --n-spk 2 \
        --steps 30 --gpu 1
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import lightning as L
import torch

RECIPE_DIR = Path("/home/milowu/A4Audio/PureSound/egs/voice_isolate")
REPO_ROOT = Path("/home/milowu/A4Audio/PureSound")
sys.path.insert(0, str(REPO_ROOT))

import egs.voice_isolate.main as recipe_main  # noqa: E402
from puresound.config import load_recipe, with_overrides  # noqa: E402
from puresound.recipes import init_loss_func, init_model_for_task  # noqa: E402
from puresound.system.optim import create_optimizer_and_scheduler  # noqa: E402


class PeakMemory(L.Callback):
    """Peak allocated/reserved bytes and wall time per batch.

    The counters are reset in ``on_train_start`` and read with no device
    argument: CUDA is not initialised before ``fit``, and indexing a device
    before init raises "Invalid device argument". By then Lightning has already
    set the process's current device to the one it was given.

    The per-batch wall time is measured start-of-batch to end-of-batch, so it
    includes the dataloader wait -- which is what a training run pays. The
    first batch is reported separately because it carries the workers' warm-up
    and cuDNN's autotuning.
    """

    def __init__(self):
        self.allocated = 0
        self.reserved = 0
        self.steps = 0
        self.device = None
        self.batch_times: list[float] = []
        self._t0 = None

    def on_train_start(self, trainer, module):
        del trainer
        self.device = str(module.device)
        torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        del trainer, module, batch, batch_idx
        self._t0 = time.perf_counter()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        del trainer, module, outputs, batch, batch_idx
        self.steps += 1
        if self._t0 is not None:
            self.batch_times.append(time.perf_counter() - self._t0)
        self.allocated = max(self.allocated, torch.cuda.max_memory_allocated())
        self.reserved = max(self.reserved, torch.cuda.max_memory_reserved())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(RECIPE_DIR / "config/exp/train_dpcrn_v20_r1a.yaml"))
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--n-spk", type=int, required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--disable-session-rows", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.chdir(RECIPE_DIR)
    if args.steps > 30:
        raise SystemExit("this is a smoke test: <= 30 optimiser steps")

    recipe = load_recipe(args.config, expected_task="voice_isolation",
                         expected_purpose="train")
    overrides = {
        "num_workers": args.num_workers,
        "length_schedule": [
            {"seconds": args.seconds, "n_spk": args.n_spk, "prob": 1.0}
        ],
    }
    recipe = with_overrides(recipe, trainer=overrides)
    if args.disable_session_rows and recipe.augmentation_session_rows is not None:
        recipe = with_overrides(recipe, augmentation_session_rows={"enabled": False})

    train_dl, _ = recipe_main.init_dataloader(recipe)

    # --- the module, exactly as runner.run_training assembles it -------- #
    init_model = init_model_for_task(recipe.task)
    loss_func_list, loss_weights = init_loss_func(recipe.loss_func)
    module = init_model(recipe.model)
    module.register_loss_func(loss_func_list, loss_weights)
    optimizer, scheduler = create_optimizer_and_scheduler(
        overall_params_and_lr_factor=module.get_total_param_groups(),
        optimizer_args=recipe.optimizer,
        scheduler_args=recipe.scheduler,
    )
    module.register_optimizer(optimizer)
    module.register_scheduler(scheduler)
    module.register_warmup_step(recipe.scheduler.warmup_step)

    peak = PeakMemory()
    trainer_args = dict(recipe.trainer.lightning_trainer_args)
    trainer_args["max_epochs"] = 1
    trainer = L.Trainer(
        **trainer_args,
        accelerator="gpu",
        devices=[args.gpu],
        strategy="auto",
        limit_train_batches=args.steps,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        enable_checkpointing=False,
        logger=False,
        callbacks=[peak],
    )
    trainer.fit(module, train_dataloaders=train_dl)

    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    times = peak.batch_times
    steady = times[1:] if len(times) > 1 else times
    result = {
        "config": args.config,
        "session_rows": bool(
            recipe.augmentation_session_rows is not None
            and recipe.augmentation_session_rows.enabled
        ),
        "seconds": args.seconds,
        "n_spk": args.n_spk,
        "rows_per_step": args.n_spk * recipe.trainer.n_utt_per_speaker,
        "accumulate_grad_batches": trainer_args.get("accumulate_grad_batches"),
        "precision": trainer_args.get("precision"),
        "steps": peak.steps,
        "peak_allocated_gib": round(peak.allocated / 2**30, 3),
        "peak_reserved_gib": round(peak.reserved / 2**30, 3),
        "batch_time_median_s": round(statistics.median(steady), 3) if steady else None,
        "batch_time_mean_s": round(statistics.fmean(steady), 3) if steady else None,
        "batch_time_first_s": round(times[0], 3) if times else None,
        "batch_time_total_s": round(sum(times), 1) if times else None,
        "device_total_gib": round(total / 2**30, 3),
        "devices": 1,
        "device": peak.device,
        "note": "single device, no DDP; the v16 header numbers are 2-GPU DDP",
    }
    print(json.dumps(result))
    if args.out:
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
