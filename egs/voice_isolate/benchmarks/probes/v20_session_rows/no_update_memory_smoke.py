"""Measure v20 forward/backward memory without performing an optimizer update.

This is intentionally a dry smoke test. It builds the production dataloader and
module, runs at most ``--steps`` training steps (including the paired-view and
channel-consistency forwards), calls ``backward()``, then discards gradients.
There is no Lightning ``Trainer.fit`` and no ``optimizer.step``. The parameter
snapshot at the end is an additional guard against accidentally turning this
probe into training.

Example (run from ``egs/voice_isolate``)::

    uv run python benchmarks/probes/v20_session_rows/no_update_memory_smoke.py \
      --config config/exp/train_dpcrn_v20_r1a.yaml --seconds 12 --n-spk 6 \
      --steps 2 --gpu 0 --out /tmp/v20-12s.json
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
RECIPE_DIR = REPO_ROOT / "egs" / "voice_isolate"
sys.path.insert(0, str(REPO_ROOT))


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def validate_coverage(case, effective):
    """A paired memory result is valid only if the measured step used a pair."""
    if case == "paired" and effective <= 0:
        raise RuntimeError("no effective consistency pair in measured step; memory result is not valid")


def coverage_recipe(recipe, seconds):
    """Use identical forced session/view draws in both cases; baseline drops views."""
    from puresound.config import with_overrides
    session = recipe.augmentation_session_rows
    if session is None or not session.enabled or not (
        max(session.min_seconds, session.paired_view_min_seconds) <= seconds <= session.max_seconds
    ):
        raise ValueError("smoke needs a session-enabled bucket within its configured duration limits")
    return with_overrides(recipe, augmentation_session_rows={"prob": 1.0, "paired_view_prob": 1.0})


def run(args):
    if args.steps < 1 or args.steps > 30:
        raise ValueError("--steps must be between 1 and 30")
    if not torch.cuda.is_available():
        raise RuntimeError("this smoke test requires a CUDA device; it does not train")

    os.chdir(RECIPE_DIR)
    from puresound.config import load_recipe, with_overrides
    from puresound.recipes import init_loss_func, init_model_for_task
    import egs.voice_isolate.main as recipe_main

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    recipe = load_recipe(args.config, expected_task="voice_isolation", expected_purpose="train")
    recipe = with_overrides(
        recipe,
        trainer={
            "num_workers": args.num_workers,
            "length_schedule": [{"seconds": args.seconds, "n_spk": args.n_spk, "prob": 1.0}],
        },
    )
    recipe = coverage_recipe(recipe, args.seconds)
    train_loader, _ = recipe_main.init_dataloader(recipe)

    losses, weights = init_loss_func(recipe.loss_func)
    module = init_model_for_task(recipe.task)(recipe.model)
    module.register_loss_func(losses, weights)
    module.cuda(args.gpu)
    if args.ckpt:
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = blob.get("state_dict", blob)
        module.load_state_dict(state, strict=False)
        module.cuda(args.gpu)

    # Direct training_step calls are deliberate: Lightning's fit loop would
    # invoke the optimizer. The small logger shim keeps the module contract
    # intact while this probe remains independent of a Trainer.
    step_logs = {}
    module.log = lambda name, value, **kw: step_logs.__setitem__(name, float(value))
    module.puresound_logging.update = lambda *a, **kw: None
    module.train()
    before = [parameter.detach().cpu().clone() for parameter in module.parameters()]
    device = torch.device("cuda", args.gpu)
    use_amp = "bf16" in str(recipe.trainer.lightning_trainer_args.get("precision", ""))
    torch.cuda.reset_peak_memory_stats(device)
    timings, losses_seen = [], []
    paired_rows_seen = 0
    paired_steps = 0
    effective_pairs = 0
    actual_view_rows = 0
    module.zero_grad(set_to_none=True)
    iterator = iter(train_loader)
    started = time.perf_counter()
    for step in range(args.steps):
        batch = _move(next(iterator), device)
        step_logs.clear()
        if args.case == "baseline":
            batch.pop("paired_view", None)
        if batch.get("paired_view") is not None:
            paired_steps += 1
            paired_rows_seen += int(batch["paired_view"]["source_indices"].numel())
            config = module.paired_view_consistency
            if config is not None:
                actual_view_rows += min(config.max_rows, int(batch["paired_view"]["source_indices"].numel()))
        t0 = time.perf_counter()
        amp_context = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
        with amp_context:
            result = module.training_step(batch, step)
            loss = result["loss"] if isinstance(result, dict) else result
            (loss / max(1, args.accumulate)).backward()
        effective = sum(value for name, value in step_logs.items()
                        if name.startswith("train_paired_") and name.endswith("_count"))
        validate_coverage(args.case, effective)
        effective_pairs += effective
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - t0)
        losses_seen.append(float(loss.detach().cpu()))
        if (step + 1) % max(1, args.accumulate) == 0:
            # Clearing gradients releases graph storage. This is explicitly not
            # optimizer.step(); parameters must remain bit-identical.
            module.zero_grad(set_to_none=True)
    module.zero_grad(set_to_none=True)
    unchanged = all(torch.equal(old, new.detach().cpu()) for old, new in zip(before, module.parameters()))
    if not unchanged:
        raise RuntimeError("smoke changed model parameters")
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    reserved = torch.cuda.max_memory_reserved(device) / 2**30
    optimizer_state_gib = 2 * sum(p.numel() * p.element_size() for p in module.parameters() if p.requires_grad) / 2**30
    return {
        "config": str(Path(args.config).resolve()),
        "case": args.case,
        "seed": args.seed,
        "effective_pairs": effective_pairs,
        "actual_view_rows": actual_view_rows,
        "seconds": args.seconds,
        "n_spk": args.n_spk,
        "steps": args.steps,
        "accumulate_grad_batches": args.accumulate,
        "precision": recipe.trainer.lightning_trainer_args.get("precision", "32-true"),
        "paired_view_enabled": bool(recipe.model["lightning_module"]["module_args"].get("paired_view_consistency", {}).get("enabled", False)),
        "paired_steps": paired_steps,
        "paired_rows_seen": paired_rows_seen,
        "peak_allocated_gib": round(peak, 3),
        "peak_reserved_gib": round(reserved, 3),
        "adamw_state_estimate_gib": round(optimizer_state_gib, 3),
        "wall_seconds": round(time.perf_counter() - started, 2),
        "step_seconds": [round(value, 3) for value in timings],
        "losses": losses_seen,
        "parameters_unchanged": unchanged,
        "optimizer_step_called": False,
        "note": "forward/backward only; no Trainer.fit and no optimizer update",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(RECIPE_DIR / "config/exp/train_dpcrn_v20_r1a.yaml"))
    parser.add_argument("--case", choices=("baseline", "paired"), default="paired",
                        help="run separately for independent peaks; baseline drops forced auxiliary views")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--n-spk", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = run(args)
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.out:
        Path(args.out).write_text(rendered)


if __name__ == "__main__":
    main()
