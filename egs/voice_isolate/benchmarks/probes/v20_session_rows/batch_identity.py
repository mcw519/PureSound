"""Does an enabled-but-inert `augmentation_session_rows` change a single batch?

The block's first contract is that a recipe carrying it produces the same rows a
recipe without it produces, wherever the block cannot fire. Two ways it cannot:

* ``prob: 0`` -- the probability draw is inside the short circuit, so no row
  consumes randomness;
* ``min_seconds`` above every bucket -- eligibility is tested BEFORE the draw,
  which is what makes the short buckets of the R1a recipe comparable to v16's.

Both are compared here against the shipped v16 recipe over N whole batches from
the REAL train loader, tensor by tensor. Determinism is established first (the
same config twice), because a comparison between two configs is worthless if the
loader is not reproducible to begin with.

    cd egs/voice_isolate
    uv run python benchmarks/probes/v20_session_rows/batch_identity.py \
        --batches 20 --num-workers 8 --seed 7
"""
from __future__ import annotations
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

RECIPE_DIR = Path("/home/milowu/A4Audio/PureSound/egs/voice_isolate")
REPO_ROOT = Path("/home/milowu/A4Audio/PureSound")
sys.path.insert(0, str(REPO_ROOT))

import egs.voice_isolate.main as recipe_main  # noqa: E402
from puresound.config import load_recipe, with_overrides  # noqa: E402

#: Every tensor key a row can carry that the block could plausibly disturb.
COMPARED = (
    "noisy_speech", "clean_speech", "consistency_noise", "far_target",
    "vad_target", "background_vad_target", "spkid", "length",
    "foreground_distance", "foreground_drr", "nearest_interferer_distance",
    "strongest_interferer_drr", "n_interferers", "target_absent",
    "realized_speech_sir", "noise_snr", "overlap_fraction", "turn_taking",
    "mix_mode", "far_count", "rt60", "volume_gain", "hpf_cutoff", "src_target_sr",
)


def batches(recipe, n: int, seed: int, workers: int):
    """N batches from the real train loader, under a fixed global seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    recipe = with_overrides(recipe, trainer={"num_workers": workers})
    train_dl, _ = recipe_main.init_dataloader(recipe)
    out = []
    for index, batch in enumerate(train_dl):
        if index >= n:
            break
        out.append({k: v for k, v in batch.items() if k in COMPARED})
    return out


def compare(left, right, label: str) -> int:
    bad = 0
    for index, (a, b) in enumerate(zip(left, right)):
        keys = set(a) | set(b)
        for key in sorted(keys):
            if key not in a or key not in b:
                print(f"  [{label}] batch {index}: key {key} on one side only")
                bad += 1
                continue
            x, y = a[key], b[key]
            if x.shape != y.shape:
                print(f"  [{label}] batch {index}: {key} shape {x.shape} vs {y.shape}")
                bad += 1
                continue
            same = torch.equal(x, y) or bool(
                torch.isnan(x).eq(torch.isnan(y)).all()
                and torch.equal(x[~torch.isnan(x)], y[~torch.isnan(y)])
            )
            if not same:
                delta = float((x - y).abs().max())
                print(f"  [{label}] batch {index}: {key} differs, max|d| = {delta:.3e}")
                bad += 1
    print(f"  [{label}] {len(left)} batches, {bad} mismatching tensors")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(RECIPE_DIR / "config/exp/train_dpcrn_v16_lengthmix.yaml"))
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.chdir(RECIPE_DIR)

    base = load_recipe(args.config, expected_task="voice_isolation", expected_purpose="train")
    variants = {
        "determinism (same config twice)": base,
        "enabled, prob 0": with_overrides(
            base, augmentation_session_rows={"enabled": True, "prob": 0.0}
        ),
        "enabled, prob 1, min_seconds 60": with_overrides(
            base,
            augmentation_session_rows={
                "enabled": True, "prob": 1.0, "min_seconds": 60.0,
                "rir_move_prob": 1.0, "pair_prob": 1.0,
            },
        ),
    }
    reference = batches(base, args.batches, args.seed, args.num_workers)
    rows = sum(b["noisy_speech"].shape[0] for b in reference)
    print(f"reference: {len(reference)} batches, {rows} rows, seed {args.seed}")
    failures = 0
    for label, recipe in variants.items():
        candidate = batches(recipe, args.batches, args.seed, args.num_workers)
        failures += compare(reference, candidate, label)
    print("RESULT:", "IDENTICAL" if failures == 0 else f"{failures} MISMATCHES")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
