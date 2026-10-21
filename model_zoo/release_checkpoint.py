"""Turn a training checkpoint into a release checkpoint.

A Lightning checkpoint carries the critic, the loss modules' buffers, the
optimizer, the scheduler and the sampler position. A release carries only what
inference loads: the model's weights, plus a provenance block that names the
source run, the recipes and the benchmark record. The weights are copied
untouched, so the release computes bit for bit what the run did.

usage (.venv, repo root):
  python model_zoo/release_checkpoint.py RUN_CKPT OUT_CKPT --released-as ID
      --training-recipe PATH --inference-recipe PATH --benchmark-record PATH
      [--dry-blend 1.0] [--drop-prefix loss_func_list. --drop-prefix metric_disc.]
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

DEFAULT_DROP = ("loss_func_list.", "metric_disc.")
KEEP_TOP = ("epoch", "global_step", "pytorch-lightning_version", "state_dict", "hyper_parameters")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def release(run_ckpt: Path, out: Path, provenance: dict, drop: tuple[str, ...]) -> dict:
    ckpt = torch.load(run_ckpt, map_location="cpu", weights_only=False)
    weights = {k: v for k, v in ckpt["state_dict"].items() if not k.startswith(drop)}
    dropped = len(ckpt["state_dict"]) - len(weights)
    out_ckpt = {k: ckpt[k] for k in KEEP_TOP if k in ckpt}
    out_ckpt["state_dict"] = weights
    out_ckpt["release_provenance"] = dict(provenance, source_checkpoint_sha256=sha256(run_ckpt))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, out)
    return {"kept": len(weights), "dropped": dropped, "sha256": sha256(out)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_ckpt", type=Path)
    ap.add_argument("out_ckpt", type=Path)
    ap.add_argument("--released-as", required=True)
    ap.add_argument("--training-recipe", required=True)
    ap.add_argument("--inference-recipe", required=True)
    ap.add_argument("--benchmark-record", required=True)
    ap.add_argument("--dry-blend", type=float, default=1.0)
    ap.add_argument("--drop-prefix", action="append", default=None,
                    help="state_dict prefixes that are training-only (default: %s)" % ", ".join(DEFAULT_DROP))
    a = ap.parse_args()
    provenance = {
        "released_as": a.released_as,
        "source_checkpoint": str(a.run_ckpt),
        "training_recipe": a.training_recipe,
        "inference_recipe": a.inference_recipe,
        "benchmark_record": a.benchmark_record,
        "dry_blend": a.dry_blend,
    }
    info = release(a.run_ckpt, a.out_ckpt, provenance, tuple(a.drop_prefix or DEFAULT_DROP))
    print(f"{a.out_ckpt}: kept {info['kept']} tensors, dropped {info['dropped']}, sha256 {info['sha256']}")


if __name__ == "__main__":
    main()
