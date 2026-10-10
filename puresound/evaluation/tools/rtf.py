"""CPU real-time factor: the gate that is a hard requirement, not a trade-off.

RTF is processing time divided by audio duration. Below 1.0 the system keeps up
with real time on this machine; the deployment budget is well under that, because
the model is not the only thing running.

Measured on **CPU by default and single-threaded by default**, because that is the
constraint that actually binds: a GPU number says nothing about whether this ships,
and a number taken with every core of a workstation says nothing about a device
that has four. Pass ``--threads 0`` to use whatever torch would pick.

Run::

    python -m puresound.evaluation.tools.rtf --recipe config/dpcrn.yaml --ckpt x.ckpt
"""

from __future__ import annotations

import argparse
import time
from typing import Sequence

import numpy as np
import torch

from puresound.evaluation.records import StageResult, write_stage
from puresound.evaluation.systems import System, load_system


def measure(
    system: System,
    *,
    sample_rate: int = 16000,
    seconds: float = 10.0,
    repeats: int = 5,
    warmup: int = 2,
    seed: int = 0,
) -> list[float]:
    """Per-repeat RTF. Discards ``warmup`` runs -- the first call allocates and
    autotunes, and including it reports the setup rather than the steady state."""
    generator = torch.Generator().manual_seed(seed)
    wav = torch.randn(1, int(sample_rate * seconds), generator=generator) * 0.1

    for _ in range(warmup):
        system.process(wav, sample_rate)

    factors = []
    for _ in range(repeats):
        start = time.perf_counter()
        system.process(wav, sample_rate)
        factors.append((time.perf_counter() - start) / seconds)
    return factors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puresound.evaluation.tools.rtf",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--recipe", default=None)
    parser.add_argument("--ckpt", default=None, help="Omit to time the passthrough.")
    parser.add_argument("--task", default="noise_suppression")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-blend", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Torch CPU threads; 0 leaves torch's own choice alone.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Fail the stage above this RTF. Without it the stage is a monitor.",
    )
    parser.add_argument("--stage-name", default="cpu_rtf")
    parser.add_argument("--out", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    system = load_system(
        args.recipe,
        args.ckpt,
        task=args.task,
        device=args.device,
        dry_blend=args.dry_blend,
        name="model",
    )

    factors = measure(
        system,
        sample_rate=args.sample_rate,
        seconds=args.seconds,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    median = float(np.median(factors))
    spread = float(max(factors) - min(factors))

    role = "gate" if args.budget is not None else "monitor"
    decision = "pass"
    if args.budget is not None:
        decision = "pass" if median <= args.budget else "fail"

    stage = StageResult(
        name=args.stage_name,
        metric="rtf",
        role=role,
        n=len(factors),
        value=median,
        verdict=decision,
        direction="lower_is_better",
        notes=(
            f"{args.device}, {args.threads or 'default'} thread(s), "
            f"{args.seconds:g}s of audio"
        ),
        extra={
            "budget": args.budget,
            "spread": spread,
            "per_repeat": [round(value, 5) for value in factors],
            "threads": args.threads,
            "device": args.device,
        },
    )

    print(f"system   : {system.name} ({system.describe()})")
    print(f"device   : {args.device}, threads={args.threads or 'default'}")
    print(f"RTF      : {median:.4f} median, spread {spread:.4f} over {len(factors)} run(s)")
    if args.budget is not None:
        margin = args.budget - median
        print(f"budget   : {args.budget:.4f}  ->  {decision.upper()} (margin {margin:+.4f})")
    else:
        print("budget   : none given, so this is a monitor rather than a gate")

    if args.out:
        write_stage(args.out, stage)
        print(f"wrote {args.out}")
    return 0 if decision == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
