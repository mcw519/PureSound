"""Prove a checkpoint loads whole into every recipe the benchmark will use.

Each stage builds its model from a recipe of its own. If one of them disagrees with
the checkpoint, that stage silently scores a partly untrained model and reports a
regression nobody can explain. Checking all of them first costs seconds and is the
difference between a wrong number and no number.

Run::

    python -m puresound.evaluation.tools.preflight --ckpt x.ckpt config/*.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from puresound.evaluation.systems import load_checkpoint_system


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puresound.evaluation.tools.preflight",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("recipes", nargs="+", help="Every recipe the benchmark uses.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--task", default="noise_suppression")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    failures = 0

    for recipe in args.recipes:
        name = Path(recipe).name
        try:
            system = load_checkpoint_system(recipe, args.ckpt, task=args.task)
        except Exception as error:  # noqa: BLE001 -- the whole point is to report it
            print(f"FAIL  {name}: {error}", file=sys.stderr)
            failures += 1
            continue

        unused = system.unused_checkpoint_params
        suffix = f" ({len(unused)} checkpoint param(s) unused here)" if unused else ""
        print(f"OK    {name}: every model parameter loaded{suffix}")

    if failures:
        print(f"\n{failures} recipe(s) would score a partly untrained model.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
