"""Merge the stages' results into one record, and say what the gate decided.

Each stage writes its own file so a stage can be re-run on its own without
re-running the rest. This is the step that turns them into the record the
benchmark directory keeps.

Run::

    python -m puresound.evaluation.tools.collect --tag candidate \\
        --checkpoint exp/.../epoch=39.ckpt --recipe config/dpcrn.yaml \\
        --out benchmarks/records/candidate.json /tmp/gate_candidate/*.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from puresound.evaluation.records import GateRecord, chain_commit, merge_stage_files


def parse_inference(values: Sequence[str]) -> dict[str, Any]:
    """``dry_blend=0.9`` -> ``{"dry_blend": 0.9}``, numbers and booleans parsed."""
    parsed: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"expected NAME=VALUE, got {item!r}")
        name, raw = item.split("=", 1)
        lowered = raw.strip().lower()
        if lowered in ("true", "false"):
            parsed[name] = lowered == "true"
            continue
        try:
            parsed[name] = float(raw) if any(c in raw for c in ".eE") else int(raw)
        except ValueError:
            parsed[name] = raw
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puresound.evaluation.tools.collect",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("stage_files", nargs="+")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--inference", action="append", default=[])
    parser.add_argument(
        "--require-stage", action="append", default=[],
        help="Required stage name; repeatable. Collection fails if one is absent.",
    )
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    missing_files = [path for path in args.stage_files if not Path(path).is_file()]
    if missing_files:
        print("Missing stage result file(s): " + ", ".join(missing_files), file=sys.stderr)
        return 1

    stages = merge_stage_files(args.stage_files)
    if not stages:
        print("No stage results found; nothing to record.", file=sys.stderr)
        return 1

    stage_names = {stage.name for stage in stages}
    missing_stages = sorted(set(args.require_stage) - stage_names)
    if missing_stages:
        print("Missing required stage(s): " + ", ".join(missing_stages), file=sys.stderr)
        return 1

    record = GateRecord(
        tag=args.tag,
        checkpoint=args.checkpoint,
        recipe=args.recipe,
        chain_commit=chain_commit(),
        inference=parse_inference(args.inference),
        stages=stages,
    )
    record.write(args.out)
    print(record.summary())
    print(f"\nwrote {args.out}")

    # A failed gate has to be visible to whatever ran the driver, not only in a file.
    return 0 if record.verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
