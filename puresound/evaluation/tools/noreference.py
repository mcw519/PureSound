"""No-reference scoring, with the per-category breakdown a mean hides.

The DNS dev set ships no clean reference, so it is scored no-reference -- DNSMOS
P.835, which is what the challenge reports and therefore the one number here that
is comparable to anything published.

Two things keep that from being taken for more than it is:

- It is scored **paired against the unprocessed clip**, with an interval. DNSMOS on
  its own does not say whether a system helped; a distortion-free passthrough scores
  well by doing nothing.
- It is broken down by tag. Noise suppression fails per class -- it will hold up on
  a fan and fall over on a baby crying -- and a mean over classes reports neither.
  The DNS dev set's filenames carry the class, which is why the inventory has it.

Run::

    python -m puresound.evaluation.tools.noreference --inventory data/dns5/dns5_devset.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch

from puresound.audio.io import AudioIO
from puresound.evaluation.records import StageResult, write_stage
from puresound.evaluation.statistics import paired_bootstrap_ci, verdict
from puresound.evaluation.parallel import default_jobs, map_items
from puresound.evaluation.systems import (
    Passthrough,
    PrecomputedSystem,
    System,
    load_system,
    run_system,
)
from puresound.dataset.corpus.records import AudioRecord, read_inventory


#: DNSMOS returns four scores; SIG and BAK are kept apart on purpose. OVRL alone
#: cannot tell "removed the noise" from "removed the speech", and those are the two
#: failure directions of this task.
SCORES = ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovr", "dnsmos_p808")
DEFAULT_GATE_SCORE = "dnsmos_ovr"


def score_clip(wav: torch.Tensor, sample_rate: int) -> dict[str, float]:
    from puresound.metrics import Metrics

    # One ORT thread per worker: the pool already pins torch to one thread, and
    # onnxruntime does not follow that setting.
    return Metrics.dnsmos_p835(clean=None, enhanced=wav, sr=sample_rate, num_threads=1)


def _build_systems(spec: dict[str, Any]) -> dict[str, Any]:
    """Construct the systems once per worker, not once per clip."""
    systems: list[System] = [Passthrough()]
    if spec.get("ckpt"):
        systems.append(
            load_system(
                spec["recipe"], spec["ckpt"], task=spec["task"],
                device=spec["device"], dry_blend=spec["dry_blend"], name="model",
            )
        )
    if spec.get("precomputed"):
        systems.append(PrecomputedSystem(Path(spec["precomputed"]), spec["precomputed_name"]))
    return {"systems": systems, "sample_rate": spec["sample_rate"]}


def _score_clip(record: AudioRecord, state: dict[str, Any]) -> dict[str, dict[str, float]]:
    wav, file_rate = AudioIO.open(str(record.path), resample_to=state["sample_rate"])
    wav = wav.reshape(1, -1)
    key = Path(record.path).stem
    return {
        system.name: score_clip(run_system(system, key, wav, file_rate), file_rate)
        for system in state["systems"]
    }


def run(
    records: Sequence[AudioRecord],
    spec: dict[str, Any],
    *,
    sample_rate: int = 16000,
    limit: int | None = None,
    jobs: int | None = None,
    progress_every: int = 50,
) -> dict[str, list[dict[str, float]]]:
    """Score every record through every system, in worker processes.

    DNSMOS is the slowest thing in the gate, and it is CPU-bound -- so this is
    the stage that gains most from not running on one core.
    """
    chosen = list(records)[:limit] if limit else list(records)
    spec = {**spec, "sample_rate": sample_rate}
    per_clip = map_items(
        _score_clip, chosen, jobs=jobs,
        builder=lambda: _build_systems(spec),
        progress=lambda done, total: print(f"  {done}/{total}", flush=True),
        progress_every=progress_every,
    )
    names = list(per_clip[0]) if per_clip else []
    return {name: [row[name] for row in per_clip] for name in names}


def by_tag(
    records: Sequence[AudioRecord],
    treatment: Sequence[dict[str, float]],
    baseline: Sequence[dict[str, float]],
    *,
    tag: str,
    score: str,
) -> dict[str, tuple[int, float]]:
    """Mean paired delta per tag value, so a class-specific failure is visible."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for record, after, before in zip(records, treatment, baseline):
        buckets[str(record.tags.get(tag, "unknown"))].append(after[score] - before[score])
    return {
        name: (len(deltas), float(np.mean(deltas)))
        for name, deltas in sorted(buckets.items(), key=lambda item: -len(item[1]))
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puresound.evaluation.tools.noreference",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--inventory", required=True, help="JSONL from the corpus tools.")
    parser.add_argument("--recipe", default=None)
    parser.add_argument("--ckpt", default=None, help="Omit to score the baseline alone.")
    parser.add_argument("--precomputed", default=None, metavar="DIR",
                        help="Score audio another system already produced, one file per clip named by the clip's stem.")
    parser.add_argument("--precomputed-name", default=None)
    parser.add_argument("--task", default="noise_suppression")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-blend", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--score", default=DEFAULT_GATE_SCORE, choices=SCORES)
    parser.add_argument("--tag", action="append", default=None,
                        help="Tag to break results down by; repeatable.")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--stage-name", default="dns_devset_dnsmos")
    parser.add_argument("--role", default="monitor", choices=["gate", "monitor"])
    parser.add_argument(
        "--jobs", type=int, default=None,
        help="Worker processes. Defaults to half the cores; 1 runs inline.",
    )
    parser.add_argument("--out", default=None, help="Write the stage result here.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = list(read_inventory(args.inventory))
    if not records:
        print(f"{args.inventory} is empty.", file=sys.stderr)
        return 1

    baseline = Passthrough()
    if args.ckpt and args.precomputed:
        print("--ckpt and --precomputed are two systems; score them in two runs.", file=sys.stderr)
        return 2
    spec = {
        "recipe": args.recipe, "ckpt": args.ckpt, "task": args.task,
        "device": args.device, "dry_blend": args.dry_blend,
        "precomputed": args.precomputed,
        "precomputed_name": args.precomputed_name or (Path(args.precomputed).name if args.precomputed else None),
    }
    n_systems = 2 if (args.ckpt or args.precomputed) else 1

    print(f"scoring {args.limit or len(records)} clip(s) through {n_systems} system(s) "
          f"on {args.jobs or default_jobs()} worker(s)")
    results = run(records, spec, sample_rate=args.sample_rate, limit=args.limit, jobs=args.jobs)
    scored = records[: args.limit] if args.limit else records

    print(f"\n-- {baseline.name} --")
    for name in SCORES:
        values = [item[name] for item in results[baseline.name]]
        print(f"  {name:<14} mean {np.mean(values):.4f}  median {np.median(values):.4f}")

    if n_systems == 1:
        print("\nNo checkpoint given: this is the do-nothing reference every stage "
              "is read against, not a result.")
        return 0

    treatment = results[spec["precomputed_name"] or "model"]
    reference = results[baseline.name]
    print("\n-- model, paired against unprocessed --")
    stages: list[StageResult] = []
    for name in SCORES:
        after = [item[name] for item in treatment]
        before = [item[name] for item in reference]
        interval = paired_bootstrap_ci(after, before, aggregate=np.mean)
        decision = verdict(interval, direction="higher_is_better")
        role = args.role if name == args.score else "monitor"
        stage = StageResult.from_interval(
            f"{args.stage_name}.{name}",
            metric=name,
            role=role,
            value=float(np.mean(after)),
            baseline=float(np.mean(before)),
            interval=interval,
            verdict=decision,
            direction="higher_is_better",
            notes="no-reference; a passthrough scores well by not distorting anything",
        )
        stages.append(stage)
        print("  " + stage.line())

    for tag in args.tag or []:
        print(f"\n-- delta {args.score} by {tag} (worst first) --")
        table = by_tag(scored, treatment, reference, tag=tag, score=args.score)
        worst = sorted(table.items(), key=lambda item: item[1][1])
        for name, (count, delta) in worst[: args.top]:
            print(f"  {delta:+.4f}  n={count:<5} {name}")

    if args.out:
        write_stage(args.out, stages)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
