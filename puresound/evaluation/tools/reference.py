"""Reference-based scoring on a frozen (mix, clean) set.

PESQ, STOI/ESTOI and SI-SDR against the clean target -- the layer whose numbers are
comparable to published ones, which is what a no-reference score can never be.

Everything is scored **paired against the unprocessed mixture**, because that is the
only reading that survives contact with a different test set. An absolute PESQ of
2.6 means nothing without knowing that doing nothing scores 2.1 here and 2.5 there.

Bands come from the manifest, so the per-SNR view is measured rather than assumed:
a system can gain a point of PESQ at 15 dB and lose speech at 0 dB, and the mean
over both reports neither.

Run::

    python -m puresound.evaluation.tools.reference --set-dir data_report/ns_testset \\
        --recipe config/infer_dpcrn.yaml --ckpt exp/.../epoch=39.ckpt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
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


#: ``(metric, direction)``. SI-SDR is reported as SI-SDRi -- the improvement over
#: the mixture -- because the absolute value is dominated by the item's own SNR.
#: The last two are the bottleneck measures from `evaluation.spectral`. They are
#: here because a model can pass every quality metric above while smearing
#: harmonics or transients, which none of those metrics reads.
METRICS = ("pesq_wb", "stoi", "estoi", "sisdr", "harmonic_gap_db", "transient_corr")
DEFAULT_GATE_METRIC = "pesq_wb"


def score_pair(
    enhanced: torch.Tensor, clean: torch.Tensor, sample_rate: int
) -> dict[str, float]:
    from puresound.metrics import Metrics

    # A model's output is not the same length as its input: the STFT/iSTFT round
    # trip drops a partial frame. Every metric here compares sample-for-sample, so
    # trim both to what they share -- otherwise SI-SDR raises and the windowed
    # metrics quietly score a shift.
    enhanced = enhanced.reshape(1, -1)
    clean = clean.reshape(1, -1)
    length = min(enhanced.shape[-1], clean.shape[-1])
    enhanced, clean = enhanced[..., :length], clean[..., :length]

    # `pesq_wb` is fixed at 16 kHz and takes no rate; the STOI pair does. The
    # `except` below catches only what a metric raises on bad audio, so a wrong
    # call signature fails loudly instead of becoming a NaN for every item.
    calls = (
        ("pesq_wb", Metrics.pesq_wb, ()),
        ("stoi", Metrics.stoi, (sample_rate,)),
        ("estoi", Metrics.estoi, (sample_rate,)),
    )
    scores: dict[str, float] = {}
    for name, func, extra in calls:
        try:
            value = func(clean, enhanced, *extra)
        except (ValueError, RuntimeError, ZeroDivisionError):
            # What a metric raises when it refuses THIS audio -- silence, a clip
            # too short for its window. A NaN here is a real gap in the data.
            value = float("nan")
        scores[name] = float(value)

    from puresound.nnet.loss.sdr import si_snr

    try:
        scores["sisdr"] = float(si_snr(enhanced, clean).reshape(-1).item())
    except Exception:
        scores["sisdr"] = float("nan")

    from puresound.evaluation.spectral import (
        harmonic_contrast_gap_db,
        transient_correlation,
    )

    scores["harmonic_gap_db"] = harmonic_contrast_gap_db(enhanced, clean, sample_rate)
    scores["transient_corr"] = transient_correlation(enhanced, clean, sample_rate)
    return scores


def read_manifest(set_dir: Path) -> list[dict[str, Any]]:
    path = set_dir / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Build the set with "
            "`python -m puresound.evaluation.tools.build_eval_set`."
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_provenance(set_dir: Path) -> dict[str, Any]:
    path = set_dir / "provenance.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Rebuild the frozen set so its synthesis commit, "
            "recipe digest and seed are recorded."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _build_systems(spec: dict[str, Any]) -> dict[str, Any]:
    """Construct the systems once per worker, not once per item."""
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
    return {"systems": systems, "set_dir": Path(spec["set_dir"])}


def _score_item(item: dict[str, Any], state: dict[str, Any]) -> dict[str, dict[str, float]]:
    set_dir = state["set_dir"]
    sample_rate = int(item["sample_rate"])
    mix, _ = AudioIO.open(str(set_dir / item["mix"]), resample_to=sample_rate)
    clean, _ = AudioIO.open(str(set_dir / item["clean"]), resample_to=sample_rate)
    mix, clean = mix.reshape(1, -1), clean.reshape(1, -1)
    key = Path(item["mix"]).stem
    return {
        system.name: score_pair(run_system(system, key, mix, sample_rate), clean, sample_rate)
        for system in state["systems"]
    }


def run(
    set_dir: Path,
    items: Sequence[dict[str, Any]],
    spec: dict[str, Any],
    *,
    jobs: int | None = None,
    progress_every: int = 50,
) -> dict[str, list[dict[str, float]]]:
    """Score every item through every system, in worker processes.

    Reading, inference and metrics all ride in the same worker: splitting them
    would ship a waveform between processes for every item, which costs more than
    the metric does.
    """
    spec = {**spec, "set_dir": str(set_dir)}
    per_item = map_items(
        _score_item, list(items), jobs=jobs,
        builder=lambda: _build_systems(spec),
        progress=lambda done, total: print(f"  {done}/{total}", flush=True),
        progress_every=progress_every,
    )
    names = list(per_item[0]) if per_item else []
    return {name: [row[name] for row in per_item] for name in names}


def _finite_pairs(after: Sequence[float], before: Sequence[float]) -> tuple[list[float], list[float]]:
    """Drop pairs where either side is NaN -- PESQ refuses some inputs, and a
    metric that silently becomes NaN for a few items would otherwise poison a mean."""
    kept = [
        (left, right)
        for left, right in zip(after, before)
        if np.isfinite(left) and np.isfinite(right)
    ]
    return [pair[0] for pair in kept], [pair[1] for pair in kept]


def by_band(
    items: Sequence[dict[str, Any]],
    treatment: Sequence[dict[str, float]],
    baseline: Sequence[dict[str, float]],
    *,
    key: str,
    metric: str,
) -> dict[str, tuple[int, float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for item, after, before in zip(items, treatment, baseline):
        delta = after[metric] - before[metric]
        if np.isfinite(delta):
            buckets[str(item.get(key, "unknown"))].append(delta)
    return {
        name: (len(values), float(np.mean(values)))
        for name, values in sorted(buckets.items())
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puresound.evaluation.tools.reference",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--set-dir", required=True)
    parser.add_argument("--recipe", default=None)
    parser.add_argument("--ckpt", default=None, help="Omit to score the baseline alone.")
    parser.add_argument(
        "--precomputed", default=None, metavar="DIR",
        help="Score audio another system already produced: one file per item, named "
        "by the input file's stem. Puts a third-party model on the same axis as ours.",
    )
    parser.add_argument("--precomputed-name", default=None, help="System name in the record; defaults to the directory name.")
    parser.add_argument("--task", default="noise_suppression")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-blend", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metric", default=DEFAULT_GATE_METRIC, choices=METRICS)
    parser.add_argument("--band", action="append", default=["snr_band"])
    parser.add_argument("--stage-name", default="frozen_testset")
    parser.add_argument("--role", default="gate", choices=["gate", "monitor"])
    parser.add_argument(
        "--jobs", type=int, default=None,
        help="Worker processes. Defaults to half the cores, leaving room for a "
        "training run; 1 runs inline.",
    )
    parser.add_argument("--out", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_dir = Path(args.set_dir).expanduser().resolve()
    items = read_manifest(set_dir)
    provenance = read_provenance(set_dir)
    if args.limit:
        items = items[: args.limit]
    if not items:
        print(f"{set_dir} has no items.", file=sys.stderr)
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

    print(f"scoring {len(items)} item(s) through {n_systems} system(s) "
          f"on {args.jobs or default_jobs()} worker(s)")
    results = run(set_dir, items, spec, jobs=args.jobs)

    print(f"\n-- {baseline.name} --")
    for name in METRICS:
        values = [item[name] for item in results[baseline.name]]
        finite = [value for value in values if np.isfinite(value)]
        print(f"  {name:<8} mean {np.mean(finite):.4f}  median {np.median(finite):.4f}"
              f"  (n={len(finite)}/{len(values)})")

    if n_systems == 1:
        print("\nNo checkpoint given: this is the do-nothing reference, not a result.")
        return 0

    # The treatment is whichever second system the spec built: a checkpoint is
    # always called "model"; a precomputed directory carries its own name.
    treatment_name = spec["precomputed_name"] or "model"
    treatment, reference = results[treatment_name], results[baseline.name]
    print(f"\n-- {treatment_name}, paired against the unprocessed mixture --")
    stages: list[StageResult] = []
    for name in METRICS:
        after, before = _finite_pairs(
            [item[name] for item in treatment], [item[name] for item in reference]
        )
        if not after:
            print(f"  {name}: no finite pairs, skipped")
            continue
        interval = paired_bootstrap_ci(after, before, aggregate=np.mean)
        decision = verdict(interval, direction="higher_is_better")
        stage = StageResult.from_interval(
            f"{args.stage_name}.{name}",
            metric=name,
            role=args.role if name == args.metric else "monitor",
            value=float(np.mean(after)),
            baseline=float(np.mean(before)),
            interval=interval,
            verdict=decision,
            direction="higher_is_better",
            notes=f"frozen set {set_dir.name}",
            set_provenance=provenance,
        )
        stages.append(stage)
        print("  " + stage.line())

    for key in args.band:
        print(f"\n-- delta {args.metric} by {key} --")
        for name, (count, delta) in by_band(
            items, treatment, reference, key=key, metric=args.metric
        ).items():
            print(f"  {delta:+.4f}  n={count:<5} {name}")

    if args.out:
        write_stage(args.out, stages)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
