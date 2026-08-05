#!/usr/bin/env python
"""Is "synthetic RIRs have no noise floor" a valid target, or another chain artifact?

The spectral-tilt gap turned out to be a mixture of measurement chains rather
than room physics (``compare_measured_tilt_by_corpus.py``), so the remaining
synthetic/measured difference deserves the same scrutiny before anyone acts on
it.

Two questions, and they have different answers a priori:

*Existence.* Every real measurement has a noise floor — microphone self-noise,
preamp noise, ambient sound during the sweep. Every synthetic RIR has none.
That should hold in every corpus regardless of chain, and it is the part worth
modelling.

*Level.* Where the floor sits is set by the chain's SNR, so it should vary
between corpora by construction. If it does, "match the measured dynamic range"
is as invalid a target as matching the pooled tilt was, even though "add a
floor at all" stays valid.

The same variance decomposition separates them: spread within a corpus is room
and position; spread between corpora is chain.

Exit code 0 if existence is universal (every measured corpus has floors, no
synthetic bank does), 1 otherwise.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.rir.metrics import estimate_noise_floor_lundeby

DEFAULT_MEASURED = Path(
    "/work/any_exp_link/puresound_exp/real_rir_16k_train_view/items"
)

#: Between-corpus spread above this many dB means the floor *level* is a chain
#: property and cannot serve as a matching target.
LEVEL_POOLING_TOLERANCE_DB = 4.0

MINIMUM_ROOMS = 3


def _late_decay_flatness(signal: np.ndarray, sample_rate: int) -> float | None:
    """Late Schroeder slope over early Schroeder slope.

    1.0 means the decay keeps its rate to the end; 0 means it has flattened
    onto a floor. Reported alongside the Lundeby detection rate because that
    detector has a 15 dB minimum-range threshold, so a corpus that trimmed its
    tails before publication registers as "no floor" for a reason that has
    nothing to do with whether the measurement had noise. This ratio needs no
    threshold.
    """
    x = np.asarray(signal, dtype=np.float64)
    peak = int(np.argmax(np.abs(x)))
    segment = x[peak:]
    if segment.size < sample_rate // 4:
        return None
    energy = np.cumsum(segment[::-1] ** 2)[::-1]
    if energy[0] <= 0.0:
        return None
    curve = 10.0 * np.log10(np.maximum(energy / energy[0], 1e-30))
    seconds = np.arange(curve.size) / sample_rate
    early = (curve <= -5.0) & (curve >= -25.0)
    if early.sum() < sample_rate // 50:
        return None
    early_slope = float(np.polyfit(seconds[early], curve[early], 1)[0])
    if early_slope >= -1.0:
        return None
    count = curve.size
    window = slice(int(0.70 * count), int(0.95 * count))
    if window.stop - window.start < sample_rate // 50:
        return None
    late_slope = float(np.polyfit(seconds[window], curve[window], 1)[0])
    return late_slope / early_slope


def _measure(
    signal: np.ndarray, sample_rate: int
) -> tuple[bool, float | None, float | None]:
    """Return floor detected, dynamic range above it, and late-decay flatness.

    ``direct_index`` is left unset on purpose so the estimator keys on the
    signal peak for measured and synthetic alike; anything else would compare
    two different definitions.
    """
    estimate = estimate_noise_floor_lundeby(signal, sample_rate)
    return (
        bool(estimate.correction_applied),
        estimate.dynamic_range_db,
        _late_decay_flatness(signal, sample_rate),
    )


def collect_measured(root: Path, per_corpus: int, sample_rate: int, seed: int):
    by_corpus: dict[str, list[Path]] = collections.defaultdict(list)
    for meta in glob.glob(str(root / "*.json")):
        by_corpus[Path(meta).name.split("_")[0]].append(Path(meta))

    rng = random.Random(seed)
    rows = []
    for corpus, paths in sorted(by_corpus.items()):
        rng.shuffle(paths)
        used = 0
        for meta_path in paths:
            if used >= per_corpus:
                break
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                audio, file_rate = sf.read(
                    meta_path.with_suffix(".wav"), always_2d=True
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if file_rate != sample_rate:
                continue
            room = meta_path.name.rsplit("_", 1)[0]
            for entry in metadata.get("scene", {}).get("channel_map", []):
                index = int(entry["channel"])
                if index >= audio.shape[1]:
                    continue
                detected, dynamic, flatness = _measure(
                    audio[:, index].astype(np.float64), sample_rate
                )
                rows.append((corpus, room, detected, dynamic, flatness))
            used += 1
    return rows


def collect_synthetic(bank_root: Path, per_bank: int, sample_rate: int, seed: int):
    rng = random.Random(seed)
    paths = sorted(glob.glob(str(bank_root / "*" / "*.wav")))
    rng.shuffle(paths)
    rows = []
    for wav_path in paths[:per_bank]:
        try:
            audio, file_rate = sf.read(wav_path, always_2d=True)
        except (OSError, ValueError):
            continue
        if file_rate != sample_rate:
            continue
        room = Path(wav_path).parent.name
        for index in range(audio.shape[1]):
            detected, dynamic, flatness = _measure(
                audio[:, index].astype(np.float64), sample_rate
            )
            rows.append((bank_root.name, room, detected, dynamic, flatness))
    return rows


def summarize(rows, label: str) -> dict:
    groups = sorted({group for group, _, _, _, _ in rows})
    print(f"\n{label}")
    print(f"  {'group':<26} {'rooms':>6} {'ch':>6} {'floor found':>12} "
          f"{'dyn dB':>8} {'room sd':>8} {'flatness':>9}")
    summary = {}
    for group in groups:
        subset = [r for r in rows if r[0] == group]
        detected = [r[2] for r in subset]
        per_room: dict[str, list[float]] = collections.defaultdict(list)
        for _group, room, found, dynamic, _flat in subset:
            if found and dynamic is not None and np.isfinite(dynamic):
                per_room[room].append(float(dynamic))
        room_medians = [float(np.median(v)) for v in per_room.values() if v]
        finite = [
            float(d) for _g, _r, f, d, _fl in subset
            if f and d is not None and np.isfinite(d)
        ]
        flat = [
            float(fl) for _g, _r, _f, _d, fl in subset
            if fl is not None and np.isfinite(fl)
        ]
        sd = (
            float(np.std(room_medians, ddof=1)) if len(room_medians) > 1
            else float("nan")
        )
        summary[group] = {
            "rooms": len({r[1] for r in subset}),
            "channels": len(subset),
            "floor_found_fraction": float(np.mean(detected)),
            "median_dynamic_range_db": float(np.median(finite)) if finite else None,
            "room_sd_db": sd,
            "rooms_with_floor": len(room_medians),
            "median_late_decay_flatness": (
                float(np.median(flat)) if flat else None
            ),
        }
        median = summary[group]["median_dynamic_range_db"]
        flatness = summary[group]["median_late_decay_flatness"]
        print(f"  {group:<26} {summary[group]['rooms']:>6} {len(subset):>6} "
              f"{100 * np.mean(detected):>11.1f}% "
              f"{'n/a' if median is None else f'{median:>8.1f}'} "
              f"{sd:>8.2f} "
              f"{'n/a' if flatness is None else f'{flatness:>9.2f}'}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured", type=Path, default=DEFAULT_MEASURED)
    parser.add_argument(
        "--synthetic", type=Path, nargs="*", default=(),
        help="synthetic bank roots (<root>/<room>/<item>.wav)",
    )
    parser.add_argument("--per-corpus", type=int, default=180)
    parser.add_argument("--per-bank", type=int, default=120)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()

    measured = collect_measured(
        args.measured, args.per_corpus, args.sample_rate, args.seed
    )
    if not measured:
        print(f"no usable measured RIRs under {args.measured}", file=sys.stderr)
        return 1
    measured_summary = summarize(measured, "measured corpora")

    synthetic_summary = {}
    for bank in args.synthetic:
        rows = collect_synthetic(bank, args.per_bank, args.sample_rate, args.seed)
        if rows:
            synthetic_summary.update(summarize(rows, f"synthetic: {bank.name}"))

    eligible = [
        name for name, row in measured_summary.items()
        if row["rooms_with_floor"] >= MINIMUM_ROOMS
        and np.isfinite(row["room_sd_db"])
    ]
    within = float(np.mean([measured_summary[n]["room_sd_db"] for n in eligible]))
    levels = [measured_summary[n]["median_dynamic_range_db"] for n in eligible]
    between = float(np.std(levels, ddof=1)) if len(levels) > 1 else 0.0

    print("\n--- existence ---")
    every_measured_has_floors = all(
        row["floor_found_fraction"] > 0.5 for row in measured_summary.values()
    )
    no_synthetic_has_floors = all(
        row["floor_found_fraction"] < 0.05 for row in synthetic_summary.values()
    )
    print(f"  every measured corpus has floors : {every_measured_has_floors}")
    print(f"  no synthetic bank has floors     : {no_synthetic_has_floors}"
          f"{'' if synthetic_summary else '  (no synthetic bank supplied)'}")

    print("\n--- level ---")
    print(f"  within-corpus room sd  {within:.2f} dB")
    print(f"  between-corpus sd      {between:.2f} dB  ({between / within:.2f}x)")
    print(f"  corpus median range    {min(levels):.1f} to {max(levels):.1f} dB "
          f"({max(levels) - min(levels):.1f} dB apart)")
    level_poolable = between <= LEVEL_POOLING_TOLERANCE_DB
    print(f"  poolable as a target   {level_poolable}")

    existence_holds = every_measured_has_floors and (
        no_synthetic_has_floors or not synthetic_summary
    )
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(
            json.dumps(
                {
                    "schema_version": "puresound.noise_floor_by_corpus.v1",
                    "measured": measured_summary,
                    "synthetic": synthetic_summary,
                    "existence_is_universal": existence_holds,
                    "within_corpus_room_sd_db": within,
                    "between_corpus_sd_db": between,
                    "level_pooling_tolerance_db": LEVEL_POOLING_TOLERANCE_DB,
                    "level_is_poolable": level_poolable,
                },
                indent=2, sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"\nreport: {args.output_report}")

    print()
    if existence_holds:
        print("EXISTENCE HOLDS — a noise floor is present in every measured "
              "corpus and absent from synthetic output, so *having* one is a "
              "real and chain-independent difference worth modelling.")
    else:
        print("EXISTENCE DOES NOT HOLD — the presence of a floor is not "
              "universal, so it cannot be treated as a synthetic/measured "
              "distinction.")
    if not level_poolable:
        print("LEVEL IS NOT A TARGET — the floor's depth varies between "
              "corpora by more than it varies between rooms, so it is a "
              "property of each measurement chain. Model a floor; do not tune "
              "to the pooled dynamic range.")
    return 0 if existence_holds else 1


if __name__ == "__main__":
    raise SystemExit(main())
