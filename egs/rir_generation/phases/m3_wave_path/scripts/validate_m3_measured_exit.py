#!/usr/bin/env python3
"""Run the M3 early-timing and C50 measured-bank exit probe."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

from egs.rir_generation.compare_bank_acoustics import (
    bucket_label,
    discover_items,
)
from puresound.audio.hybrid_rir import (
    AnalyticModalLowFrequencyBackend,
    HybridRIRConfig,
    PathEventHighFrequencyBackend,
    generate_hybrid_rir,
)
from puresound.audio.rir_metrics import clarity_db, direct_sample
from puresound.audio.rir_scene import RoomSceneV2


REPORT_SCHEMA_VERSION = "puresound.m3_measured_exit.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare paired M1/M3 synthetic early timing and C50 with a "
            "deterministic held-out measured-RIR sample."
        )
    )
    parser.add_argument(
        "--m1-bank",
        type=Path,
        default=Path("egs/rir_generation/exp/rir_realism/m1/rir_m1_probe100"),
    )
    parser.add_argument(
        "--measured-bank",
        type=Path,
        default=Path(
            "/work/any_exp_link/puresound_exp/"
            "real_rir_16k_heldout_view/items"
        ),
    )
    parser.add_argument("--synthetic-items", type=int, default=20)
    parser.add_argument("--measured-channels", type=int, default=100)
    parser.add_argument("--path-event-max-order", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def _dominant_early_reflection_delay_ms(
    rir: np.ndarray,
    sample_rate_hz: int,
    *,
    exclusion_ms: float = 2.5,
    maximum_delay_ms: float = 50.0,
) -> float | None:
    values = np.asarray(rir, dtype=np.float64)
    if values.ndim != 1 or values.size < 4 or not np.any(values != 0.0):
        return None
    direct = direct_sample(values)
    magnitude = np.abs(values)
    peak = float(magnitude[direct])
    if peak <= 0.0:
        return None
    start = direct + max(
        1,
        round(exclusion_ms * 1e-3 * sample_rate_hz),
    )
    stop = min(
        values.size,
        direct + round(maximum_delay_ms * 1e-3 * sample_rate_hz),
    )
    if stop <= start:
        return None
    tail = magnitude[max(start, values.size * 3 // 4) :]
    noise_floor = (
        float(np.median(tail)) if tail.size else 0.0
    )
    threshold = max(0.04 * peak, 8.0 * noise_floor)
    peaks, properties = find_peaks(
        magnitude[start:stop],
        height=threshold,
        distance=max(1, round(0.2e-3 * sample_rate_hz)),
    )
    if peaks.size == 0 or properties["peak_heights"].size == 0:
        return None
    dominant_peak = int(
        peaks[int(np.argmax(properties["peak_heights"]))]
    )
    return float(
        (start + dominant_peak - direct)
        / sample_rate_hz
        * 1000.0
    )


def _early_reflection_energy_centroid_ms(
    rir: np.ndarray,
    sample_rate_hz: int,
    *,
    exclusion_ms: float = 2.5,
    maximum_delay_ms: float = 50.0,
) -> float | None:
    """Return the continuous energy center of the direct-excluded early field.

    Unlike a single peak pick, this statistic does not jump discontinuously
    when two coherent reflections exchange which local maximum is taller.
    The fixed direct exclusion matches the project's DRR convention and the
    50 ms endpoint matches C50.
    """
    values = np.asarray(rir, dtype=np.float64)
    if values.ndim != 1 or values.size < 4 or not np.any(values != 0.0):
        return None
    direct = direct_sample(values)
    start = direct + max(
        1,
        round(exclusion_ms * 1e-3 * sample_rate_hz),
    )
    stop = min(
        values.size,
        direct + round(maximum_delay_ms * 1e-3 * sample_rate_hz),
    )
    if stop <= start:
        return None
    early_energy = np.square(values[start:stop])
    total = float(np.sum(early_energy))
    if not math.isfinite(total) or total <= 0.0:
        return None
    delay_ms = (
        np.arange(start, stop, dtype=np.float64)
        - float(direct)
    ) / float(sample_rate_hz) * 1000.0
    return float(np.sum(delay_ms * early_energy) / total)


def _row(
    rir: np.ndarray,
    *,
    sample_rate_hz: int,
    distance_m: float,
) -> dict[str, Any]:
    return {
        "bucket": bucket_label(float(distance_m)),
        "distance_m": float(distance_m),
        "c50_db": float(clarity_db(rir, sample_rate_hz, 50.0)),
        "dominant_early_reflection_delay_ms": (
            _dominant_early_reflection_delay_ms(
                rir,
                sample_rate_hz,
            )
        ),
        "early_reflection_energy_centroid_ms": (
            _early_reflection_energy_centroid_ms(
                rir,
                sample_rate_hz,
            )
        ),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["bucket"]].append(row)
    return {
        bucket: {
            "channels": len(values),
            "c50_db_median": float(
                np.median([value["c50_db"] for value in values])
            ),
            "dominant_early_reflection_delay_ms_median": (
                float(np.median(timing))
                if (
                    timing := [
                        value["dominant_early_reflection_delay_ms"]
                        for value in values
                        if value[
                            "dominant_early_reflection_delay_ms"
                        ]
                        is not None
                    ]
                )
                else None
            ),
            "early_reflection_energy_centroid_ms_median": (
                float(np.median(centroids))
                if (
                    centroids := [
                        value["early_reflection_energy_centroid_ms"]
                        for value in values
                        if value[
                            "early_reflection_energy_centroid_ms"
                        ]
                        is not None
                    ]
                )
                else None
            ),
            "timing_qualified_channels": len(timing),
            "centroid_qualified_channels": len(centroids),
        }
        for bucket, values in sorted(buckets.items())
    }


def _load_paired_synthetic(
    bank: Path,
    *,
    item_count: int,
    max_order: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = discover_items(bank)[:item_count]
    if len(pairs) < item_count:
        raise ValueError("M1 bank contains fewer items than requested")
    m1_rows = []
    m3_rows = []
    high_backend = PathEventHighFrequencyBackend(max_order=max_order)
    low_backend = AnalyticModalLowFrequencyBackend()
    for metadata_path, wav_path in pairs:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scene = RoomSceneV2.from_dict(metadata["scene"])
        config_values = {
            key: value
            for key, value in metadata["config"].items()
            if key in HybridRIRConfig.__dataclass_fields__
        }
        config = HybridRIRConfig(**config_values)
        m1, sample_rate = sf.read(
            wav_path,
            dtype="float32",
            always_2d=True,
        )
        # The frozen M1 WAV is a complete hybrid RIR.  Compare it with the
        # complete M3 hybrid candidate, not with an isolated high-band impulse
        # response: otherwise C50 mostly measures the deliberately absent
        # low-frequency modal tail rather than the M3 path model.
        m3_tensor, _m3_metadata = generate_hybrid_rir(
            config=config,
            scene=scene,
            low_backend=low_backend,
            high_backend=high_backend,
        )
        m3 = m3_tensor.detach().cpu().numpy()
        for channel, distance in enumerate(scene.source_distances()):
            m1_rows.append(
                _row(
                    m1[:, channel],
                    sample_rate_hz=int(sample_rate),
                    distance_m=distance,
                )
            )
            m3_rows.append(
                _row(
                    m3[channel],
                    sample_rate_hz=config.sample_rate,
                    distance_m=distance,
                )
            )
    return m1_rows, m3_rows


def _load_measured(
    bank: Path,
    *,
    channel_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    pairs = discover_items(bank)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    rows = []
    for metadata_path, wav_path in pairs:
        if len(rows) >= channel_count:
            break
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        channel_map = metadata["scene"]["channel_map"]
        wav, sample_rate = sf.read(
            wav_path,
            dtype="float32",
            always_2d=True,
        )
        valid = [
            item
            for item in channel_map
            if item.get("distance_m") is not None
            and 0 <= int(item["channel"]) < wav.shape[1]
        ]
        if not valid:
            continue
        selected = rng.choice(valid)
        rows.append(
            _row(
                wav[:, int(selected["channel"])],
                sample_rate_hz=int(sample_rate),
                distance_m=float(selected["distance_m"]),
            )
        )
    if len(rows) < channel_count:
        raise ValueError("measured bank contains too few valid channels")
    return rows


def _aggregate_gap(
    candidate: dict[str, Any],
    measured: dict[str, Any],
    metric: str,
) -> tuple[float, list[str]]:
    common = [
        bucket
        for bucket in candidate
        if (
            bucket in measured
            and candidate[bucket][metric] is not None
            and measured[bucket][metric] is not None
            and candidate[bucket]["channels"] >= 3
            and measured[bucket]["channels"] >= 3
        )
    ]
    if not common:
        raise ValueError(f"no qualified common buckets for {metric}")
    return (
        float(
            np.mean(
                [
                    abs(
                        candidate[bucket][metric]
                        - measured[bucket][metric]
                    )
                    for bucket in common
                ]
            )
        ),
        common,
    )


def main() -> None:
    args = _parse_args()
    m1_rows, m3_rows = _load_paired_synthetic(
        args.m1_bank,
        item_count=int(args.synthetic_items),
        max_order=int(args.path_event_max_order),
    )
    measured_rows = _load_measured(
        args.measured_bank,
        channel_count=int(args.measured_channels),
        seed=int(args.seed),
    )
    summaries = {
        "m1_paired": _summary(m1_rows),
        "m3_paired": _summary(m3_rows),
        "measured_heldout": _summary(measured_rows),
    }
    m1_c50_gap, c50_buckets = _aggregate_gap(
        summaries["m1_paired"],
        summaries["measured_heldout"],
        "c50_db_median",
    )
    m3_c50_gap, _ = _aggregate_gap(
        summaries["m3_paired"],
        summaries["measured_heldout"],
        "c50_db_median",
    )
    timing_metric = "early_reflection_energy_centroid_ms_median"
    m1_timing_gap, timing_buckets = _aggregate_gap(
        summaries["m1_paired"],
        summaries["measured_heldout"],
        timing_metric,
    )
    m3_timing_gap, _ = _aggregate_gap(
        summaries["m3_paired"],
        summaries["measured_heldout"],
        timing_metric,
    )
    c50_improved = m3_c50_gap < m1_c50_gap
    timing_improved = m3_timing_gap < m1_timing_gap
    accepted = bool(c50_improved and timing_improved)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output_report"
        },
        "comparison_contract": {
            "m1_candidate": (
                "frozen complete M1 hybrid RIR from paired bank"
            ),
            "m3_candidate": (
                "complete hybrid RIR with the same analytic modal low "
                "backend and the opt-in M3 PathEvent high backend"
            ),
            "paired_scene_geometry_and_configuration": True,
            "measured_timing_metric_role": (
                "distributional direct-excluded 2.5-50 ms early-energy "
                "centroid; not annotated reflection-path correspondence"
            ),
        },
        "summaries": summaries,
        "aggregate_gap": {
            "c50_db": {
                "qualified_buckets": c50_buckets,
                "m1_mean_absolute_median_gap": m1_c50_gap,
                "m3_mean_absolute_median_gap": m3_c50_gap,
                "relative_improvement": (
                    (m1_c50_gap - m3_c50_gap) / m1_c50_gap
                ),
            },
            "early_reflection_energy_centroid_ms": {
                "qualified_buckets": timing_buckets,
                "m1_mean_absolute_median_gap": m1_timing_gap,
                "m3_mean_absolute_median_gap": m3_timing_gap,
                "relative_improvement": (
                    (m1_timing_gap - m3_timing_gap) / m1_timing_gap
                ),
            },
            "dominant_early_reflection_delay_ms": {
                "role": (
                    "diagnostic_only_due_to_discontinuous_peak_identity"
                ),
                "m1_mean_absolute_median_gap": _aggregate_gap(
                    summaries["m1_paired"],
                    summaries["measured_heldout"],
                    "dominant_early_reflection_delay_ms_median",
                )[0],
                "m3_mean_absolute_median_gap": _aggregate_gap(
                    summaries["m3_paired"],
                    summaries["measured_heldout"],
                    "dominant_early_reflection_delay_ms_median",
                )[0],
            },
        },
        "acceptance": {
            "paired_synthetic_geometry": True,
            "heldout_measured_bank_used": True,
            "c50_gap_improved": c50_improved,
            "early_timing_gap_improved": timing_improved,
            "m3_measured_exit_accepted": accepted,
        },
        "decision": (
            "accept_m3_measured_early_timing_and_c50_exit"
            if accepted
            else "retain_m3_measured_exit_blocker"
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(args.output_report),
                "aggregate_gap": report["aggregate_gap"],
                "acceptance": report["acceptance"],
            },
            indent=2,
        )
    )
    if not accepted:
        raise RuntimeError("M3 measured exit gate did not improve both metrics")


if __name__ == "__main__":
    main()
