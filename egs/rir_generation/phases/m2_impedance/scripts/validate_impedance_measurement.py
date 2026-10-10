#!/usr/bin/env python
"""Fit and validate a normalized complex-impedance measurement.

The report uses alternating frequency bins for training and holdout, verifies
passivity on a dense frequency grid, and optionally compares a second
measurement of the nominally same sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.rir.physics.impedance.admittance import (
    FirstOrderRelaxationAdmittance,
)
from puresound.audio.rir.physics.impedance.fitting import (
    fit_normalized_complex_impedance_measurement,
)
from puresound.audio.rir.physics.impedance.measurements import (
    NormalizedComplexImpedanceMeasurement,
)
from puresound.audio.rir.physics.impedance.modes import solve_1d_impedance_cavity_modes


SCHEMA_VERSION = "puresound.impedance_measurement_validation.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--comparison-csv", type=Path)
    parser.add_argument("--comparison-metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acceptance-threshold", type=float, default=0.1)
    parser.add_argument("--dense-points", type=int, default=4096)
    parser.add_argument("--sound-speed", type=float, default=343.0)
    return parser.parse_args()


def _load(csv_path: Path, metadata_path: Path):
    return NormalizedComplexImpedanceMeasurement.from_csv_and_metadata(
        csv_path,
        metadata_path,
    )


def build_report(args: argparse.Namespace) -> dict:
    measurement = _load(args.csv, args.metadata)
    fit = fit_normalized_complex_impedance_measurement(
        measurement,
        acceptance_threshold=args.acceptance_threshold,
    )
    dense_frequencies = np.geomspace(
        measurement.frequencies_hz[0],
        measurement.frequencies_hz[-1],
        args.dense_points,
    )
    dense_admittance = np.asarray(
        [
            fit.model.normalized_admittance(float(frequency))
            for frequency in dense_frequencies
        ]
    )
    dense_reflection = np.asarray(
        [
            fit.model.reflection_coefficient(float(frequency))
            for frequency in dense_frequencies
        ]
    )

    cross_measurement = None
    if args.comparison_csv is not None or args.comparison_metadata is not None:
        if args.comparison_csv is None or args.comparison_metadata is None:
            raise ValueError(
                "comparison CSV and metadata must be provided together"
            )
        comparison = _load(
            args.comparison_csv,
            args.comparison_metadata,
        )
        if comparison.frequencies_hz != measurement.frequencies_hz:
            raise ValueError("comparison frequencies must match")
        difference = (
            measurement.cayley_reflection_coefficients()
            - comparison.cayley_reflection_coefficients()
        )
        cross_measurement = {
            "measurement_id": comparison.measurement_id,
            "rms_complex_cayley_difference": float(
                np.sqrt(np.mean(np.square(np.abs(difference))))
            ),
            "maximum_complex_cayley_difference": float(
                np.max(np.abs(difference))
            ),
        }

    resonance_hz = float(fit.model.resonance_frequencies_hz[0])
    cavity_length_m = float(args.sound_speed / (2.0 * resonance_hz))
    reference_reflection_magnitude = abs(
        fit.model.reflection_coefficient(resonance_hz)
    )
    magnitude_only_admittance = (
        (1.0 - reference_reflection_magnitude)
        / (1.0 + reference_reflection_magnitude)
    )
    magnitude_only = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=magnitude_only_admittance,
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=resonance_hz,
    )
    resonant_mode = solve_1d_impedance_cavity_modes(
        cavity_length_m,
        fit.model,
        sound_speed_m_s=args.sound_speed,
    )[0]
    magnitude_only_mode = solve_1d_impedance_cavity_modes(
        cavity_length_m,
        magnitude_only,
        sound_speed_m_s=args.sound_speed,
    )[0]

    return {
        "schema_version": SCHEMA_VERSION,
        "measurement": measurement.metadata(),
        "fit": fit.metadata(),
        "dense_passivity": {
            "num_frequencies": int(args.dense_points),
            "minimum_admittance_real": float(np.min(dense_admittance.real)),
            "maximum_reflection_magnitude": float(
                np.max(np.abs(dense_reflection))
            ),
            "passed": bool(
                np.min(dense_admittance.real) >= -1e-12
                and np.max(np.abs(dense_reflection)) <= 1.0 + 1e-12
            ),
        },
        "cross_measurement": cross_measurement,
        "one_dimensional_modal_diagnostic": {
            "local_reaction_assumption": True,
            "cavity_length_m": cavity_length_m,
            "reference_frequency_hz": resonance_hz,
            "reference_reflection_magnitude": (
                reference_reflection_magnitude
            ),
            "resonant_boundary": resonant_mode.metadata(),
            "magnitude_only_boundary": magnitude_only_mode.metadata(),
            "q_ratio": float(
                resonant_mode.q_factor / magnitude_only_mode.q_factor
            ),
        },
        "accepted": bool(
            fit.accepted
            and np.min(dense_admittance.real) >= -1e-12
            and np.max(np.abs(dense_reflection)) <= 1.0 + 1e-12
        ),
    }


def main() -> None:
    args = parse_args()
    if args.dense_points < 16:
        raise ValueError("dense-points must be at least 16")
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output} "
        f"(accepted={report['accepted']}, "
        f"measurement={report['measurement']['measurement_id']})"
    )


if __name__ == "__main__":
    main()
