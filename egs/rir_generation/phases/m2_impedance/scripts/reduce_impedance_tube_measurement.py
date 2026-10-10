#!/usr/bin/env python
"""Reduce repeated two-microphone H12 sweeps to complex surface impedance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.rir.physics.impedance.measurements import (
    COMPLEX_IMPEDANCE_MEASUREMENT_SCHEMA_VERSION,
    ComplexImpedanceMeasurement,
)
from puresound.audio.rir.physics.impedance.tube import (
    IMPEDANCE_TUBE_TRANSFER_MEASUREMENT_SCHEMA_VERSION,
    TwoMicrophoneTubeGeometry,
    load_microphone_switch_csv,
    load_transfer_repeats_csv,
    reduce_two_microphone_repeats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-csv", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--microphone-switch-csv", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--minimum-frequency", type=float)
    parser.add_argument("--maximum-frequency", type=float)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _required_mapping(
    metadata: dict,
    key: str,
) -> dict:
    value = metadata.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"metadata.{key} must be an object")
    return value


def _required_finite(
    mapping: dict,
    key: str,
    *,
    positive: bool = False,
) -> float:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"metadata field {key!r} is required") from error
    if not math.isfinite(value) or (positive and value <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"metadata field {key!r} must be {qualifier}")
    return value


def _validate_metadata(metadata: dict) -> None:
    if (
        metadata.get("schema_version")
        != IMPEDANCE_TUBE_TRANSFER_MEASUREMENT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported impedance-tube transfer schema")
    if not metadata.get("measurement_id") or not metadata.get("method"):
        raise ValueError("measurement_id and method are required")
    if metadata.get("phasor_convention") != "exp(+i*omega*t)":
        raise ValueError("phasor_convention must be exp(+i*omega*t)")

    environment = _required_mapping(metadata, "environment")
    _required_finite(environment, "air_density_kg_m3", positive=True)
    _required_finite(environment, "sound_speed_m_s", positive=True)
    _required_finite(environment, "temperature_c")
    humidity = _required_finite(environment, "relative_humidity_percent")
    if not 0.0 <= humidity <= 100.0:
        raise ValueError("relative humidity must be in [0, 100]")
    _required_finite(environment, "pressure_pa", positive=True)

    tube = _required_mapping(metadata, "tube")
    if tube.get("shape") != "circular":
        raise ValueError("only circular impedance tubes are currently supported")
    for key in (
        "diameter_m",
        "microphone_1_distance_from_sample_m",
        "microphone_2_distance_from_sample_m",
    ):
        _required_finite(tube, key, positive=True)

    acquisition = _required_mapping(metadata, "acquisition")
    frequency_range = acquisition.get("analysis_frequency_range_hz")
    if (
        not isinstance(frequency_range, list)
        or len(frequency_range) != 2
        or not all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in frequency_range
        )
        or float(frequency_range[1]) <= float(frequency_range[0])
    ):
        raise ValueError(
            "acquisition.analysis_frequency_range_hz must be [min, max]"
        )
    _required_finite(acquisition, "minimum_coherence")
    _required_finite(acquisition, "source_spl_db", positive=True)

    sample = _required_mapping(metadata, "sample")
    for key in ("material_label", "sample_id", "mounting", "backing"):
        if not sample.get(key):
            raise ValueError(f"sample.{key} is required")
    _required_finite(sample, "thickness_m", positive=True)
    _required_finite(sample, "air_gap_m")

    provenance = _required_mapping(metadata, "provenance")
    for key in ("source_url", "license"):
        if not provenance.get(key):
            raise ValueError(f"provenance.{key} is required")


def build_outputs(args: argparse.Namespace) -> tuple[dict, object]:
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    _validate_metadata(metadata)
    frequencies, transfers, coherence, repeat_ids = load_transfer_repeats_csv(
        args.transfer_csv
    )

    acquisition = metadata["acquisition"]
    configured_range = acquisition["analysis_frequency_range_hz"]
    minimum_frequency = (
        float(args.minimum_frequency)
        if args.minimum_frequency is not None
        else float(configured_range[0])
    )
    maximum_frequency = (
        float(args.maximum_frequency)
        if args.maximum_frequency is not None
        else float(configured_range[1])
    )
    if (
        not math.isfinite(minimum_frequency)
        or not math.isfinite(maximum_frequency)
        or minimum_frequency <= 0.0
        or maximum_frequency <= minimum_frequency
    ):
        raise ValueError("selected frequency range must be finite, positive, and ordered")
    mask = (frequencies >= minimum_frequency) & (
        frequencies <= maximum_frequency
    )
    if int(np.sum(mask)) < 3:
        raise ValueError("selected frequency range contains fewer than three bins")

    correction = None
    calibration_hash = None
    if args.microphone_switch_csv is not None:
        calibration_frequencies, full_correction = load_microphone_switch_csv(
            args.microphone_switch_csv
        )
        if not np.array_equal(calibration_frequencies, frequencies):
            raise ValueError(
                "microphone-switch calibration and sample frequency grids must match"
            )
        correction = full_correction[mask]
        calibration_hash = _sha256(args.microphone_switch_csv)
    elif bool(acquisition.get("microphone_switch_calibration_required", True)):
        raise ValueError(
            "metadata requires a microphone-switch calibration CSV"
        )

    environment = metadata["environment"]
    tube = metadata["tube"]
    geometry = TwoMicrophoneTubeGeometry(
        microphone_1_distance_from_sample_m=float(
            tube["microphone_1_distance_from_sample_m"]
        ),
        microphone_2_distance_from_sample_m=float(
            tube["microphone_2_distance_from_sample_m"]
        ),
        tube_diameter_m=float(tube["diameter_m"]),
        sound_speed_m_s=float(environment["sound_speed_m_s"]),
        minimum_spacing_sine=float(
            acquisition.get("minimum_spacing_sine", 0.05)
        ),
    )
    reduction = reduce_two_microphone_repeats(
        frequencies[mask],
        transfers[:, mask],
        coherence[:, mask],
        geometry,
        air_density_kg_m3=float(environment["air_density_kg_m3"]),
        microphone_correction=correction,
        minimum_coherence=float(acquisition["minimum_coherence"]),
        passivity_tolerance=float(
            acquisition.get("passivity_tolerance", 1e-6)
        ),
    )

    provenance = dict(metadata["provenance"])
    transformations = list(provenance.get("transformations", []))
    transformations.append(
        "reduced repeated calibrated H12=P(x2)/P(x1) spectra with "
        "two-microphone plane-wave decomposition"
    )
    provenance.update(
        {
            "raw_transfer_csv": str(args.transfer_csv),
            "raw_transfer_sha256": _sha256(args.transfer_csv),
            "microphone_switch_csv": (
                str(args.microphone_switch_csv)
                if args.microphone_switch_csv is not None
                else None
            ),
            "microphone_switch_sha256": calibration_hash,
            "transformations": transformations,
        }
    )
    output_metadata = {
        "schema_version": COMPLEX_IMPEDANCE_MEASUREMENT_SCHEMA_VERSION,
        "measurement_id": str(metadata["measurement_id"]),
        "method": str(metadata["method"]),
        "incidence": "normal",
        "phasor_convention": "exp(+i*omega*t)",
        "environment": dict(environment),
        "sample": dict(metadata["sample"]),
        "tube": reduction.geometry.metadata(),
        "acquisition": {
            **dict(acquisition),
            "repeat_ids": list(repeat_ids),
        },
        "reduction": reduction.metadata(),
        "provenance": provenance,
        "applicability": dict(
            metadata.get(
                "applicability",
                {
                    "scope": "measurement_validation_pending",
                    "automatic_scene_catalog_mapping": False,
                },
            )
        ),
    }
    measurement = ComplexImpedanceMeasurement(
        measurement_id=output_metadata["measurement_id"],
        method=output_metadata["method"],
        incidence="normal",
        frequencies_hz=reduction.frequencies_hz,
        impedance_real_pa_s_m=reduction.impedance_real_pa_s_m,
        impedance_imag_pa_s_m=reduction.impedance_imag_pa_s_m,
        impedance_real_std_pa_s_m=(
            reduction.impedance_real_std_pa_s_m
            if reduction.num_repeats >= 2
            else ()
        ),
        impedance_imag_std_pa_s_m=(
            reduction.impedance_imag_std_pa_s_m
            if reduction.num_repeats >= 2
            else ()
        ),
        air_density_kg_m3=float(environment["air_density_kg_m3"]),
        sound_speed_m_s=float(environment["sound_speed_m_s"]),
        sample=output_metadata["sample"],
        provenance=output_metadata["provenance"],
    )
    return output_metadata, measurement


def write_outputs(
    output_csv: Path,
    output_metadata_path: Path,
    metadata: dict,
    measurement: ComplexImpedanceMeasurement,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    has_uncertainty = bool(measurement.impedance_real_std_pa_s_m)
    fieldnames = [
        "frequency_hz",
        "impedance_real_pa_s_m",
        "impedance_imag_pa_s_m",
    ]
    if has_uncertainty:
        fieldnames.extend(
            [
                "impedance_real_std_pa_s_m",
                "impedance_imag_std_pa_s_m",
            ]
        )
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, frequency in enumerate(measurement.frequencies_hz):
            row = {
                "frequency_hz": frequency,
                "impedance_real_pa_s_m": (
                    measurement.impedance_real_pa_s_m[index]
                ),
                "impedance_imag_pa_s_m": (
                    measurement.impedance_imag_pa_s_m[index]
                ),
            }
            if has_uncertainty:
                row.update(
                    {
                        "impedance_real_std_pa_s_m": (
                            measurement.impedance_real_std_pa_s_m[index]
                        ),
                        "impedance_imag_std_pa_s_m": (
                            measurement.impedance_imag_std_pa_s_m[index]
                        ),
                    }
                )
            writer.writerow(row)
    output_metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    metadata, measurement = build_outputs(args)
    write_outputs(
        args.output_csv,
        args.output_metadata,
        metadata,
        measurement,
    )
    print(
        f"wrote {args.output_csv} and {args.output_metadata} "
        f"(measurement={measurement.measurement_id}, "
        f"bins={len(measurement.frequencies_hz)}, "
        f"repeats={metadata['reduction']['num_repeats']})"
    )


if __name__ == "__main__":
    main()
