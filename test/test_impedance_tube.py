import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from puresound.audio.rir.physics.impedance.admittance import PassiveMultiPoleAdmittance
from puresound.audio.rir.physics.impedance.measurements import (
    ComplexImpedanceMeasurement,
)
from puresound.audio.rir.physics.impedance.tube import (
    TwoMicrophoneTubeGeometry,
    microphone_switch_calibration_factor,
    reduce_two_microphone_repeats,
    reflection_from_two_microphone_transfer,
    transfer_from_surface_reflection,
)


def _fixture():
    frequencies = np.linspace(200.0, 1200.0, 21)
    geometry = TwoMicrophoneTubeGeometry(
        microphone_1_distance_from_sample_m=0.10,
        microphone_2_distance_from_sample_m=0.05,
        tube_diameter_m=0.10,
        sound_speed_m_s=343.0,
    )
    model = PassiveMultiPoleAdmittance(
        normalized_admittance_static=0.04,
        pole_frequencies_hz=(250.0, 800.0),
        normalized_admittance_lowpass=(0.02, 0.01),
        normalized_admittance_highpass=(0.25, 0.55),
    )
    reflection = np.asarray(
        [
            model.reflection_coefficient(float(frequency))
            for frequency in frequencies
        ]
    )
    transfer = transfer_from_surface_reflection(
        frequencies,
        reflection,
        geometry,
    )
    return frequencies, geometry, model, reflection, transfer


def test_two_microphone_round_trip_and_switch_calibration():
    frequencies, geometry, _, reflection, transfer = _fixture()
    mismatch = 1.08 * np.exp(0.07j)
    calibration_reflection = np.full(frequencies.shape, 0.25 - 0.1j)
    calibration_transfer = transfer_from_surface_reflection(
        frequencies,
        calibration_reflection,
        geometry,
    )
    correction = microphone_switch_calibration_factor(
        mismatch * calibration_transfer,
        mismatch / calibration_transfer,
    )

    recovered = reflection_from_two_microphone_transfer(
        frequencies,
        mismatch * transfer / correction,
        geometry,
    )

    np.testing.assert_allclose(correction, mismatch, atol=1e-12)
    np.testing.assert_allclose(recovered, reflection, atol=1e-12)


def test_reduction_recovers_impedance_and_repeatability():
    frequencies, geometry, model, _, transfer = _fixture()
    mismatch = 0.96 * np.exp(-0.05j)
    correction = np.full(frequencies.shape, mismatch, dtype=np.complex128)
    transfers = np.stack([mismatch * transfer] * 3)
    coherence = np.full(transfers.shape, 0.997)

    reduction = reduce_two_microphone_repeats(
        frequencies,
        transfers,
        coherence,
        geometry,
        air_density_kg_m3=1.204,
        microphone_correction=correction,
        minimum_coherence=0.98,
    )

    expected = np.asarray(
        [
            model.surface_impedance_pa_s_m(
                float(frequency),
                1.204,
                343.0,
            )
            for frequency in frequencies
        ]
    )
    np.testing.assert_allclose(
        reduction.complex_impedance_pa_s_m,
        expected,
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        reduction.impedance_real_std_pa_s_m,
        0.0,
        atol=1e-12,
    )
    assert reduction.num_repeats == 3
    assert reduction.metadata()["calibration_applied"] is True
    assert reduction.metadata()["maximum_reflection_magnitude"] <= 1.0


def test_reduction_rejects_bad_coherence_and_invalid_tube_band():
    frequencies, geometry, _, _, transfer = _fixture()
    coherence = np.full((1, frequencies.size), 0.99)
    coherence[0, 5] = 0.4
    with pytest.raises(ValueError, match="coherence gate failed"):
        reduce_two_microphone_repeats(
            frequencies,
            transfer[np.newaxis, :],
            coherence,
            geometry,
            air_density_kg_m3=1.204,
            minimum_coherence=0.95,
        )

    with pytest.raises(ValueError, match="transverse mode"):
        geometry.validate_frequencies((1800.0, 1950.0, 2050.0))
    with pytest.raises(ValueError, match="ill-conditioned"):
        geometry.validate_frequencies((10.0, 20.0, 30.0))


def test_impedance_tube_cli_writes_loadable_complex_measurement(tmp_path):
    frequencies, geometry, model, _, transfer = _fixture()
    mismatch = 1.04 * np.exp(0.04j)
    calibration_reflection = np.full(frequencies.shape, 0.2 - 0.15j)
    calibration_transfer = transfer_from_surface_reflection(
        frequencies,
        calibration_reflection,
        geometry,
    )
    transfer_csv = tmp_path / "raw.csv"
    calibration_csv = tmp_path / "calibration.csv"
    metadata_path = tmp_path / "raw.json"
    output_csv = tmp_path / "impedance.csv"
    output_metadata = tmp_path / "impedance.json"

    with transfer_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "repeat_id",
                "frequency_hz",
                "h12_real",
                "h12_imag",
                "coherence",
            ],
        )
        writer.writeheader()
        for repeat_id in ("run_1", "run_2"):
            for frequency, value in zip(frequencies, mismatch * transfer):
                writer.writerow(
                    {
                        "repeat_id": repeat_id,
                        "frequency_hz": frequency,
                        "h12_real": value.real,
                        "h12_imag": value.imag,
                        "coherence": 0.998,
                    }
                )
    with calibration_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frequency_hz",
                "h12_original_real",
                "h12_original_imag",
                "h12_swapped_real",
                "h12_swapped_imag",
            ],
        )
        writer.writeheader()
        for frequency, normal, swapped in zip(
            frequencies,
            mismatch * calibration_transfer,
            mismatch / calibration_transfer,
        ):
            writer.writerow(
                {
                    "frequency_hz": frequency,
                    "h12_original_real": normal.real,
                    "h12_original_imag": normal.imag,
                    "h12_swapped_real": swapped.real,
                    "h12_swapped_imag": swapped.imag,
                }
            )
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "puresound.impedance_tube_transfer_measurement.v1"
                ),
                "measurement_id": "synthetic_tube_roundtrip",
                "method": "ISO 10534-2 two-microphone transfer function",
                "phasor_convention": "exp(+i*omega*t)",
                "environment": {
                    "air_density_kg_m3": 1.204,
                    "sound_speed_m_s": 343.0,
                    "temperature_c": 20.0,
                    "relative_humidity_percent": 50.0,
                    "pressure_pa": 101325.0,
                },
                "tube": {
                    "shape": "circular",
                    "diameter_m": 0.10,
                    "microphone_1_distance_from_sample_m": 0.10,
                    "microphone_2_distance_from_sample_m": 0.05,
                },
                "acquisition": {
                    "analysis_frequency_range_hz": [200.0, 1200.0],
                    "minimum_coherence": 0.98,
                    "minimum_spacing_sine": 0.05,
                    "source_spl_db": 80.0,
                    "microphone_switch_calibration_required": True,
                },
                "sample": {
                    "sample_id": "synthetic_fixture",
                    "material_label": "synthetic passive fixture",
                    "thickness_m": 0.05,
                    "mounting": "flush, sealed perimeter",
                    "backing": "rigid",
                    "air_gap_m": 0.0,
                },
                "provenance": {
                    "source_url": "https://example.invalid/lab-run",
                    "license": "CC0-1.0",
                },
                "applicability": {
                    "scope": "synthetic_roundtrip_test_only",
                    "automatic_scene_catalog_mapping": False,
                },
            }
        ),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(
                repo_root
                / "egs"
                / "rir_generation"
                / "reduce_impedance_tube_measurement.py"
            ),
            "--transfer-csv",
            str(transfer_csv),
            "--metadata",
            str(metadata_path),
            "--microphone-switch-csv",
            str(calibration_csv),
            "--output-csv",
            str(output_csv),
            "--output-metadata",
            str(output_metadata),
        ],
        check=True,
        cwd=repo_root,
    )

    measurement = ComplexImpedanceMeasurement.from_csv_and_metadata(
        output_csv,
        output_metadata,
    )
    expected = np.asarray(
        [
            model.surface_impedance_pa_s_m(
                float(frequency),
                1.204,
                343.0,
            )
            for frequency in frequencies
        ]
    )
    np.testing.assert_allclose(
        measurement.complex_impedance_pa_s_m,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )
    output_sidecar = json.loads(output_metadata.read_text(encoding="utf-8"))
    assert output_sidecar["reduction"]["num_repeats"] == 2
    assert output_sidecar["reduction"]["calibration_applied"] is True
    assert output_sidecar["applicability"][
        "automatic_scene_catalog_mapping"
    ] is False
