import csv
import json
from pathlib import Path

import numpy as np
import pytest

from puresound.audio.rir.physics.impedance.admittance import (
    PassiveMultiPoleAdmittance,
    characteristic_impedance_pa_s_m,
)
from puresound.audio.rir.physics.impedance.measurements import (
    COMPLEX_IMPEDANCE_MEASUREMENT_SCHEMA_VERSION,
    ComplexImpedanceMeasurement,
    NormalizedComplexImpedanceMeasurement,
)
from puresound.audio.rir.physics.impedance.fitting import (
    fit_complex_impedance_measurement,
    fit_normalized_complex_impedance_measurement,
)


def _write_measurement(tmp_path, model, *, include_uncertainty=True):
    frequencies = (60.0, 80.0, 120.0, 180.0, 250.0, 300.0)
    air_density = 1.204
    sound_speed = 343.0
    csv_path = tmp_path / "measurement.csv"
    metadata_path = tmp_path / "measurement.json"
    fieldnames = [
        "frequency_hz",
        "impedance_real_pa_s_m",
        "impedance_imag_pa_s_m",
    ]
    if include_uncertainty:
        fieldnames += [
            "impedance_real_std_pa_s_m",
            "impedance_imag_std_pa_s_m",
        ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frequency in frequencies:
            impedance = model.surface_impedance_pa_s_m(
                frequency,
                air_density,
                sound_speed,
            )
            row = {
                "frequency_hz": frequency,
                "impedance_real_pa_s_m": impedance.real,
                "impedance_imag_pa_s_m": impedance.imag,
            }
            if include_uncertainty:
                row.update(
                    {
                        "impedance_real_std_pa_s_m": 2.0,
                        "impedance_imag_std_pa_s_m": 3.0,
                    }
                )
            writer.writerow(row)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": COMPLEX_IMPEDANCE_MEASUREMENT_SCHEMA_VERSION,
                "measurement_id": "synthetic_fixture",
                "method": "ISO 10534-2 two-microphone transfer function",
                "incidence": "normal",
                "environment": {
                    "air_density_kg_m3": air_density,
                    "sound_speed_m_s": sound_speed,
                },
                "sample": {
                    "material_label": "synthetic passive fixture",
                    "thickness_m": 0.1,
                    "backing": "rigid",
                },
                "provenance": {
                    "source_url": "https://example.invalid/synthetic-fixture",
                    "license": "CC0-1.0",
                },
            }
        ),
        encoding="utf-8",
    )
    return csv_path, metadata_path


def test_complex_measurement_loads_phase_uncertainty_and_provenance(tmp_path):
    model = PassiveMultiPoleAdmittance(
        normalized_admittance_static=0.01,
        pole_frequencies_hz=(80.0, 240.0),
        normalized_admittance_lowpass=(0.02, 0.01),
        normalized_admittance_highpass=(0.3, 0.6),
    )
    csv_path, metadata_path = _write_measurement(tmp_path, model)

    measurement = ComplexImpedanceMeasurement.from_csv_and_metadata(
        csv_path,
        metadata_path,
    )

    expected = np.asarray(
        [
            model.reflection_coefficient(frequency)
            for frequency in measurement.frequencies_hz
        ]
    )
    np.testing.assert_allclose(measurement.reflection_coefficients(), expected)
    assert measurement.metadata()["has_uncertainty"] is True
    assert measurement.metadata()["maximum_reflection_magnitude"] <= 1.0
    json.dumps(measurement.metadata(), allow_nan=False)


def test_complex_measurement_rejects_active_impedance():
    z0 = characteristic_impedance_pa_s_m(1.204, 343.0)

    with pytest.raises(ValueError, match="negative real"):
        ComplexImpedanceMeasurement(
            measurement_id="active",
            method="test",
            incidence="normal",
            frequencies_hz=(100.0, 200.0, 300.0),
            impedance_real_pa_s_m=(-z0, z0, z0),
            impedance_imag_pa_s_m=(0.0, 0.0, 0.0),
            air_density_kg_m3=1.204,
            sound_speed_m_s=343.0,
            sample={"material_label": "active fixture"},
            provenance={
                "source_url": "https://example.invalid",
                "license": "CC0-1.0",
            },
        )


def test_complex_measurement_rejects_non_normal_or_missing_provenance():
    kwargs = {
        "measurement_id": "fixture",
        "method": "test",
        "frequencies_hz": (100.0, 200.0, 300.0),
        "impedance_real_pa_s_m": (400.0, 400.0, 400.0),
        "impedance_imag_pa_s_m": (0.0, 0.0, 0.0),
        "air_density_kg_m3": 1.204,
        "sound_speed_m_s": 343.0,
        "sample": {"material_label": "fixture"},
        "provenance": {
            "source_url": "https://example.invalid",
            "license": "CC0-1.0",
        },
    }
    with pytest.raises(ValueError, match="normal-incidence"):
        ComplexImpedanceMeasurement(incidence="diffuse", **kwargs)
    with pytest.raises(ValueError, match="source_url"):
        ComplexImpedanceMeasurement(
            incidence="normal",
            **{**kwargs, "provenance": {"license": "CC0-1.0"}},
        )


def test_passive_multi_pole_fit_recovers_synthetic_complex_measurement(tmp_path):
    expected = PassiveMultiPoleAdmittance(
        normalized_admittance_static=0.01,
        pole_frequencies_hz=(80.0, 240.0),
        normalized_admittance_lowpass=(0.02, 0.01),
        normalized_admittance_highpass=(0.3, 0.6),
    )
    csv_path, metadata_path = _write_measurement(tmp_path, expected)
    measurement = ComplexImpedanceMeasurement.from_csv_and_metadata(
        csv_path,
        metadata_path,
    )

    fit = fit_complex_impedance_measurement(
        measurement,
        num_poles=2,
        pole_frequencies_hz=expected.pole_frequencies_hz,
        acceptance_threshold=1e-5,
    )

    assert fit.accepted
    assert fit.maximum_complex_reflection_error < 1e-7
    assert fit.source_measurement_id == measurement.measurement_id
    assert fit.weighting_strategy == (
        "inverse_propagated_complex_reflection_standard_deviation"
    )
    for frequency_hz in np.linspace(60.0, 300.0, 101):
        assert fit.model.reflection_coefficient(
            frequency_hz
        ) == pytest.approx(
            expected.reflection_coefficient(frequency_hz),
            abs=1e-6,
        )
        assert abs(fit.model.reflection_coefficient(frequency_hz)) <= 1.0
    json.dumps(fit.metadata(), allow_nan=False)


def test_direct_normalized_liner_measurements_fit_held_out_frequencies():
    measurement_dir = (
        Path(__file__).resolve().parents[1]
        / "egs"
        / "rir_generation"
        / "measurements"
        / "zenodo_15195587"
    )
    nasa = NormalizedComplexImpedanceMeasurement.from_csv_and_metadata(
        measurement_dir / "nasa_gfit_noflow_130db_kt.csv",
        measurement_dir / "nasa_gfit_noflow_130db_kt.json",
    )
    ufsc = NormalizedComplexImpedanceMeasurement.from_csv_and_metadata(
        measurement_dir / "ufsc_noflow_130db_kt.csv",
        measurement_dir / "ufsc_noflow_130db_kt.json",
    )

    assert nasa.acoustic_field_geometry == "grazing_duct"
    assert nasa.mean_flow_mach == 0.0
    assert nasa.provenance["source_sha256"] == (
        "ba4cf7cf293d2b20ed590eb78ed8c133484771acbd011c680466d289abdc1a72"
    )
    assert nasa.applicability["automatic_scene_catalog_mapping"] is False
    assert nasa.frequencies_hz == ufsc.frequencies_hz

    fit = fit_normalized_complex_impedance_measurement(
        nasa,
        acceptance_threshold=0.1,
    )
    assert fit.accepted
    assert fit.held_out_metrics is not None
    assert fit.training_metrics.maximum_complex_reflection_error < 0.082
    assert fit.held_out_metrics.maximum_complex_reflection_error < 0.060
    assert fit.model.resonance_frequencies_hz[0] == pytest.approx(
        1646.7,
        abs=1.0,
    )

    cross_rig_error = (
        nasa.cayley_reflection_coefficients()
        - ufsc.cayley_reflection_coefficients()
    )
    cross_rig_rms = float(
        np.sqrt(np.mean(np.square(np.abs(cross_rig_error))))
    )
    assert fit.held_out_metrics.rms_complex_reflection_error < cross_rig_rms
    for frequency_hz in np.linspace(500.0, 2500.0, 1001):
        assert fit.model.normalized_admittance(frequency_hz).real >= 0.0
        assert abs(fit.model.reflection_coefficient(frequency_hz)) <= 1.0
    json.dumps(fit.metadata(), allow_nan=False)
