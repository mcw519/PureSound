import json

import numpy as np
import pytest

from puresound.audio.impedance_residues import (
    FixedPoleResidueCase,
    IMPEDANCE_MODAL_RESIDUE_SCHEMA_VERSION,
    LEGACY_IMPEDANCE_MODAL_RESIDUE_SCHEMA_VERSION,
    ImpedanceModalResidueCalibration,
    fit_fixed_pole_modal_residues,
)
from puresound.audio.rir_source_convention import (
    FDTD_PRESSURE_CELL_SOURCE_CONVENTION,
    FREE_FIELD_1_OVER_R_RIR_CONVENTION,
    PRESSURE_STATE_TO_FREE_FIELD_RESIDUE_TRANSFORM,
)


def _synthetic_case(case_id, split, gains):
    time_s = np.arange(600, dtype=np.float64) / 2000.0
    frequencies = np.asarray([70.0, 115.0, 170.0])
    real_basis = np.asarray(
        [
            gain
            * np.cos(2.0 * np.pi * frequency * time_s + 0.1 * index)
            * np.exp(-(5.0 + index) * time_s)
            for index, (frequency, gain) in enumerate(zip(frequencies, gains))
        ]
    )
    imaginary_basis = np.asarray(
        [
            gain
            * np.sin(2.0 * np.pi * frequency * time_s + 0.1 * index)
            * np.exp(-(5.0 + index) * time_s)
            for index, (frequency, gain) in enumerate(zip(frequencies, gains))
        ]
    )
    exponent = 0.4
    weights = (100.0 / frequencies) ** exponent
    beta = np.asarray([0.63, -0.17])
    target = np.column_stack(
        (weights @ real_basis, weights @ imaginary_basis)
    ) @ beta
    return FixedPoleResidueCase(
        case_id=case_id,
        split=split,
        mode_frequencies_hz=frequencies,
        real_mode_basis=real_basis,
        imaginary_mode_basis=imaginary_basis,
        target=target,
        analysis_mask=(time_s >= 0.03) & (time_s < 0.28),
        metadata={"synthetic": True},
    )


def test_fixed_pole_fit_recovers_complex_scale_and_transfers_to_holdout():
    cases = (
        _synthetic_case("train_a", "train", (1.0, 0.7, -0.4)),
        _synthetic_case("train_b", "train", (-0.3, 1.2, 0.8)),
        _synthetic_case("holdout", "holdout", (0.6, -0.8, 1.1)),
    )

    result = fit_fixed_pole_modal_residues(
        cases,
        reference_id="synthetic_residue",
        boundary_reference_id="synthetic_boundary",
        valid_frequency_range_hz=(60.0, 180.0),
        exponent_candidates=(-0.2, 0.0, 0.2, 0.4, 0.6, 0.8),
    )

    assert result.calibration.frequency_exponent == pytest.approx(0.4)
    assert result.calibration.complex_scale == pytest.approx(0.63 + 0.17j)
    holdout = result.report["aggregate"]["holdout"]["calibrated"]
    assert holdout["mean_nrmse"] < 1e-12
    assert holdout["mean_correlation"] == pytest.approx(1.0)
    assert result.report["acceptance"]["accepted"] is True


def test_fixed_pole_fit_reports_position_room_and_grid_holdouts():
    cases = (
        _synthetic_case("train_a", "train", (1.0, 0.7, -0.4)),
        _synthetic_case(
            "position",
            "position_holdout",
            (0.6, -0.8, 1.1),
        ),
        _synthetic_case("room", "room_holdout", (-0.2, 0.9, 0.5)),
        _synthetic_case("grid", "grid_holdout", (0.4, 0.3, -1.2)),
    )

    result = fit_fixed_pole_modal_residues(
        cases,
        reference_id="synthetic_residue",
        boundary_reference_id="synthetic_boundary",
        valid_frequency_range_hz=(60.0, 180.0),
        exponent_candidates=(0.2, 0.4, 0.6),
        acceptance_splits=(
            "position_holdout",
            "room_holdout",
            "grid_holdout",
        ),
    )

    assert result.report["acceptance"]["accepted"] is True
    assert set(result.report["acceptance"]["by_split"]) == {
        "position_holdout",
        "room_holdout",
        "grid_holdout",
    }
    assert result.report["aggregate"]["room_holdout"][
        "calibrated"
    ]["mean_nrmse"] < 1e-12


def test_residue_calibration_json_round_trip(tmp_path):
    calibration = ImpedanceModalResidueCalibration(
        reference_id="residue_test",
        boundary_reference_id="boundary_test",
        complex_scale=0.4 + 0.1j,
        frequency_exponent=0.3,
        reference_frequency_hz=100.0,
        valid_frequency_range_hz=(60.0, 240.0),
        source={"evidence_tier": "synthetic_validation"},
        applicability={
            "scope": "unit_test",
            "production_material_mapping_enabled": False,
        },
        fit_summary={"accepted": True},
    )
    path = tmp_path / "calibration.json"

    calibration.to_json(path)
    restored = ImpedanceModalResidueCalibration.from_json(path)

    assert restored == calibration
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == IMPEDANCE_MODAL_RESIDUE_SCHEMA_VERSION
    assert (
        metadata["fitted_source_convention"]
        == FDTD_PRESSURE_CELL_SOURCE_CONVENTION
    )
    assert (
        metadata["rendered_rir_convention"]
        == FREE_FIELD_1_OVER_R_RIR_CONVENTION
    )
    assert (
        metadata["residue_transform"]
        == PRESSURE_STATE_TO_FREE_FIELD_RESIDUE_TRANSFORM
    )
    assert restored.frequency_weight(200.0) == pytest.approx(0.5**0.3)


def test_residue_calibration_reads_legacy_v1_with_explicit_default_transform(
    tmp_path,
):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": LEGACY_IMPEDANCE_MODAL_RESIDUE_SCHEMA_VERSION,
                "reference_id": "legacy_residue",
                "boundary_reference_id": "legacy_boundary",
                "complex_scale": {"real": 0.4, "imag": 0.1},
                "frequency_exponent": 0.3,
                "reference_frequency_hz": 100.0,
                "valid_frequency_range_hz": [60.0, 240.0],
                "source": {"evidence_tier": "independent_numerical_reference"},
                "applicability": {"scope": "unit_test"},
                "fit_summary": {"accepted": True},
            }
        ),
        encoding="utf-8",
    )

    restored = ImpedanceModalResidueCalibration.from_json(path)

    assert (
        restored.fitted_source_convention
        == FDTD_PRESSURE_CELL_SOURCE_CONVENTION
    )
    assert restored.rendered_rir_convention == FREE_FIELD_1_OVER_R_RIR_CONVENTION
    assert (
        restored.residue_transform
        == PRESSURE_STATE_TO_FREE_FIELD_RESIDUE_TRANSFORM
    )


def test_residue_calibration_rejects_unknown_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": "unknown"}', encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported"):
        ImpedanceModalResidueCalibration.from_json(path)
