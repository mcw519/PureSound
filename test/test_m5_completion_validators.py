import json
from pathlib import Path

import pytest

from egs.rir_generation.phases.m5_calibration.scripts import (
    validate_m5_constrained_residual,
    validate_m5_exit,
    validate_m5_group_identifiability,
    validate_m5_measured_runner,
    validate_m5_spatial_calibration,
)

# End-to-end evidence-chain validator: builds, QCs and releases a real bank, so it
# runs for tens of seconds. Excluded by `run_repo_checks.py --suite standard`.
pytestmark = pytest.mark.slow


def test_m5_2d_group_identifiability_passes_and_rejects_scattering_split():
    report, artifacts = validate_m5_group_identifiability.build_report(
        maximum_evaluations=40
    )

    assert report["exit"]["passed"] is True
    assert (
        report["exit"]["individual_absorption_scattering_separation_claimed"] is False
    )
    assert len(report["selection_decision"]["accepted_for_m5_3"]) == 6
    assert len(report["selection_decision"]["deferred_to_synchronized_m5_4"]) == 6
    assert set(artifacts) == {
        "holdout_target",
        "holdout_initial",
        "holdout_recovered",
    }
    json.dumps(report, allow_nan=False)


def test_m5_3_runner_executes_complete_non_evidence_fixture(tmp_path):
    report = validate_m5_measured_runner.build_report(
        tmp_path,
        records_per_room=4,
        sample_count=2400,
        max_order=2,
        maximum_evaluations=40,
    )

    assert report["exit"]["passed"] is True
    assert report["exit"]["m5_3_runner_implementation_complete"] is True
    assert report["exit"]["measured_room_empirical_fit_complete"] is False
    json.dumps(report, allow_nan=False)


def test_m5_4_spatial_and_m5_5_residual_implementation_gates_pass():
    spatial, spatial_artifacts = validate_m5_spatial_calibration.build_report()
    residual, residual_artifacts = validate_m5_constrained_residual.build_report()

    assert spatial["exit"]["passed"] is True
    assert spatial["exit"]["measured_spatial_calibration_complete"] is False
    assert residual["exit"]["passed"] is True
    assert residual["exit"]["measured_residual_training_complete"] is False
    assert set(spatial_artifacts) == {
        "target_synchronized_pair",
        "selected_candidate",
        "wrong_zero_scattering",
        "wrong_omni_directivity",
    }
    assert set(residual_artifacts) == {
        "heldout_target",
        "heldout_physical_only",
        "heldout_residual_only",
        "heldout_combined",
        "interpolated_combined",
    }


def test_m5_6_exit_marks_code_complete_but_empirical_open():
    reports = {
        stage: json.loads(path.read_text(encoding="utf-8"))
        for stage, path in validate_m5_exit.DEFAULT_REPORTS.items()
    }
    report = validate_m5_exit.build_report(reports)

    assert report["implementation_exit"]["passed"] is True
    assert report["implementation_exit"]["m5_1_through_m5_6_code_complete"] is True
    assert report["empirical_exit"]["passed"] is False
    assert report["production_enablement"]["ready"] is False
    assert Path(validate_m5_exit.DEFAULT_OUTPUT_REPORT).name == "m5_exit_report.json"
    json.dumps(report, allow_nan=False)
