import json

from egs.rir_generation.phases.m5_calibration.scripts.validate_m5_m4_parameter_mapping import build_report


def test_m5_2c_actual_m4_mapping_passes_without_claiming_measured_fit():
    report, artifacts = build_report(maximum_evaluations=50)

    assert report["milestone"] == "M5.2c"
    assert report["exit"]["passed"] is True
    assert report["exit"]["m5_2c_actual_m4_parameter_mapping_complete"] is True
    assert report["exit"]["measured_inverse_fit_complete"] is False
    assert report["checks"]["true_mixing_profile_selected"] is True
    assert report["checks"]["production_default_remains_pyroomacoustics"] is True
    assert report["independent_m5_1_oracle"]["relative_total_reduction"] >= 0.5
    assert set(artifacts) == {
        "holdout_target",
        "holdout_initial",
        "holdout_recovered",
    }
    json.dumps(report, allow_nan=False)
