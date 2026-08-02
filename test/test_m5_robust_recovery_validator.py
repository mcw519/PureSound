import json

from egs.rir_generation.phases.m5_calibration.scripts.validate_m5_robust_recovery import build_report


def test_m5_2b_robust_exit_keeps_measured_fit_open():
    report, artifacts = build_report(maximum_evaluations=80)

    assert report["exit"]["passed"] is True
    assert report["exit"]["m5_2b_robust_synthetic_recovery_complete"] is True
    assert report["exit"]["measured_inverse_fit_complete"] is False
    assert report["next_stage"]["blocked_until_controlled_campaign_ready"] is True
    assert report["measurement_audit"]["train"][
        "all_known_nuisance_corrections_restore_arrival"
    ]
    assert report["independent_m5_1_oracle"]["robust_holdout_relative_reduction"] >= 0.5
    assert set(artifacts) == {
        "holdout_clean",
        "holdout_raw_perturbed",
        "holdout_corrected_target",
        "holdout_initial",
        "holdout_waveform_ablation",
        "holdout_robust_recovered",
    }
    json.dumps(report, allow_nan=False)
