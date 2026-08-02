import json

from egs.rir_generation.phases.m5_calibration.scripts.validate_m5_synthetic_recovery import build_report


def test_m5_2_synthetic_recovery_exit_keeps_measured_claim_open():
    report, artifacts = build_report(
        duration_s=0.256,
        maximum_evaluations=60,
    )

    assert report["exit"]["passed"] is True
    assert report["exit"]["synthetic_recovery_complete"] is True
    assert report["exit"]["measured_inverse_fit_complete"] is False
    assert report["multistart_max_parameter_spread"] <= 1e-6
    assert report["next_stage"]["blocked_until_controlled_campaign_ready"] is True
    assert set(artifacts) == {
        "holdout_target",
        "holdout_initial",
        "holdout_recovered",
    }
    json.dumps(report, allow_nan=False)
