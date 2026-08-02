import json

import pytest

from egs.rir_generation.phases.m4_spatial_late_field.scripts import validate_multiband_fdn


def _target_report():
    targets = {"500": 0.45, "1000": 0.38, "2000": 0.32}
    return {
        "milestone": "M4.2",
        "banks": {
            "measured": {
                "summary": {
                    "overall": {
                        "bands": {
                            key: {
                                "mixing_time_s": {
                                    "p10": 0.005,
                                    "median": 0.024,
                                    "p90": 0.06,
                                },
                                "late_median_normalized_density": {
                                    "p10": 0.8,
                                    "median": 1.0,
                                    "p90": 1.3,
                                },
                                "decay": {
                                    "t20": {
                                        "rt60_s": {
                                            "p10": 0.2,
                                            "median": target,
                                            "p90": 0.7,
                                        }
                                    }
                                },
                            }
                            for key, target in targets.items()
                        }
                    }
                }
            }
        },
    }


def test_m4_3_validator_passes_structural_and_high_band_gates():
    report, rir = validate_multiband_fdn.build_report(
        _target_report(),
        reference_tag="measured",
        sample_rate=8000,
        duration_s=1.2,
        delay_line_count=16,
        seed=19,
        qualified_centers_hz=(500.0, 1000.0, 2000.0),
        maximum_rt60_relative_error=0.12,
    )

    assert report["milestone"] == "M4.3"
    assert report["exit"]["structural_checks_passed"] is True
    assert report["exit"]["qualified_high_band_checks_passed"] is True
    assert report["exit"]["passed"] is True
    assert all(report["structural_checks"].values())
    assert rir.shape == (9600,)
    json.dumps(report, allow_nan=False)


def test_m4_3_validator_rejects_missing_qualified_target():
    with pytest.raises(ValueError, match="missing measured targets"):
        validate_multiband_fdn.build_report(
            _target_report(),
            reference_tag="measured",
            sample_rate=8000,
            duration_s=1.0,
            delay_line_count=16,
            seed=20,
            qualified_centers_hz=(4000.0,),
            maximum_rt60_relative_error=0.1,
        )
