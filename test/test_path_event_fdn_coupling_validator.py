import json

from egs.rir_generation import generate_hybrid_rir as generate_hybrid_rir_cli
from egs.rir_generation.phases.m4_spatial_late_field.scripts import (
    validate_path_event_fdn_coupling,
)
from puresound.audio.rir.render.high_frequency import (
    PathEventFDNHighFrequencyBackend,
    PyroomacousticsHighFrequencyBackend,
)


def _target_report():
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
                                    "p90": 0.1,
                                },
                                "late_median_normalized_density": {
                                    "p10": 0.75,
                                    "median": 1.0,
                                    "p90": 1.25,
                                },
                            }
                            for key in ("500", "1000", "2000")
                        }
                    }
                }
            }
        },
    }


def test_m4_4_validator_passes_coupling_and_full_hybrid_fixture():
    report, artifacts = validate_path_event_fdn_coupling.build_report(
        _target_report(),
        sample_rate=8000,
        duration_s=0.8,
        max_order=2,
        scene_seed=31,
        fdn_seed=32,
        qualified_centers_hz=(500.0, 1000.0, 2000.0),
        maximum_rt60_relative_error=0.25,
        maximum_late_density_absolute_error=0.25,
    )

    assert report["milestone"] == "M4.4"
    assert report["exit"]["structural_checks_passed"] is True
    assert report["exit"]["qualified_acoustic_checks_passed"] is True
    assert report["exit"]["full_hybrid_compatibility_checks_passed"] is True
    assert report["exit"]["passed"] is True
    assert all(report["structural_checks"].values())
    assert all(report["full_hybrid"]["checks"].values())
    assert set(artifacts) == {
        "m3_coherent_high",
        "m4_coupled_high",
        "m3_full_hybrid",
        "m4_full_hybrid",
    }
    json.dumps(report, allow_nan=False)


def test_generate_cli_builds_explicit_m4_backend_and_keeps_default_backend():
    m4 = generate_hybrid_rir_cli._make_high_backend(
        {
            "high_backend": "path-events-m4",
            "pra_max_order": 4,
            "pra_n_rays": 100,
            "fdn_mixing_time_ms": 24.0,
            "fdn_transition_ms": 16.0,
            "fdn_delay_lines": 16,
            "fdn_seed": 33,
        }
    )
    default = generate_hybrid_rir_cli._make_high_backend(
        {
            "high_backend": "pyroomacoustics",
            "pra_max_order": 4,
            "pra_n_rays": 100,
        }
    )

    assert isinstance(m4, PathEventFDNHighFrequencyBackend)
    assert m4.mixing_time_s == 0.024
    assert m4.transition_duration_s == 0.016
    assert isinstance(default, PyroomacousticsHighFrequencyBackend)
