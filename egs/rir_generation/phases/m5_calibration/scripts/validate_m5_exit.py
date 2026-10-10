#!/usr/bin/env python3
"""Aggregate M5 implementation separately from empirical/production exit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[5]
CONFIG_DIR = REPO_ROOT / "egs/rir_generation/phases/m5_calibration/reports"
DEFAULT_REPORTS = {
    "M5.1": CONFIG_DIR / "m5_measurement_contract_report.json",
    "M5.2": CONFIG_DIR / "m5_synthetic_recovery_report.json",
    "M5.2b": CONFIG_DIR / "m5_robust_recovery_report.json",
    "M5.2c": CONFIG_DIR / "m5_m4_parameter_mapping_report.json",
    "M5.2d": CONFIG_DIR / "m5_group_identifiability_report.json",
    "M5.3i": CONFIG_DIR / "m5_measured_runner_validation_report.json",
    "M5.3": CONFIG_DIR / "m5_measured_fit_status_report.json",
    "M5.4": CONFIG_DIR / "m5_spatial_calibration_report.json",
    "M5.5": CONFIG_DIR / "m5_constrained_residual_report.json",
}
DEFAULT_OUTPUT_REPORT = CONFIG_DIR / "m5_exit_report.json"
DEFAULT_EMPIRICAL_REPORTS = {
    "controlled_listening": CONFIG_DIR / "m5_controlled_listening_report.json",
    "downstream": CONFIG_DIR / "m5_downstream_report.json",
}
REQUIRED_ARTIFACTS = {
    "m5_2c_recovered": (
        REPO_ROOT / "egs/rir_generation/exp/rir_realism/m5/rir_m5_m4_parameter_mapping/m5_2c_holdout_recovered.wav"
    ),
    "m5_2d_recovered": (
        REPO_ROOT / "egs/rir_generation/exp/rir_realism/m5/rir_m5_group_identifiability/m5_2d_holdout_recovered.wav"
    ),
    "m5_3_runner_fixture_campaign": (
        REPO_ROOT / "egs/rir_generation/exp/rir_realism/m5/rir_m5_measured_runner_fixture/campaign.json"
    ),
    "m5_4_selected_array": (
        REPO_ROOT / "egs/rir_generation/exp/rir_realism/m5/rir_m5_spatial_calibration/m5_4_selected_candidate.wav"
    ),
    "m5_5_physical": (
        REPO_ROOT / "egs/rir_generation/exp/rir_realism/m5/rir_m5_constrained_residual/m5_5_heldout_physical_only.wav"
    ),
    "m5_5_residual": (
        REPO_ROOT / "egs/rir_generation/exp/rir_realism/m5/rir_m5_constrained_residual/m5_5_heldout_residual_only.wav"
    ),
    "m5_5_combined": (
        REPO_ROOT / "egs/rir_generation/exp/rir_realism/m5/rir_m5_constrained_residual/m5_5_heldout_combined.wav"
    ),
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_report(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    report_paths: Mapping[str, Path] | None = None,
    artifact_paths: Mapping[str, Path] | None = None,
    empirical_reports: Mapping[str, Mapping[str, Any] | None] | None = None,
    empirical_report_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Return honest implementation, empirical, and production states."""

    paths = dict(report_paths or DEFAULT_REPORTS)
    artifacts = dict(artifact_paths or REQUIRED_ARTIFACTS)
    empirical_paths = dict(empirical_report_paths or DEFAULT_EMPIRICAL_REPORTS)
    empirical = dict(empirical_reports or {})
    expected_milestones = {
        **{stage: stage for stage in DEFAULT_REPORTS},
        "M5.3i": "M5.3-implementation",
    }
    stage_identity = {
        stage: reports.get(stage, {}).get("milestone") == expected_milestones[stage]
        for stage in DEFAULT_REPORTS
    }
    m5_1 = reports.get("M5.1", {})
    m5_2 = reports.get("M5.2", {})
    m5_2b = reports.get("M5.2b", {})
    m5_2c = reports.get("M5.2c", {})
    m5_2d = reports.get("M5.2d", {})
    m5_3i = reports.get("M5.3i", {})
    m5_3 = reports.get("M5.3", {})
    m5_4 = reports.get("M5.4", {})
    m5_5 = reports.get("M5.5", {})
    stage_gates = {
        "m5_1_measurement_loss_contract_passed": bool(
            stage_identity["M5.1"]
            and m5_1.get("implementation_exit", {}).get("passed", False)
        ),
        "m5_2_noise_free_recovery_passed": bool(
            stage_identity["M5.2"] and m5_2.get("exit", {}).get("passed", False)
        ),
        "m5_2b_robust_recovery_passed": bool(
            stage_identity["M5.2b"] and m5_2b.get("exit", {}).get("passed", False)
        ),
        "m5_2c_actual_m4_profile_passed": bool(
            stage_identity["M5.2c"] and m5_2c.get("exit", {}).get("passed", False)
        ),
        "m5_2d_group_identifiability_passed": bool(
            stage_identity["M5.2d"] and m5_2d.get("exit", {}).get("passed", False)
        ),
        "m5_3_runner_executes_complete_campaign_fixture": bool(
            stage_identity["M5.3i"]
            and m5_3i.get("exit", {}).get("passed", False)
            and m5_3i.get("exit", {}).get("m5_3_runner_implementation_complete", False)
            and m5_3i.get("exit", {}).get("measured_room_empirical_fit_complete")
            is False
        ),
        "m5_3_status_is_fail_closed_or_executed": bool(
            stage_identity["M5.3"]
            and (
                (
                    m5_3.get("exit", {}).get("blocked", False)
                    and not m5_3.get("exit", {}).get(
                        "m5_3_measured_fit_executed", True
                    )
                    and m5_3.get("checks", {}).get(
                        "fitting_not_started_with_incomplete_evidence", False
                    )
                )
                or (
                    m5_3.get("exit", {}).get("passed", False)
                    and m5_3.get("exit", {}).get(
                        "m5_3_measured_fit_executed", False
                    )
                )
            )
        ),
        "m5_4_spatial_profile_implementation_passed": bool(
            stage_identity["M5.4"] and m5_4.get("exit", {}).get("passed", False)
        ),
        "m5_5_constrained_residual_implementation_passed": bool(
            stage_identity["M5.5"] and m5_5.get("exit", {}).get("passed", False)
        ),
    }
    artifact_checks = {name: path.is_file() for name, path in artifacts.items()}
    invariant_checks = {
        "mono_absorption_scattering_ambiguity_not_hidden": bool(
            m5_2d.get("exit", {}).get(
                "individual_absorption_scattering_separation_claimed"
            )
            is False
        ),
        "measured_fit_claim_matches_campaign_readiness": bool(
            m5_3.get("exit", {}).get("m5_3_measured_fit_executed", False)
            == m5_3.get("campaign_audit", {}).get(
                "ready_for_m5_inverse_calibration", False
            )
        ),
        "measured_spatial_fit_not_falsely_claimed": bool(
            m5_4.get("exit", {}).get("measured_spatial_calibration_complete") is False
        ),
        "measured_residual_training_not_falsely_claimed": bool(
            m5_5.get("exit", {}).get("measured_residual_training_complete") is False
        ),
        "production_default_unchanged": bool(
            m5_2c.get("model_mapping", {}).get("production_default_changed") is False
        ),
    }
    implementation_passed = bool(
        all(stage_gates.values())
        and all(artifact_checks.values())
        and all(invariant_checks.values())
    )

    listening = empirical.get("controlled_listening")
    downstream = empirical.get("downstream")
    measured_executed = bool(
        m5_3.get("exit", {}).get("m5_3_measured_fit_executed", False)
    )
    empirical_checks = {
        "controlled_campaign_ready": bool(
            m5_1.get("controlled_measurement_exit", {}).get("passed", False)
        ),
        "measured_room_fit_executed": measured_executed,
        "heldout_positions_improved": bool(
            m5_3.get("exit", {}).get("heldout_positions_improved", False)
        ),
        "heldout_physical_rooms_improved": bool(
            m5_3.get("exit", {}).get("heldout_physical_rooms_improved", False)
        ),
        "measured_synchronized_spatial_calibration_complete": bool(
            m5_4.get("exit", {}).get("measured_spatial_calibration_complete", False)
        ),
        "measured_residual_training_complete": bool(
            m5_5.get("exit", {}).get("measured_residual_training_complete", False)
        ),
        "controlled_listening_passed": bool(
            listening and listening.get("exit", {}).get("passed", False)
        ),
        "room_disjoint_downstream_passed": bool(
            downstream and downstream.get("exit", {}).get("passed", False)
        ),
    }
    empirical_passed = bool(all(empirical_checks.values()))
    missing_empirical = [
        name for name, passed in empirical_checks.items() if not passed
    ]
    return {
        "schema_version": "puresound.m5_exit.v1",
        "milestone": "M5.6",
        "scope": "implementation_exit_with_separate_empirical_status",
        "input_reports": {stage: _display(paths[stage]) for stage in DEFAULT_REPORTS},
        "stage_identity": stage_identity,
        "stage_gates": stage_gates,
        "artifact_checks": artifact_checks,
        "artifact_paths": {name: _display(path) for name, path in artifacts.items()},
        "invariant_checks": invariant_checks,
        "implementation_exit": {
            "passed": implementation_passed,
            "m5_1_through_m5_6_code_complete": implementation_passed,
            "measured_runner_available_and_current_status_valid": bool(
                stage_gates["m5_3_status_is_fail_closed_or_executed"]
            ),
        },
        "empirical_report_paths": {
            name: _display(path) for name, path in empirical_paths.items()
        },
        "empirical_checks": empirical_checks,
        "empirical_exit": {
            "passed": empirical_passed,
            "missing_evidence": missing_empirical,
        },
        "production_enablement": {
            "ready": bool(implementation_passed and empirical_passed),
            "m5_renderer_default_enabled": False,
            "reason": (
                "implementation is complete, but production remains disabled "
                "until every controlled measured-room, listening, and "
                "room-disjoint downstream gate passes"
            ),
        },
        "next_action": {
            "external_dependency": (
                "acquire/ingest a qualifying repeated-ESS controlled campaign"
            ),
            "then_run": [
                "fit_m5_measured_campaign.py",
                "measured synchronized M5.4 candidate profile",
                "measured residual training and ablation",
                "controlled listening and downstream evaluation",
                "validate_m5_exit.py",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    args = parser.parse_args()
    try:
        reports = {stage: _load(path) for stage, path in DEFAULT_REPORTS.items()}
        empirical = {
            name: (_load(path) if path.is_file() else None)
            for name, path in DEFAULT_EMPIRICAL_REPORTS.items()
        }
        report = build_report(reports, empirical_reports=empirical)
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    for name, passed in report["stage_gates"].items():
        print(f"stage\t{name}\t{'PASS' if passed else 'FAIL'}")
    for name, passed in report["artifact_checks"].items():
        print(f"artifact\t{name}\t{'PASS' if passed else 'FAIL'}")
    print(
        "# M5 implementation exit: "
        f"{'PASS' if report['implementation_exit']['passed'] else 'FAIL'}"
    )
    print(
        "# M5 empirical/production exit: "
        f"{'PASS' if report['empirical_exit']['passed'] else 'OPEN'}"
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"# wrote {args.output_report}")
    return 0 if report["implementation_exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
