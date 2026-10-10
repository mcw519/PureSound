#!/usr/bin/env python
"""Aggregate the M4.1-M4.6 implementation exit without overstating evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
CONFIG_DIR = REPO_ROOT / "egs/rir_generation/phases/m4_spatial_late_field/reports"
DEFAULT_REPORTS = {
    "M4.1": CONFIG_DIR / "m4_late_field_baseline.json",
    "M4.2": CONFIG_DIR / "m4_multiband_late_field.json",
    "M4.3": CONFIG_DIR / "m4_multiband_fdn_report.json",
    "M4.4": CONFIG_DIR / "m4_path_event_fdn_coupling_report.json",
    "M4.5": CONFIG_DIR / "m4_spatial_rir_report.json",
    "M4.6": CONFIG_DIR / "m4_binaural_brir_report.json",
}
DEFAULT_OUTPUT_REPORT = CONFIG_DIR / "m4_exit_report.json"
REQUIRED_ARTIFACTS = {
    "receiver_array_rir": (
        REPO_ROOT
        / "egs/rir_generation/exp/rir_realism/m4/rir_m4_spatial/m4_spatial_receiver_rir_2ch.wav"
    ),
    "ambisonic_rir": (
        REPO_ROOT
        / "egs/rir_generation/exp/rir_realism/m4/rir_m4_spatial/m4_spatial_ambisonic_rir_4ch.wav"
    ),
    "analytic_brir": (
        REPO_ROOT / "egs/rir_generation/exp/rir_realism/m4/rir_m4_spatial/m4_analytic_brir_2ch.wav"
    ),
}


def _load_reports(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    reports = {}
    for stage, path in paths.items():
        reports[stage] = json.loads(path.read_text(encoding="utf-8"))
    return reports


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_report(
    reports: dict[str, dict[str, Any]],
    *,
    report_paths: dict[str, Path] | None = None,
    artifact_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Return separate implementation and empirical/production exit states."""

    expected_stages = tuple(DEFAULT_REPORTS)
    stage_identity = {
        "M4.1": reports.get("M4.1", {}).get("milestone") == "M4.1",
        "M4.2": reports.get("M4.2", {}).get("milestone") == "M4.2",
        "M4.3": reports.get("M4.3", {}).get("milestone") == "M4.3",
        "M4.4": reports.get("M4.4", {}).get("milestone") == "M4.4",
        "M4.5": reports.get("M4.5", {}).get("stage") == "M4.5",
        "M4.6": reports.get("M4.6", {}).get("stage") == "M4.6",
    }
    stage_gates = {
        "M4.1_measurement_foundation_present": bool(stage_identity["M4.1"]),
        "M4.2_multiband_spatial_contract_present": bool(
            stage_identity["M4.2"]
            and "spatial_contract" in reports.get("M4.2", {})
        ),
        "M4.3_multiband_fdn_passed": bool(
            stage_identity["M4.3"]
            and reports["M4.3"].get("exit", {}).get("passed", False)
        ),
        "M4.4_early_late_coupling_passed": bool(
            stage_identity["M4.4"]
            and reports["M4.4"].get("exit", {}).get("passed", False)
        ),
        "M4.5_spatial_array_ambisonic_passed": bool(
            stage_identity["M4.5"]
            and reports["M4.5"].get("exit", {}).get("passed", False)
        ),
        "M4.6_directivity_brir_api_passed": bool(
            stage_identity["M4.6"]
            and reports["M4.6"].get("exit", {}).get("passed", False)
        ),
    }
    resolved_artifacts = artifact_paths or REQUIRED_ARTIFACTS
    artifact_checks = {
        name: path.is_file() for name, path in resolved_artifacts.items()
    }
    invariant_checks = {
        "production_default_unchanged": bool(
            reports.get("M4.5", {})
            .get("renderer", {})
            .get("production_default_changed")
            is False
        ),
        "measured_hrtf_not_falsely_bundled": bool(
            reports.get("M4.6", {})
            .get("exit", {})
            .get("measured_hrtf_bundled")
            is False
        ),
        "measured_spatial_exit_not_falsely_claimed": bool(
            reports.get("M4.5", {})
            .get("exit", {})
            .get("measured_multi_receiver_validation_complete")
            is False
        ),
    }
    implementation_passed = bool(
        all(stage_gates.values())
        and all(artifact_checks.values())
        and all(invariant_checks.values())
    )
    missing_empirical_evidence = [
        "measured synchronized multi-receiver RIR validation",
        "licensed measured HRTF decoder calibration",
        "controlled listening test for metallic coloration and spatial plausibility",
    ]
    paths = report_paths or DEFAULT_REPORTS
    return {
        "schema_version": "puresound.m4_exit.v1",
        "milestone": "M4",
        "scope": "implementation_exit_with_separate_empirical_status",
        "expected_stages": list(expected_stages),
        "input_reports": {
            stage: _display_path(paths[stage]) for stage in expected_stages
        },
        "stage_identity": stage_identity,
        "stage_gates": stage_gates,
        "artifact_checks": artifact_checks,
        "artifact_paths": {
            name: _display_path(path)
            for name, path in resolved_artifacts.items()
        },
        "invariant_checks": invariant_checks,
        "implementation_exit": {
            "passed": implementation_passed,
            "m4_1_through_m4_6_complete": implementation_passed,
            "spatial_renderer_available_as_explicit_opt_in": implementation_passed,
        },
        "empirical_exit": {
            "passed": False,
            "missing_evidence": missing_empirical_evidence,
        },
        "production_enablement": {
            "ready": False,
            "default_backend": "pyroomacoustics",
            "m4_renderer": "explicit opt-in API/CLI",
            "reason": (
                "implementation gates pass, while measured spatial, licensed "
                "HRTF, and controlled-listening evidence remains open"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_OUTPUT_REPORT,
    )
    args = parser.parse_args()
    try:
        reports = _load_reports(DEFAULT_REPORTS)
        report = build_report(reports)
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        parser.error(str(exc))

    for key, value in report["stage_gates"].items():
        print(f"stage\t{key}\t{'PASS' if value else 'FAIL'}")
    for key, value in report["artifact_checks"].items():
        print(f"artifact\t{key}\t{'PASS' if value else 'FAIL'}")
    print(
        "# M4 implementation exit: "
        f"{'PASS' if report['implementation_exit']['passed'] else 'FAIL'}"
    )
    print("# M4 empirical/production exit: OPEN")
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"# wrote {args.output_report}")
    return 0 if report["implementation_exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
