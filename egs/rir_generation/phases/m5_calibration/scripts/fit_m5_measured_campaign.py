#!/usr/bin/env python3
"""Run the fail-closed M5.3 controlled measured-room calibration pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.rir_measured_calibration import (  # noqa: E402
    M5_MEASURED_ROOM_FIT_POLICY,
    CampaignNotReadyError,
    run_measured_campaign_fit,
)
from puresound.audio.rir_measurement_campaign import (  # noqa: E402
    RIRMeasurementCampaign,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--minimum-records-per-room", type=int, default=12)
    parser.add_argument("--position-holdout-fraction", type=float, default=0.25)
    parser.add_argument(
        "--mixing-time-ms",
        type=float,
        nargs="+",
        default=(20.0, 24.0, 32.0),
    )
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--maximum-evaluations", type=int, default=60)
    args = parser.parse_args()

    campaign = RIRMeasurementCampaign.from_json(
        args.campaign.read_text(encoding="utf-8")
    )
    try:
        report = run_measured_campaign_fit(
            campaign,
            args.asset_root,
            minimum_records_per_room=args.minimum_records_per_room,
            position_holdout_fraction=args.position_holdout_fraction,
            mixing_time_candidates_s=tuple(
                value * 1e-3 for value in args.mixing_time_ms
            ),
            max_order=args.max_order,
            maximum_evaluations=args.maximum_evaluations,
        )
        exit_code = 0 if report["exit"]["passed"] else 1
    except CampaignNotReadyError as exc:
        report = {
            "schema_version": "puresound.m5_measured_room_fit_report.v1",
            "milestone": "M5.3",
            "policy": M5_MEASURED_ROOM_FIT_POLICY,
            "campaign_id": campaign.campaign_id,
            "campaign_root": str(args.asset_root),
            "campaign_audit": exc.audit,
            "checks": {
                "controlled_campaign_audit_passed": False,
                "fitting_not_started_with_incomplete_evidence": True,
            },
            "exit": {
                "passed": False,
                "m5_3_measured_fit_executed": False,
                "blocked": True,
                "reason": str(exc),
                "production_enabled": False,
            },
        }
        exit_code = 2
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "# M5.3 measured-room fit: "
        + ("PASS" if report["exit"]["passed"] else "BLOCKED/FAIL (see report)")
    )
    print(f"# wrote {args.output_report}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
