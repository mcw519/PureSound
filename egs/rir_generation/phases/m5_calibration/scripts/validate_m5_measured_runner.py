#!/usr/bin/env python3
"""Validate the M5.3 runner on a complete synthetic campaign fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.rir_m5_pipeline import (  # noqa: E402
    GroupedPathObservation,
    GroupedPathParameters,
    render_grouped_path_observation,
)
from puresound.audio.rir_measured_calibration import (  # noqa: E402
    deterministic_position_assignments,
    run_measured_campaign_fit,
)
from puresound.audio.rir_measurement_campaign import (  # noqa: E402
    CalibratedTransducer,
    MeasuredRIRRecord,
    MeasuredRoom,
    MeasurementAsset,
    MeasurementPose,
    RIRMeasurementCampaign,
    SweepCapture,
    sha256_file,
)
from puresound.audio.rir_path_events import (  # noqa: E402
    generate_shoebox_path_events,
)
from puresound.audio.rir_scene import SHOEBOX_BOUNDARIES  # noqa: E402


DEFAULT_REPORT = (
    REPO_ROOT / "egs/rir_generation/phases/m5_calibration/reports/m5_measured_runner_validation_report.json"
)
DEFAULT_FIXTURE_ROOT = REPO_ROOT / "egs/rir_generation/exp/rir_realism/m5/rir_m5_measured_runner_fixture"


def _asset(path: Path, root: Path, channels: int, frames: int) -> MeasurementAsset:
    return MeasurementAsset(
        path=str(path.relative_to(root)),
        sha256=sha256_file(path),
        media_type="audio/wav",
        sample_rate=8000,
        channel_count=int(channels),
        frame_count=int(frames),
    )


def _write_audio(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, values, 8000, subtype="FLOAT")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _channel_seed(measurement_id: str, receiver_index: int) -> int:
    digest = hashlib.sha256(
        f"{measurement_id}:{receiver_index}:20260802".encode()
    ).digest()
    return int.from_bytes(digest[:4], "little")


def _target_rir(
    room: MeasuredRoom,
    measurement_id: str,
    source_pose: MeasurementPose,
    receiver_poses: tuple[MeasurementPose, ...],
    sample_count: int,
    max_order: int,
) -> np.ndarray:
    sound_speed = 331.3 + 0.606 * 22.0 + 0.0124 * 50.0
    surface_ids = {boundary: boundary for boundary in SHOEBOX_BOUNDARIES}
    parameters = GroupedPathParameters(
        mixing_time_s=0.024,
        reflection_adjustment_db_by_group={
            boundary: -1.5 for boundary in SHOEBOX_BOUNDARIES
        },
        target_rt60_s_by_hz={500.0: 0.50, 1000.0: 0.50, 2000.0: 0.50},
    )
    channels = []
    for receiver_index, receiver_pose in enumerate(receiver_poses):
        event_set = generate_shoebox_path_events(
            dimensions_m=room.dimensions_m,
            source_position_m=source_pose.position_m,
            receiver_position_m=receiver_pose.position_m,
            sound_speed_m_s=sound_speed,
            scene_id=f"{room.room_id}:{measurement_id}",
            source_id="source-0",
            receiver_id=f"receiver-{receiver_index}",
            surface_ids=surface_ids,
            max_order=int(max_order),
            edge_corner_policy="exclude",
        )
        direct = next(
            event for event in event_set.events if event.path_type == "direct"
        )
        observation = GroupedPathObservation(
            observation_id=f"{measurement_id}:{receiver_index}",
            event_set=event_set,
            surface_group_by_id={boundary: boundary for boundary in SHOEBOX_BOUNDARIES},
            sample_rate=8000,
            sample_count=int(sample_count),
            direct_sample=int(round(direct.delay_s * 8000)),
            fdn_seed=_channel_seed(measurement_id, receiver_index),
            delay_line_count=4,
        )
        channels.append(render_grouped_path_observation(observation, parameters).rir)
    return np.vstack(channels)


def _build_campaign(
    root: Path,
    *,
    records_per_room: int,
    sample_count: int,
    max_order: int,
) -> RIRMeasurementCampaign:
    root.mkdir(parents=True, exist_ok=True)
    calibration_paths = {
        name: root / f"calibration/{name}.wav"
        for name in ("source", "receiver-0", "receiver-1")
    }
    for path in calibration_paths.values():
        _write_audio(path, np.ones(64, dtype=np.float64))
    transducers = (
        CalibratedTransducer(
            transducer_id="source-0",
            kind="source",
            manufacturer="synthetic",
            model="reference-source",
            serial_number="synthetic-s0",
            calibration_asset=_asset(calibration_paths["source"], root, 1, 64),
            calibration_date_utc="2026-08-02T00:00:00Z",
            reference_axis="+x",
            provenance={"fixture": True},
        ),
        *tuple(
            CalibratedTransducer(
                transducer_id=f"receiver-{index}",
                kind="receiver",
                manufacturer="synthetic",
                model="reference-microphone",
                serial_number=f"synthetic-r{index}",
                calibration_asset=_asset(
                    calibration_paths[f"receiver-{index}"], root, 1, 64
                ),
                calibration_date_utc="2026-08-02T00:00:00Z",
                reference_axis="+x",
                provenance={"fixture": True},
            )
            for index in range(2)
        ),
    )
    rooms = tuple(
        MeasuredRoom(
            room_id=f"fixture-room-{index}",
            room_type="office",
            coordinate_frame="right-handed xyz metres",
            geometry_uncertainty_m=0.002,
            dimensions_m=(5.0 + 0.4 * index, 4.2 + 0.2 * index, 2.9),
            provenance={"fixture": True},
        )
        for index in range(3)
    )
    receiver_poses = (
        MeasurementPose(
            position_m=(2.45, 2.05, 1.35),
            orientation_ypr_deg=(0.0, 0.0, 0.0),
            position_std_m=0.002,
            orientation_std_deg=0.5,
        ),
        MeasurementPose(
            position_m=(2.55, 2.05, 1.35),
            orientation_ypr_deg=(180.0, 0.0, 0.0),
            position_std_m=0.002,
            orientation_std_deg=0.5,
        ),
    )
    common = {}
    for name, channels in (
        ("raw-0", 2),
        ("raw-1", 2),
        ("noise", 2),
        ("inverse", 1),
    ):
        path = root / f"shared/{name}.wav"
        values = np.zeros((sample_count, channels), dtype=np.float64)
        _write_audio(path, values if channels > 1 else values[:, 0])
        common[name] = _asset(path, root, channels, sample_count)
    source_positions = (
        (0.9, 0.9, 1.45),
        (1.2, 3.1, 1.45),
        (4.0, 1.1, 1.45),
        (3.8, 3.2, 1.45),
        (1.8, 1.6, 1.45),
        (3.2, 2.7, 1.45),
    )
    records = []
    for room_index, room in enumerate(rooms):
        for position_index in range(records_per_room):
            measurement_id = f"{room.room_id}-position-{position_index}"
            source_pose = MeasurementPose(
                position_m=source_positions[position_index],
                orientation_ypr_deg=(0.0, 0.0, 0.0),
                position_std_m=0.002,
                orientation_std_deg=0.5,
            )
            target = _target_rir(
                room,
                measurement_id,
                source_pose,
                receiver_poses,
                sample_count,
                max_order,
            )
            rir_path = root / f"rirs/{measurement_id}.wav"
            _write_audio(rir_path, target.T)
            rir_asset = _asset(rir_path, root, 2, sample_count)
            capture = SweepCapture(
                sample_rate=8000,
                start_frequency_hz=40.0,
                end_frequency_hz=3500.0,
                duration_s=3.0,
                fade_s=0.05,
                silence_s=0.5,
                playback_level_dbfs=-12.0,
                raw_recordings=(common["raw-0"], common["raw-1"]),
                inverse_filter_asset=common["inverse"],
                noise_recording_asset=common["noise"],
                deconvolved_rir_asset=rir_asset,
                latency_correction_samples=0,
                deconvolution_config={
                    "fixture": True,
                    "calibrated_and_latency_corrected": True,
                },
            )
            records.append(
                MeasuredRIRRecord(
                    measurement_id=measurement_id,
                    room_id=room.room_id,
                    session_id=f"fixture-session-{room_index}",
                    timestamp_utc="2026-08-02T00:00:00Z",
                    source_id="source-0",
                    source_pose=source_pose,
                    receiver_ids=("receiver-0", "receiver-1"),
                    receiver_poses=receiver_poses,
                    capture=capture,
                    temperature_c=22.0,
                    relative_humidity_percent=50.0,
                    pressure_pa=101325.0,
                    synchronized_receivers=True,
                    notes="synthetic implementation fixture; not measured evidence",
                )
            )
    return RIRMeasurementCampaign(
        campaign_id="m5-measured-runner-synthetic-fixture",
        rooms=rooms,
        transducers=transducers,
        records=tuple(records),
        room_splits={
            rooms[0].room_id: "train",
            rooms[1].room_id: "validation",
            rooms[2].room_id: "test",
        },
        provenance={
            "origin": "synthetic implementation fixture",
            "qualifies_as_measured_evidence": False,
        },
    )


def build_report(
    fixture_root: Path,
    *,
    records_per_room: int = 4,
    sample_count: int = 2400,
    max_order: int = 2,
    maximum_evaluations: int = 40,
) -> dict[str, Any]:
    campaign = _build_campaign(
        fixture_root,
        records_per_room=records_per_room,
        sample_count=sample_count,
        max_order=max_order,
    )
    campaign_path = fixture_root / "campaign.json"
    campaign_path.write_text(campaign.to_json() + "\n", encoding="utf-8")
    first_assignments = deterministic_position_assignments(campaign)
    repeated_assignments = deterministic_position_assignments(campaign)
    fit_report = run_measured_campaign_fit(
        campaign,
        fixture_root,
        minimum_records_per_room=records_per_room,
        max_order=max_order,
        maximum_evaluations=maximum_evaluations,
    )
    checks = {
        "synthetic_fixture_is_explicitly_not_measured_evidence": (
            campaign.provenance["qualifies_as_measured_evidence"] is False
        ),
        "complete_campaign_audit_passes": fit_report["campaign_audit"][
            "ready_for_m5_inverse_calibration"
        ],
        "position_split_is_deterministic": (first_assignments == repeated_assignments),
        "measured_runner_executes_all_stages": fit_report["exit"]["passed"],
        "train_position_holdout_is_evaluated": (
            fit_report["aggregates"]["position_holdout_mean_total"] is not None
        ),
        "validation_room_is_evaluated": (
            fit_report["aggregates"]["heldout_rooms"]["validation"]["mean_total"]
            is not None
        ),
        "test_room_is_evaluated": (
            fit_report["aggregates"]["heldout_rooms"]["test"]["mean_total"] is not None
        ),
        "runner_never_enables_production": (
            fit_report["exit"]["production_enabled"] is False
        ),
    }
    report = {
        "schema_version": "puresound.m5_measured_runner_validation.v1",
        "milestone": "M5.3-implementation",
        "scope": "synthetic_complete_campaign_fixture_not_measured_evidence",
        "fixture_campaign": _display_path(campaign_path),
        "fit_report": fit_report,
        "checks": checks,
        "exit": {
            "passed": bool(all(checks.values())),
            "m5_3_runner_implementation_complete": bool(all(checks.values())),
            "measured_room_empirical_fit_complete": False,
        },
    }
    json.dumps(report, allow_nan=False)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--records-per-room", type=int, default=4)
    parser.add_argument("--sample-count", type=int, default=2400)
    parser.add_argument("--max-order", type=int, default=2)
    parser.add_argument("--maximum-evaluations", type=int, default=40)
    args = parser.parse_args()
    report = build_report(
        args.fixture_root,
        records_per_room=args.records_per_room,
        sample_count=args.sample_count,
        max_order=args.max_order,
        maximum_evaluations=args.maximum_evaluations,
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for name, passed in report["checks"].items():
        print(f"check\t{name}\t{'PASS' if passed else 'FAIL'}")
    print(
        "# M5.3 runner implementation: "
        f"{'PASS' if report['exit']['passed'] else 'FAIL'}"
    )
    print("# real measured-room fit: OPEN")
    print(f"# wrote {args.output_report}")
    return 0 if report["exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
