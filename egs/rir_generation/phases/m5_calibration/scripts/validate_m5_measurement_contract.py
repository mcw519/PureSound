#!/usr/bin/env python3
"""Validate the M5.1 measurement schema, loss contract, and data readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from puresound.audio.rir_calibration import (
    RIR_CALIBRATION_LOSS_POLICY,
    CalibrationLossWeights,
    analyze_rir_calibration_loss,
    causality_penalty,
    spatial_coherence_distance,
)
from puresound.audio.rir_measurement_campaign import (
    RIR_MEASUREMENT_CAMPAIGN_SCHEMA_VERSION,
    ROOM_SPLITS,
    CalibratedTransducer,
    MeasuredRIRRecord,
    MeasuredRoom,
    MeasurementAsset,
    MeasurementPose,
    RIRMeasurementCampaign,
    SweepCapture,
    audit_measurement_campaign,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
PHASE_DIR = REPO_ROOT / "egs/rir_generation/phases/m5_calibration"
CONFIG_DIR = PHASE_DIR / "config"
REPORT_DIR = PHASE_DIR / "reports"
DEFAULT_REPORT = REPORT_DIR / "m5_measurement_contract_report.json"
DEFAULT_TEMPLATE = CONFIG_DIR / "m5_measurement_campaign_template.json"
DEFAULT_BANKS = {
    "train_view": Path("/work/any_exp_link/puresound_exp/real_rir_16k_train_view"),
    "heldout_view": Path("/work/any_exp_link/puresound_exp/real_rir_16k_heldout_view"),
}
ZERO_SHA256 = "0" * 64


def _audio_asset(path: str, channels: int) -> MeasurementAsset:
    return MeasurementAsset(
        path=path,
        sha256=ZERO_SHA256,
        media_type="audio/wav",
        sample_rate=48000,
        channel_count=channels,
        frame_count=288000,
    )


def build_campaign_template() -> RIRMeasurementCampaign:
    """Build a structurally valid, explicitly non-measured campaign template."""

    calibration = lambda path: _audio_asset(path, 1)
    transducers = (
        CalibratedTransducer(
            transducer_id="source-reference",
            kind="source",
            manufacturer="REPLACE_ME",
            model="REPLACE_ME",
            serial_number="REPLACE_ME",
            calibration_asset=calibration("calibration/source_response.wav"),
            calibration_date_utc="1970-01-01T00:00:00Z",
            reference_axis="+x",
            provenance={"template_only": True, "replace_with": "anechoic response"},
        ),
        CalibratedTransducer(
            transducer_id="receiver-left",
            kind="receiver",
            manufacturer="REPLACE_ME",
            model="REPLACE_ME",
            serial_number="REPLACE_ME_LEFT",
            calibration_asset=calibration("calibration/receiver_left.wav"),
            calibration_date_utc="1970-01-01T00:00:00Z",
            reference_axis="+x",
            provenance={"template_only": True, "replace_with": "calibrated response"},
        ),
        CalibratedTransducer(
            transducer_id="receiver-right",
            kind="receiver",
            manufacturer="REPLACE_ME",
            model="REPLACE_ME",
            serial_number="REPLACE_ME_RIGHT",
            calibration_asset=calibration("calibration/receiver_right.wav"),
            calibration_date_utc="1970-01-01T00:00:00Z",
            reference_axis="+x",
            provenance={"template_only": True, "replace_with": "calibrated response"},
        ),
    )
    rooms = tuple(
        MeasuredRoom(
            room_id=f"replace-me-{split}-room",
            room_type="REPLACE_ME",
            coordinate_frame="right-handed xyz, metres",
            geometry_uncertainty_m=0.01,
            dimensions_m=(5.0, 4.0, 2.8),
            provenance={"template_only": True, "method": "laser measurement"},
        )
        for split in ROOM_SPLITS
    )
    source_pose = MeasurementPose(
        position_m=(1.0, 1.0, 1.4),
        orientation_ypr_deg=(0.0, 0.0, 0.0),
        position_std_m=0.005,
        orientation_std_deg=1.0,
    )
    receiver_poses = (
        MeasurementPose(
            position_m=(3.0, 2.0, 1.4),
            orientation_ypr_deg=(180.0, 0.0, 0.0),
            position_std_m=0.005,
            orientation_std_deg=1.0,
        ),
        MeasurementPose(
            position_m=(3.17, 2.0, 1.4),
            orientation_ypr_deg=(180.0, 0.0, 0.0),
            position_std_m=0.005,
            orientation_std_deg=1.0,
        ),
    )
    records = []
    for room, split in zip(rooms, ROOM_SPLITS):
        prefix = f"captures/{room.room_id}/position-000"
        capture = SweepCapture(
            sample_rate=48000,
            start_frequency_hz=20.0,
            end_frequency_hz=22000.0,
            duration_s=6.0,
            fade_s=0.05,
            silence_s=1.0,
            playback_level_dbfs=-12.0,
            raw_recordings=(
                _audio_asset(f"{prefix}/sweep-repeat-00.wav", 2),
                _audio_asset(f"{prefix}/sweep-repeat-01.wav", 2),
            ),
            inverse_filter_asset=_audio_asset("excitation/ess-inverse.wav", 1),
            noise_recording_asset=_audio_asset(f"{prefix}/noise.wav", 2),
            deconvolved_rir_asset=_audio_asset(f"{prefix}/rir.wav", 2),
            latency_correction_samples=0,
            deconvolution_config={
                "template_only": True,
                "harmonic_separation": True,
                "window": "REPLACE_ME",
            },
        )
        records.append(
            MeasuredRIRRecord(
                measurement_id=f"replace-me-{split}-000",
                room_id=room.room_id,
                session_id=f"replace-me-{split}-session",
                timestamp_utc="1970-01-01T00:00:00Z",
                source_id="source-reference",
                source_pose=source_pose,
                receiver_ids=("receiver-left", "receiver-right"),
                receiver_poses=receiver_poses,
                capture=capture,
                temperature_c=20.0,
                relative_humidity_percent=50.0,
                pressure_pa=101325.0,
                synchronized_receivers=True,
                notes="Template only; replace every placeholder and asset hash.",
            )
        )
    return RIRMeasurementCampaign(
        campaign_id="m5-controlled-room-template-not-measurement-evidence",
        rooms=rooms,
        transducers=transducers,
        records=tuple(records),
        room_splits={room.room_id: split for room, split in zip(rooms, ROOM_SPLITS)},
        provenance={
            "template_only": True,
            "measurement_evidence": False,
            "warning": "Zero hashes and REPLACE_ME values are intentionally invalid evidence.",
        },
    )


def _nested(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _legacy_item_evidence(value: Mapping[str, Any]) -> dict[str, bool]:
    scene = value.get("scene", {})
    capture = value.get("capture", _nested(value, "measurement", "capture"))
    capture = capture if isinstance(capture, Mapping) else {}
    source_pose = scene.get("source_pose") if isinstance(scene, Mapping) else None
    receiver_poses = scene.get("receiver_poses") if isinstance(scene, Mapping) else None
    environment = scene.get("environment") if isinstance(scene, Mapping) else None
    channel_map = scene.get("channel_map") if isinstance(scene, Mapping) else None
    receiver_semantics = bool(channel_map) and all(
        isinstance(item, Mapping) and item.get("receiver_id") for item in channel_map
    )
    return {
        "stable_room_identity": bool(
            isinstance(scene, Mapping) and scene.get("room_id")
        ),
        "room_geometry_or_mesh": bool(
            isinstance(scene, Mapping)
            and (scene.get("room_dim") or scene.get("mesh_asset"))
        ),
        "source_position_and_orientation": bool(
            isinstance(source_pose, Mapping)
            and source_pose.get("position_m") is not None
            and source_pose.get("orientation_ypr_deg") is not None
        ),
        "receiver_positions_and_orientations": bool(
            isinstance(receiver_poses, Sequence)
            and not isinstance(receiver_poses, (str, bytes))
            and receiver_poses
            and all(
                isinstance(pose, Mapping)
                and pose.get("position_m") is not None
                and pose.get("orientation_ypr_deg") is not None
                for pose in receiver_poses
            )
        ),
        "calibrated_source_and_receiver_responses": bool(
            isinstance(scene, Mapping)
            and scene.get("source_calibration")
            and scene.get("receiver_calibrations")
        ),
        "temperature_humidity_and_pressure": bool(
            isinstance(environment, Mapping)
            and all(
                environment.get(key) is not None
                for key in (
                    "temperature_c",
                    "relative_humidity_percent",
                    "pressure_pa",
                )
            )
        ),
        "repeated_raw_ess_retained": bool(
            capture.get("method") == "exponential_sine_sweep"
            and isinstance(capture.get("raw_recordings"), list)
            and len(capture["raw_recordings"]) >= 2
        ),
        "deconvolution_inverse_and_noise_retained": bool(
            capture.get("deconvolution_config")
            and capture.get("inverse_filter_asset")
            and capture.get("noise_recording_asset")
            and capture.get("deconvolved_rir_asset")
        ),
        "synchronized_receiver_channel_semantics": bool(
            receiver_semantics and value.get("synchronized_receivers") is True
        ),
    }


def audit_legacy_bank_metadata(bank_root: Path, sample_limit: int) -> dict[str, Any]:
    """Check whether an existing bank can be promoted to controlled M5 evidence."""

    item_root = bank_root / "items"
    paths = (
        sorted(item_root.glob("*.json"))[:sample_limit] if item_root.is_dir() else []
    )
    field_counts: dict[str, int] = {}
    invalid_json = []
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            evidence = _legacy_item_evidence(value)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            invalid_json.append(path.name)
            continue
        for field, present in evidence.items():
            field_counts[field] = field_counts.get(field, 0) + int(present)
    fields = tuple(_legacy_item_evidence({}))
    sampled = len(paths)
    all_sampled = {
        field: sampled > 0 and field_counts.get(field, 0) == sampled for field in fields
    }
    ready = bool(sampled and not invalid_json and all(all_sampled.values()))
    return {
        "bank_root": str(bank_root),
        "available": item_root.is_dir(),
        "sample_limit": int(sample_limit),
        "sampled_json_count": sampled,
        "invalid_json": invalid_json,
        "field_presence_counts": {
            field: field_counts.get(field, 0) for field in fields
        },
        "all_sampled_items_contain": all_sampled,
        "ready_for_m5_inverse_calibration": ready,
        "note": (
            "Legacy source-channel banks remain useful acoustic references, "
            "but missing acquisition evidence cannot be reconstructed from RIR WAVs."
        ),
    }


def _loss_contract_probe() -> dict[str, Any]:
    sample_rate = 16000
    sample_count = 4096
    direct = (32, 35)
    rng = np.random.default_rng(20260801)
    time = np.arange(sample_count) / sample_rate
    common = rng.normal(size=sample_count) * np.exp(-6.0 * time) * 0.025
    independent = rng.normal(size=sample_count) * np.exp(-6.0 * time) * 0.008
    measured = np.zeros((2, sample_count), dtype=np.float64)
    measured[0, direct[0] :] = common[: sample_count - direct[0]]
    measured[1, direct[1] :] = (
        0.8 * common[: sample_count - direct[1]]
        + independent[: sample_count - direct[1]]
    )
    measured[0, direct[0]] = 1.0
    measured[1, direct[1]] = 0.9
    weights = CalibrationLossWeights(decay_regularization=0.0)
    identity = analyze_rir_calibration_loss(
        measured,
        measured.copy(),
        sample_rate,
        measured_direct_samples=direct,
        synthetic_direct_samples=direct,
        weights=weights,
    )
    delayed = np.zeros_like(measured)
    delayed[:, 8:] = measured[:, :-8]
    delayed_report = analyze_rir_calibration_loss(
        measured,
        delayed,
        sample_rate,
        measured_direct_samples=direct,
        synthetic_direct_samples=(40, 43),
        weights=weights,
    )
    noncausal = measured.copy()
    noncausal[:, 4] = 0.25
    causality = causality_penalty(noncausal, direct)
    decorrelated = measured.copy()
    decorrelated[1, direct[1] :] += rng.normal(size=sample_count - direct[1]) * 0.05
    spatial = spatial_coherence_distance(
        measured,
        decorrelated,
        sample_rate,
        direct,
        direct,
    )
    mono = spatial_coherence_distance(
        measured[0],
        measured[0],
        sample_rate,
        direct[:1],
        direct[:1],
    )
    fidelity_terms = tuple(
        name for name in identity.terms if name != "decay_regularization"
    )
    checks = {
        "identity_fidelity_terms_are_zero": all(
            abs(identity.terms[name]) <= 1e-12 for name in fidelity_terms
        ),
        "eight_sample_delay_is_detected": bool(
            delayed_report.terms["arrival_timing"] > 0.0
        ),
        "prearrival_energy_is_penalized": bool(causality["distance"] > 0.0),
        "spatial_perturbation_is_detected": bool(
            spatial["evaluable"] and spatial["distance"] > 0.0
        ),
        "mono_spatial_term_is_explicitly_unevaluable": bool(
            not mono["evaluable"]
            and mono["reason"] == "fewer_than_two_synchronized_receivers"
        ),
    }
    return {
        "policy": RIR_CALIBRATION_LOSS_POLICY,
        "checks": checks,
        "identity": identity.to_dict(),
        "delayed_arrival_timing": delayed_report.diagnostics["arrival_timing"],
        "noncausal": causality,
        "spatial_perturbation": spatial,
        "mono_spatial": mono,
        "passed": bool(all(checks.values())),
    }


def build_report(
    banks: Mapping[str, Path],
    *,
    sample_limit: int = 64,
) -> tuple[dict[str, Any], RIRMeasurementCampaign]:
    if sample_limit < 1:
        raise ValueError("sample_limit must be positive")
    template = build_campaign_template()
    roundtrip = RIRMeasurementCampaign.from_json(template.to_json())
    template_audit = audit_measurement_campaign(
        template,
        REPO_ROOT,
        verify_hashes=False,
        minimum_records_per_room=12,
    )
    loss_probe = _loss_contract_probe()
    legacy_audits = {
        name: audit_legacy_bank_metadata(path, sample_limit)
        for name, path in banks.items()
    }
    implementation_checks = {
        "campaign_schema_version_frozen": (
            template.schema_version == RIR_MEASUREMENT_CAMPAIGN_SCHEMA_VERSION
        ),
        "campaign_strict_json_roundtrip": roundtrip.to_dict() == template.to_dict(),
        "room_disjoint_split_vocabulary_frozen": tuple(ROOM_SPLITS)
        == ("train", "validation", "test"),
        "template_contains_all_room_splits": template_audit["checks"][
            "train_validation_test_rooms_present"
        ],
        "template_requires_repeated_sweeps": template_audit["checks"][
            "every_capture_has_repeated_raw_sweeps"
        ],
        "template_requires_synchronized_spatial_capture": template_audit["checks"][
            "at_least_one_synchronized_spatial_record"
        ],
        "multiobjective_loss_probe_passed": loss_probe["passed"],
    }
    implementation_passed = bool(all(implementation_checks.values()))
    controlled_data_ready = bool(
        legacy_audits
        and all(
            audit["ready_for_m5_inverse_calibration"]
            for audit in legacy_audits.values()
        )
    )
    report = {
        "schema_version": "puresound.m5_measurement_contract_report.v1",
        "milestone": "M5.1",
        "scope": "controlled_measurement_and_calibration_loss_contract",
        "implementation_checks": implementation_checks,
        "loss_contract_probe": loss_probe,
        "campaign_template": {
            "schema_version": template.schema_version,
            "template_only": True,
            "measurement_evidence": False,
            "room_count": len(template.rooms),
            "record_count": len(template.records),
            "minimum_records_per_room_for_campaign": 12,
            "audit": template_audit,
        },
        "existing_measured_bank_audits": legacy_audits,
        "implementation_exit": {
            "passed": implementation_passed,
            "m5_1_complete": implementation_passed,
        },
        "controlled_measurement_exit": {
            "passed": controlled_data_ready,
            "ready_to_start_measured_inverse_fit": controlled_data_ready,
            "reason": (
                "existing measured banks satisfy the controlled acquisition contract"
                if controlled_data_ready
                else "controlled raw ESS, calibration, geometry/pose, environment, and room-split evidence is incomplete"
            ),
        },
        "next_stage": {
            "milestone": "M5.2",
            "action": "synthetic-recovery inverse calibration baseline",
            "may_start_without_claiming_measured_room_fit": implementation_passed,
        },
    }
    json.dumps(report, allow_nan=False)
    return report, template


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bank", type=Path, default=DEFAULT_BANKS["train_view"])
    parser.add_argument(
        "--heldout-bank", type=Path, default=DEFAULT_BANKS["heldout_view"]
    )
    parser.add_argument("--sample-limit", type=int, default=64)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-template", type=Path, default=DEFAULT_TEMPLATE)
    args = parser.parse_args()
    banks = {"train_view": args.train_bank, "heldout_view": args.heldout_bank}
    report, template = build_report(banks, sample_limit=args.sample_limit)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_template.parent.mkdir(parents=True, exist_ok=True)
    args.output_template.write_text(template.to_json(indent=2) + "\n", encoding="utf-8")
    for key, passed in report["implementation_checks"].items():
        print(f"implementation\t{key}\t{'PASS' if passed else 'FAIL'}")
    print(
        "# M5.1 implementation exit: "
        f"{'PASS' if report['implementation_exit']['passed'] else 'FAIL'}"
    )
    print(
        "# controlled measurement readiness: "
        f"{'PASS' if report['controlled_measurement_exit']['passed'] else 'OPEN'}"
    )
    print(f"# wrote {args.output_report}")
    print(f"# wrote {args.output_template}")
    return 0 if report["implementation_exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
