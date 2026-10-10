import json
from pathlib import Path

import pytest

from puresound.audio.rir.calibration.measured_runner import (
    CampaignNotReadyError,
    deterministic_position_assignments,
    run_measured_campaign_fit,
)
from puresound.audio.rir.calibration.measured_campaign import (
    CalibratedTransducer,
    MeasuredRIRRecord,
    MeasuredRoom,
    MeasurementAsset,
    MeasurementPose,
    RIRMeasurementCampaign,
    SweepCapture,
    audit_measurement_campaign,
    sha256_file,
)


def _asset(root, path, *, channels=2, frames=64):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((path * 3).encode("utf-8"))
    return MeasurementAsset(
        path=path,
        sha256=sha256_file(target),
        media_type="audio/wav",
        sample_rate=16000,
        channel_count=channels,
        frame_count=frames,
    )


def _campaign(root):
    source_calibration = _asset(root, "calibration/source.wav", channels=1)
    left_calibration = _asset(root, "calibration/left.wav", channels=1)
    right_calibration = _asset(root, "calibration/right.wav", channels=1)
    transducers = (
        CalibratedTransducer(
            transducer_id="source-0",
            kind="source",
            manufacturer="unit",
            model="speaker",
            serial_number="s0",
            calibration_asset=source_calibration,
            calibration_date_utc="2026-08-01T00:00:00Z",
            reference_axis="+x",
            provenance={"method": "anechoic reference"},
        ),
        CalibratedTransducer(
            transducer_id="receiver-left",
            kind="receiver",
            manufacturer="unit",
            model="microphone",
            serial_number="m0",
            calibration_asset=left_calibration,
            calibration_date_utc="2026-08-01T00:00:00Z",
            reference_axis="+x",
            provenance={"method": "pressure calibrator"},
        ),
        CalibratedTransducer(
            transducer_id="receiver-right",
            kind="receiver",
            manufacturer="unit",
            model="microphone",
            serial_number="m1",
            calibration_asset=right_calibration,
            calibration_date_utc="2026-08-01T00:00:00Z",
            reference_axis="+x",
            provenance={"method": "pressure calibrator"},
        ),
    )
    rooms = tuple(
        MeasuredRoom(
            room_id=f"room-{index}",
            room_type="office",
            coordinate_frame="right-handed xyz, metres",
            geometry_uncertainty_m=0.01,
            dimensions_m=(5.0 + index, 4.0, 2.8),
            provenance={"method": "laser range finder"},
        )
        for index in range(3)
    )
    pose = MeasurementPose(
        position_m=(1.0, 1.0, 1.4),
        orientation_ypr_deg=(0.0, 0.0, 0.0),
        position_std_m=0.005,
        orientation_std_deg=1.0,
    )
    records = []
    for index, room in enumerate(rooms):
        prefix = f"captures/{room.room_id}"
        capture = SweepCapture(
            sample_rate=16000,
            start_frequency_hz=40.0,
            end_frequency_hz=7000.0,
            duration_s=5.0,
            fade_s=0.05,
            silence_s=1.0,
            playback_level_dbfs=-12.0,
            raw_recordings=(
                _asset(root, f"{prefix}/raw-0.wav"),
                _asset(root, f"{prefix}/raw-1.wav"),
            ),
            inverse_filter_asset=_asset(
                root,
                f"{prefix}/inverse.wav",
                channels=1,
            ),
            noise_recording_asset=_asset(root, f"{prefix}/noise.wav"),
            deconvolved_rir_asset=_asset(root, f"{prefix}/rir.wav"),
            latency_correction_samples=17,
            deconvolution_config={
                "window": "tukey",
                "harmonic_rejection": True,
            },
        )
        records.append(
            MeasuredRIRRecord(
                measurement_id=f"measurement-{index}",
                room_id=room.room_id,
                session_id=f"session-{index}",
                timestamp_utc="2026-08-01T00:00:00Z",
                source_id="source-0",
                source_pose=pose,
                receiver_ids=("receiver-left", "receiver-right"),
                receiver_poses=(pose, pose),
                capture=capture,
                temperature_c=22.0,
                relative_humidity_percent=50.0,
                pressure_pa=101325.0,
                synchronized_receivers=True,
            )
        )
    return RIRMeasurementCampaign(
        campaign_id="unit-campaign",
        rooms=rooms,
        transducers=transducers,
        records=tuple(records),
        room_splits={
            "room-0": "train",
            "room-1": "validation",
            "room-2": "test",
        },
        provenance={"operator": "unit", "license": "unit-test"},
    )


def test_campaign_round_trips_and_its_audit_checks_every_retained_asset(tmp_path):
    campaign = _campaign(tmp_path)
    restored = RIRMeasurementCampaign.from_json(campaign.to_json())
    audit = audit_measurement_campaign(
        restored,
        tmp_path,
        minimum_records_per_room=1,
    )

    assert restored.to_dict() == campaign.to_dict()
    assert all(audit["checks"].values())
    assert audit["ready_for_m5_inverse_calibration"] is True
    assert audit["split_room_counts"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    json.dumps(audit, allow_nan=False)

    corrupt = tmp_path / campaign.records[0].capture.raw_recordings[0].path
    corrupt.write_bytes(b"corrupt")

    corrupted = audit_measurement_campaign(
        campaign,
        tmp_path,
        minimum_records_per_room=1,
    )

    assert corrupted["checks"]["all_retained_asset_hashes_match"] is False
    assert (
        campaign.records[0].capture.raw_recordings[0].path
        in corrupted["hash_mismatch_paths"]
    )
    assert corrupted["ready_for_m5_inverse_calibration"] is False


def test_measurement_contract_rejects_unsafe_assets_and_unsynchronized_array(
    tmp_path,
):
    with pytest.raises(ValueError, match="relative paths"):
        MeasurementAsset(
            path="/absolute/raw.wav",
            sha256="0" * 64,
            media_type="audio/wav",
            sample_rate=16000,
            channel_count=2,
            frame_count=64,
        )

    campaign = _campaign(tmp_path)
    record = campaign.records[0]
    with pytest.raises(ValueError, match="sample-synchronized"):
        MeasuredRIRRecord(
            measurement_id="bad-sync",
            room_id=record.room_id,
            session_id=record.session_id,
            timestamp_utc=record.timestamp_utc,
            source_id=record.source_id,
            source_pose=record.source_pose,
            receiver_ids=record.receiver_ids,
            receiver_poses=record.receiver_poses,
            capture=record.capture,
            temperature_c=record.temperature_c,
            relative_humidity_percent=record.relative_humidity_percent,
            pressure_pa=record.pressure_pa,
            synchronized_receivers=False,
        )


def _template_campaign():
    template = (
        Path(__file__).resolve().parents[2]
        / "egs/rir_generation/phases/m5_calibration/config"
        / "m5_measurement_campaign_template.json"
    )
    return RIRMeasurementCampaign.from_json(template.read_text(encoding="utf-8"))


def test_measured_runner_refuses_an_incomplete_campaign_before_fitting(tmp_path):
    """The shipped template has no measured assets, so it must stop at the audit."""
    campaign = _template_campaign()
    assert RIRMeasurementCampaign.from_json(campaign.to_json()) == campaign

    with pytest.raises(CampaignNotReadyError) as caught:
        run_measured_campaign_fit(campaign, tmp_path)

    assert caught.value.audit["ready_for_m5_inverse_calibration"] is False
    assert caught.value.audit["missing_asset_paths"]
    json.dumps(caught.value.audit, allow_nan=False)
    # A training room needs a held-out position as well as a fitted one.
    with pytest.raises(ValueError, match="at least two positions"):
        deterministic_position_assignments(campaign)
