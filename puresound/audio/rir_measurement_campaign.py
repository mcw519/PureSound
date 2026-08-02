"""Strict controlled-room measurement campaign contract for M5 calibration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


RIR_MEASUREMENT_CAMPAIGN_SCHEMA_VERSION = "puresound.rir_measurement_campaign.v1"
ROOM_SPLITS = ("train", "validation", "test")


def _finite_vector(
    name: str,
    values: Sequence[float],
    size: int,
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _strict_json_mapping(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be strict-JSON serializable") from exc
    return result


def _asset_path(value: str) -> str:
    path = PurePosixPath(str(value))
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError("measurement asset paths must be safe relative paths")
    return path.as_posix()


@dataclass(frozen=True)
class MeasurementAsset:
    """Content-addressed campaign-relative file reference."""

    path: str
    sha256: str
    media_type: str
    sample_rate: int | None = None
    channel_count: int | None = None
    frame_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _asset_path(self.path))
        digest = str(self.sha256).lower()
        if len(digest) != 64 or any(
            value not in "0123456789abcdef" for value in digest
        ):
            raise ValueError("asset sha256 must contain 64 hexadecimal characters")
        if not self.media_type:
            raise ValueError("asset media_type is required")
        audio_fields = (self.sample_rate, self.channel_count, self.frame_count)
        if any(value is not None for value in audio_fields):
            if any(value is None for value in audio_fields):
                raise ValueError(
                    "audio asset shape requires sample rate, channels, and frames"
                )
            if any(int(value) <= 0 for value in audio_fields if value is not None):
                raise ValueError("audio asset shape values must be positive")
        object.__setattr__(self, "sha256", digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "sample_rate": self.sample_rate,
            "channel_count": self.channel_count,
            "frame_count": self.frame_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MeasurementAsset":
        return cls(
            path=str(value["path"]),
            sha256=str(value["sha256"]),
            media_type=str(value["media_type"]),
            sample_rate=(
                int(value["sample_rate"])
                if value.get("sample_rate") is not None
                else None
            ),
            channel_count=(
                int(value["channel_count"])
                if value.get("channel_count") is not None
                else None
            ),
            frame_count=(
                int(value["frame_count"])
                if value.get("frame_count") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class MeasurementPose:
    """Measured transducer pose and one-sigma acquisition uncertainty."""

    position_m: tuple[float, float, float]
    orientation_ypr_deg: tuple[float, float, float]
    position_std_m: float
    orientation_std_deg: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_m",
            _finite_vector("position_m", self.position_m, 3),
        )
        object.__setattr__(
            self,
            "orientation_ypr_deg",
            _finite_vector(
                "orientation_ypr_deg",
                self.orientation_ypr_deg,
                3,
            ),
        )
        if not math.isfinite(self.position_std_m) or self.position_std_m < 0.0:
            raise ValueError("position_std_m must be finite and non-negative")
        if (
            not math.isfinite(self.orientation_std_deg)
            or self.orientation_std_deg < 0.0
        ):
            raise ValueError("orientation_std_deg must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_m": list(self.position_m),
            "orientation_ypr_deg": list(self.orientation_ypr_deg),
            "position_std_m": float(self.position_std_m),
            "orientation_std_deg": float(self.orientation_std_deg),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MeasurementPose":
        return cls(
            position_m=tuple(value["position_m"]),
            orientation_ypr_deg=tuple(
                value.get("orientation_ypr_deg", (0.0, 0.0, 0.0))
            ),
            position_std_m=float(value["position_std_m"]),
            orientation_std_deg=float(value["orientation_std_deg"]),
        )


@dataclass(frozen=True)
class MeasuredRoom:
    """Room geometry and provenance shared by all positions in one room."""

    room_id: str
    room_type: str
    coordinate_frame: str
    geometry_uncertainty_m: float
    provenance: Mapping[str, Any]
    dimensions_m: tuple[float, float, float] | None = None
    mesh_asset: MeasurementAsset | None = None

    def __post_init__(self) -> None:
        if not self.room_id or not self.room_type or not self.coordinate_frame:
            raise ValueError("room id, type, and coordinate frame are required")
        if self.dimensions_m is None and self.mesh_asset is None:
            raise ValueError("room requires dimensions_m or a geometry mesh asset")
        if self.dimensions_m is not None:
            dimensions = _finite_vector("dimensions_m", self.dimensions_m, 3)
            if any(value <= 0.0 for value in dimensions):
                raise ValueError("room dimensions must be positive")
            object.__setattr__(self, "dimensions_m", dimensions)
        if (
            not math.isfinite(self.geometry_uncertainty_m)
            or self.geometry_uncertainty_m < 0.0
        ):
            raise ValueError("geometry_uncertainty_m must be finite and non-negative")
        object.__setattr__(
            self,
            "provenance",
            _strict_json_mapping("room provenance", self.provenance),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "room_type": self.room_type,
            "coordinate_frame": self.coordinate_frame,
            "geometry_uncertainty_m": float(self.geometry_uncertainty_m),
            "dimensions_m": (
                list(self.dimensions_m) if self.dimensions_m is not None else None
            ),
            "mesh_asset": (
                self.mesh_asset.to_dict() if self.mesh_asset is not None else None
            ),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MeasuredRoom":
        return cls(
            room_id=str(value["room_id"]),
            room_type=str(value["room_type"]),
            coordinate_frame=str(value["coordinate_frame"]),
            geometry_uncertainty_m=float(value["geometry_uncertainty_m"]),
            dimensions_m=(
                tuple(value["dimensions_m"])
                if value.get("dimensions_m") is not None
                else None
            ),
            mesh_asset=(
                MeasurementAsset.from_dict(value["mesh_asset"])
                if value.get("mesh_asset") is not None
                else None
            ),
            provenance=dict(value["provenance"]),
        )


@dataclass(frozen=True)
class CalibratedTransducer:
    """Source or receiver identity with a retained calibration response."""

    transducer_id: str
    kind: str
    manufacturer: str
    model: str
    serial_number: str
    calibration_asset: MeasurementAsset
    calibration_date_utc: str
    reference_axis: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.kind not in {"source", "receiver"}:
            raise ValueError("transducer kind must be source or receiver")
        required = (
            self.transducer_id,
            self.manufacturer,
            self.model,
            self.serial_number,
            self.calibration_date_utc,
            self.reference_axis,
        )
        if any(not value for value in required):
            raise ValueError("transducer identity and calibration fields are required")
        object.__setattr__(
            self,
            "provenance",
            _strict_json_mapping("transducer provenance", self.provenance),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "transducer_id": self.transducer_id,
            "kind": self.kind,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "serial_number": self.serial_number,
            "calibration_asset": self.calibration_asset.to_dict(),
            "calibration_date_utc": self.calibration_date_utc,
            "reference_axis": self.reference_axis,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibratedTransducer":
        return cls(
            transducer_id=str(value["transducer_id"]),
            kind=str(value["kind"]),
            manufacturer=str(value["manufacturer"]),
            model=str(value["model"]),
            serial_number=str(value["serial_number"]),
            calibration_asset=MeasurementAsset.from_dict(value["calibration_asset"]),
            calibration_date_utc=str(value["calibration_date_utc"]),
            reference_axis=str(value["reference_axis"]),
            provenance=dict(value["provenance"]),
        )


@dataclass(frozen=True)
class SweepCapture:
    """Repeated ESS acquisition and deterministic deconvolution assets."""

    sample_rate: int
    start_frequency_hz: float
    end_frequency_hz: float
    duration_s: float
    fade_s: float
    silence_s: float
    playback_level_dbfs: float
    raw_recordings: tuple[MeasurementAsset, ...]
    inverse_filter_asset: MeasurementAsset
    noise_recording_asset: MeasurementAsset
    deconvolved_rir_asset: MeasurementAsset
    latency_correction_samples: int
    deconvolution_config: Mapping[str, Any]
    method: str = "exponential_sine_sweep"

    def __post_init__(self) -> None:
        if self.method != "exponential_sine_sweep":
            raise ValueError("M5 controlled captures require exponential sine sweeps")
        if self.sample_rate <= 0:
            raise ValueError("sweep sample_rate must be positive")
        scalars = (
            self.start_frequency_hz,
            self.end_frequency_hz,
            self.duration_s,
            self.fade_s,
            self.silence_s,
            self.playback_level_dbfs,
        )
        if any(not math.isfinite(value) for value in scalars):
            raise ValueError("sweep configuration must be finite")
        if not 0.0 < self.start_frequency_hz < self.end_frequency_hz:
            raise ValueError("sweep frequency range must be increasing and positive")
        if self.end_frequency_hz >= 0.5 * self.sample_rate:
            raise ValueError("sweep end frequency must lie below Nyquist")
        if self.duration_s <= 0.0 or self.fade_s < 0.0 or self.silence_s < 0.0:
            raise ValueError("sweep duration must be positive and margins non-negative")
        if len(self.raw_recordings) < 2:
            raise ValueError("at least two repeated raw sweep recordings are required")
        audio_assets = (
            *self.raw_recordings,
            self.inverse_filter_asset,
            self.noise_recording_asset,
            self.deconvolved_rir_asset,
        )
        if any(asset.sample_rate != self.sample_rate for asset in audio_assets):
            raise ValueError(
                "all sweep audio assets must share the configured sample rate"
            )
        object.__setattr__(
            self,
            "deconvolution_config",
            _strict_json_mapping(
                "deconvolution_config",
                self.deconvolution_config,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "sample_rate": int(self.sample_rate),
            "start_frequency_hz": float(self.start_frequency_hz),
            "end_frequency_hz": float(self.end_frequency_hz),
            "duration_s": float(self.duration_s),
            "fade_s": float(self.fade_s),
            "silence_s": float(self.silence_s),
            "playback_level_dbfs": float(self.playback_level_dbfs),
            "raw_recordings": [asset.to_dict() for asset in self.raw_recordings],
            "inverse_filter_asset": self.inverse_filter_asset.to_dict(),
            "noise_recording_asset": self.noise_recording_asset.to_dict(),
            "deconvolved_rir_asset": self.deconvolved_rir_asset.to_dict(),
            "latency_correction_samples": int(self.latency_correction_samples),
            "deconvolution_config": dict(self.deconvolution_config),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SweepCapture":
        return cls(
            method=str(value.get("method", "")),
            sample_rate=int(value["sample_rate"]),
            start_frequency_hz=float(value["start_frequency_hz"]),
            end_frequency_hz=float(value["end_frequency_hz"]),
            duration_s=float(value["duration_s"]),
            fade_s=float(value["fade_s"]),
            silence_s=float(value["silence_s"]),
            playback_level_dbfs=float(value["playback_level_dbfs"]),
            raw_recordings=tuple(
                MeasurementAsset.from_dict(asset) for asset in value["raw_recordings"]
            ),
            inverse_filter_asset=MeasurementAsset.from_dict(
                value["inverse_filter_asset"]
            ),
            noise_recording_asset=MeasurementAsset.from_dict(
                value["noise_recording_asset"]
            ),
            deconvolved_rir_asset=MeasurementAsset.from_dict(
                value["deconvolved_rir_asset"]
            ),
            latency_correction_samples=int(value["latency_correction_samples"]),
            deconvolution_config=dict(value["deconvolution_config"]),
        )


@dataclass(frozen=True)
class MeasuredRIRRecord:
    """One source pose and one synchronized receiver-array ESS capture."""

    measurement_id: str
    room_id: str
    session_id: str
    timestamp_utc: str
    source_id: str
    source_pose: MeasurementPose
    receiver_ids: tuple[str, ...]
    receiver_poses: tuple[MeasurementPose, ...]
    capture: SweepCapture
    temperature_c: float
    relative_humidity_percent: float
    pressure_pa: float
    synchronized_receivers: bool
    notes: str = ""

    def __post_init__(self) -> None:
        required = (
            self.measurement_id,
            self.room_id,
            self.session_id,
            self.timestamp_utc,
            self.source_id,
        )
        if any(not value for value in required):
            raise ValueError(
                "measurement, room, session, timestamp, and source are required"
            )
        if not self.receiver_ids or len(self.receiver_ids) != len(self.receiver_poses):
            raise ValueError("receiver ids and poses must have equal non-zero length")
        if len(set(self.receiver_ids)) != len(self.receiver_ids):
            raise ValueError("receiver ids cannot be duplicated within one record")
        expected_channels = len(self.receiver_ids)
        capture_assets = (
            *self.capture.raw_recordings,
            self.capture.noise_recording_asset,
            self.capture.deconvolved_rir_asset,
        )
        if any(asset.channel_count != expected_channels for asset in capture_assets):
            raise ValueError("capture channel count must match synchronized receivers")
        if expected_channels > 1 and not self.synchronized_receivers:
            raise ValueError("multi-receiver captures must be sample-synchronized")
        environment = (
            self.temperature_c,
            self.relative_humidity_percent,
            self.pressure_pa,
        )
        if any(not math.isfinite(value) for value in environment):
            raise ValueError("measurement environment must be finite")
        if not -50.0 <= self.temperature_c <= 60.0:
            raise ValueError("temperature_c is outside the supported range")
        if not 0.0 <= self.relative_humidity_percent <= 100.0:
            raise ValueError("relative humidity must be in [0, 100]")
        if self.pressure_pa <= 0.0:
            raise ValueError("pressure_pa must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurement_id": self.measurement_id,
            "room_id": self.room_id,
            "session_id": self.session_id,
            "timestamp_utc": self.timestamp_utc,
            "source_id": self.source_id,
            "source_pose": self.source_pose.to_dict(),
            "receiver_ids": list(self.receiver_ids),
            "receiver_poses": [pose.to_dict() for pose in self.receiver_poses],
            "capture": self.capture.to_dict(),
            "environment": {
                "temperature_c": float(self.temperature_c),
                "relative_humidity_percent": float(self.relative_humidity_percent),
                "pressure_pa": float(self.pressure_pa),
            },
            "synchronized_receivers": bool(self.synchronized_receivers),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MeasuredRIRRecord":
        environment = value["environment"]
        return cls(
            measurement_id=str(value["measurement_id"]),
            room_id=str(value["room_id"]),
            session_id=str(value["session_id"]),
            timestamp_utc=str(value["timestamp_utc"]),
            source_id=str(value["source_id"]),
            source_pose=MeasurementPose.from_dict(value["source_pose"]),
            receiver_ids=tuple(str(item) for item in value["receiver_ids"]),
            receiver_poses=tuple(
                MeasurementPose.from_dict(item) for item in value["receiver_poses"]
            ),
            capture=SweepCapture.from_dict(value["capture"]),
            temperature_c=float(environment["temperature_c"]),
            relative_humidity_percent=float(environment["relative_humidity_percent"]),
            pressure_pa=float(environment["pressure_pa"]),
            synchronized_receivers=bool(value["synchronized_receivers"]),
            notes=str(value.get("notes", "")),
        )


@dataclass(frozen=True)
class RIRMeasurementCampaign:
    """Room-disjoint collection used for physical calibration and holdouts."""

    campaign_id: str
    rooms: tuple[MeasuredRoom, ...]
    transducers: tuple[CalibratedTransducer, ...]
    records: tuple[MeasuredRIRRecord, ...]
    room_splits: Mapping[str, str]
    provenance: Mapping[str, Any]
    schema_version: str = RIR_MEASUREMENT_CAMPAIGN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RIR_MEASUREMENT_CAMPAIGN_SCHEMA_VERSION:
            raise ValueError("unsupported RIR measurement campaign schema")
        if (
            not self.campaign_id
            or not self.rooms
            or not self.transducers
            or not self.records
        ):
            raise ValueError(
                "campaign id, rooms, transducers, and records are required"
            )

        def indexed(name: str, values: Sequence[Any], attribute: str) -> dict[str, Any]:
            result = {str(getattr(value, attribute)): value for value in values}
            if len(result) != len(values):
                raise ValueError(f"campaign {name} ids must be unique")
            return result

        rooms = indexed("room", self.rooms, "room_id")
        transducers = indexed("transducer", self.transducers, "transducer_id")
        indexed("measurement", self.records, "measurement_id")
        splits = {str(key): str(value) for key, value in self.room_splits.items()}
        if set(splits) != set(rooms):
            raise ValueError("room_splits must assign every room exactly once")
        if any(value not in ROOM_SPLITS for value in splits.values()):
            raise ValueError("room split must be train, validation, or test")
        for record in self.records:
            if record.room_id not in rooms:
                raise ValueError("measurement record references an unknown room")
            source = transducers.get(record.source_id)
            if source is None or source.kind != "source":
                raise ValueError("measurement record source is unknown or not a source")
            if any(
                receiver_id not in transducers
                or transducers[receiver_id].kind != "receiver"
                for receiver_id in record.receiver_ids
            ):
                raise ValueError(
                    "measurement record receiver is unknown or not a receiver"
                )
        object.__setattr__(self, "room_splits", splits)
        object.__setattr__(
            self,
            "provenance",
            _strict_json_mapping("campaign provenance", self.provenance),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "rooms": [room.to_dict() for room in self.rooms],
            "transducers": [item.to_dict() for item in self.transducers],
            "records": [record.to_dict() for record in self.records],
            "room_splits": dict(self.room_splits),
            "provenance": dict(self.provenance),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RIRMeasurementCampaign":
        return cls(
            schema_version=str(value["schema_version"]),
            campaign_id=str(value["campaign_id"]),
            rooms=tuple(MeasuredRoom.from_dict(item) for item in value["rooms"]),
            transducers=tuple(
                CalibratedTransducer.from_dict(item) for item in value["transducers"]
            ),
            records=tuple(
                MeasuredRIRRecord.from_dict(item) for item in value["records"]
            ),
            room_splits=dict(value["room_splits"]),
            provenance=dict(value["provenance"]),
        )

    @classmethod
    def from_json(cls, value: str) -> "RIRMeasurementCampaign":
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("campaign JSON root must be an object")
        return cls.from_dict(parsed)

    def assets(self) -> tuple[MeasurementAsset, ...]:
        assets: list[MeasurementAsset] = []
        assets.extend(
            room.mesh_asset for room in self.rooms if room.mesh_asset is not None
        )
        assets.extend(item.calibration_asset for item in self.transducers)
        for record in self.records:
            assets.extend(record.capture.raw_recordings)
            assets.extend(
                (
                    record.capture.inverse_filter_asset,
                    record.capture.noise_recording_asset,
                    record.capture.deconvolved_rir_asset,
                )
            )
        return tuple(assets)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one retained campaign asset."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit_measurement_campaign(
    campaign: RIRMeasurementCampaign,
    root: str | Path,
    *,
    verify_hashes: bool = True,
    minimum_records_per_room: int = 12,
) -> dict[str, Any]:
    """Audit retained files, split semantics, repetitions, and M5 readiness."""

    if minimum_records_per_room < 1:
        raise ValueError("minimum_records_per_room must be positive")
    campaign_root = Path(root)
    unique_assets: dict[str, MeasurementAsset] = {}
    conflicting_paths: list[str] = []
    for asset in campaign.assets():
        previous = unique_assets.get(asset.path)
        if previous is not None and previous.sha256 != asset.sha256:
            conflicting_paths.append(asset.path)
        unique_assets[asset.path] = asset
    missing = [path for path in unique_assets if not (campaign_root / path).is_file()]
    mismatched = []
    if verify_hashes:
        mismatched = [
            path
            for path, asset in unique_assets.items()
            if path not in missing and sha256_file(campaign_root / path) != asset.sha256
        ]
    records_per_room = {
        room.room_id: sum(record.room_id == room.room_id for record in campaign.records)
        for room in campaign.rooms
    }
    unique_configurations_per_room = {
        room.room_id: len(
            {
                (
                    record.source_id,
                    record.source_pose.position_m,
                    record.source_pose.orientation_ypr_deg,
                    record.receiver_ids,
                    tuple(
                        (
                            pose.position_m,
                            pose.orientation_ypr_deg,
                        )
                        for pose in record.receiver_poses
                    ),
                )
                for record in campaign.records
                if record.room_id == room.room_id
            }
        )
        for room in campaign.rooms
    }
    split_room_counts = {
        split: sum(value == split for value in campaign.room_splits.values())
        for split in ROOM_SPLITS
    }
    synchronized_spatial_records = sum(
        len(record.receiver_ids) > 1 and record.synchronized_receivers
        for record in campaign.records
    )
    checks = {
        "asset_paths_have_no_conflicting_hashes": not conflicting_paths,
        "all_assets_exist": not missing,
        "all_retained_asset_hashes_match": not mismatched,
        "train_validation_test_rooms_present": all(
            split_room_counts[split] > 0 for split in ROOM_SPLITS
        ),
        "minimum_records_per_room_met": all(
            count >= minimum_records_per_room for count in records_per_room.values()
        ),
        "minimum_unique_configurations_per_room_met": all(
            count >= minimum_records_per_room
            for count in unique_configurations_per_room.values()
        ),
        "every_capture_has_repeated_raw_sweeps": all(
            len(record.capture.raw_recordings) >= 2 for record in campaign.records
        ),
        "every_capture_retains_noise_and_inverse_filter": all(
            record.capture.noise_recording_asset is not None
            and record.capture.inverse_filter_asset is not None
            for record in campaign.records
        ),
        "at_least_one_synchronized_spatial_record": (synchronized_spatial_records > 0),
    }
    ready = bool(all(checks.values()))
    return {
        "schema_version": "puresound.rir_measurement_campaign_audit.v1",
        "campaign_id": campaign.campaign_id,
        "root": str(campaign_root),
        "room_count": len(campaign.rooms),
        "record_count": len(campaign.records),
        "unique_asset_count": len(unique_assets),
        "records_per_room": records_per_room,
        "unique_configurations_per_room": unique_configurations_per_room,
        "split_room_counts": split_room_counts,
        "synchronized_spatial_record_count": synchronized_spatial_records,
        "minimum_records_per_room": int(minimum_records_per_room),
        "conflicting_asset_paths": sorted(set(conflicting_paths)),
        "missing_asset_paths": sorted(missing),
        "hash_mismatch_paths": sorted(mismatched),
        "checks": checks,
        "ready_for_m5_inverse_calibration": ready,
    }


__all__ = [
    "RIR_MEASUREMENT_CAMPAIGN_SCHEMA_VERSION",
    "ROOM_SPLITS",
    "CalibratedTransducer",
    "MeasuredRIRRecord",
    "MeasuredRoom",
    "MeasurementAsset",
    "MeasurementPose",
    "RIRMeasurementCampaign",
    "SweepCapture",
    "audit_measurement_campaign",
    "sha256_file",
]
