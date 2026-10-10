"""Versioned moving-source scenes; independent of rendering and training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any

import numpy as np

from puresound.audio.rir.scene.schema import RoomSceneV2
from puresound.audio.rir.path_events.geometry import segment_intersects_scene_object

DYNAMIC_SCENE_VERSION = "puresound.dynamic_scene.v2"
_V1 = "puresound.dynamic_scene.v1"
# A source's part in evaluation: a talker to keep, another talker, or noise.
ROLES = ("target", "interferer", "noise")
TALKER_ROLES = ("target", "interferer")
# The training pipeline's target_rir_type choices, with its definitions:
# direct sound plus 50 ms (early) or 6 ms (direct), the whole response, or dry.
REFERENCE_RIR_TYPES = ("early", "direct", "full", "anechoic")
MAX_DURATION_S = 30.0
MAX_SPEED_M_S = 2.0
MIN_MIC_DISTANCE_M = 0.2
MAX_KEYFRAMES = 64
NEAR_RADIUS_RANGE_M = (0.2, 10.0)
GAIN_RANGE_DB = (-60.0, 24.0)
MAX_REFLECTION_ORDER = 2
MAX_SOURCES = 8


def scene_limits() -> dict[str, Any]:
    """The limits a scene is validated against, for editors to show."""
    return {
        "duration_s": MAX_DURATION_S,
        "speed_m_s": MAX_SPEED_M_S,
        "min_mic_distance_m": MIN_MIC_DISTANCE_M,
        "keyframes": MAX_KEYFRAMES,
        "near_radius_m": list(NEAR_RADIUS_RANGE_M),
        "gain_db": list(GAIN_RANGE_DB),
        "reflection_order": MAX_REFLECTION_ORDER,
        "sources": MAX_SOURCES,
        "roles": list(ROLES),
        "reference_rir": list(REFERENCE_RIR_TYPES),
    }


@dataclass(frozen=True)
class MotionKeyframe:
    time_s: float
    position_m: tuple[float, float, float]
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0

    def __post_init__(self):
        if (
            len(self.position_m) != 3
            or not np.isfinite(
                [self.time_s, *self.position_m, self.yaw_deg, self.pitch_deg]
            ).all()
        ):
            raise ValueError("keyframes require finite time, xyz and angles")


@dataclass(frozen=True)
class DynamicSource:
    source_id: str
    asset_id: str
    keyframes: tuple[MotionKeyframe, ...]
    role: str = "target"
    gain_db: float = 0.0
    start_s: float = 0.0
    repeat: bool = False

    def pose_at(self, times):
        knots = np.array([k.time_s for k in self.keyframes])
        poses = np.array(
            [(*k.position_m, k.yaw_deg, k.pitch_deg) for k in self.keyframes]
        )
        poses[:, 3:] = np.rad2deg(np.unwrap(np.deg2rad(poses[:, 3:]), axis=0))
        return np.stack(
            [np.interp(times, knots, poses[:, i]) for i in range(5)], axis=-1
        )


@dataclass(frozen=True)
class DynamicSceneSpec:
    room: RoomSceneV2
    sources: tuple[DynamicSource, ...]
    duration_s: float = 10.0
    seed: int = 0
    near_radius_m: float = 1.0
    reference_rir: str = "early"
    sample_rate: int = 16000
    max_order: int = 2
    late_reverb: bool = True
    schema_version: str = DYNAMIC_SCENE_VERSION

    def __post_init__(self):
        if self.schema_version != DYNAMIC_SCENE_VERSION:
            raise ValueError("unsupported dynamic scene version")
        if not math.isfinite(self.duration_s) or not 0 < self.duration_s <= MAX_DURATION_S:
            raise ValueError(f"duration must be in (0, {MAX_DURATION_S:g}] seconds")
        if (
            self.sample_rate != 16000
            or type(self.max_order) is not int
            or not 0 <= self.max_order <= MAX_REFLECTION_ORDER
        ):
            raise ValueError(
                f"dynamic scenes require 16 kHz and reflection order 0..{MAX_REFLECTION_ORDER}"
            )
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if type(self.late_reverb) is not bool:
            raise ValueError("late_reverb must be boolean")
        low, high = NEAR_RADIUS_RANGE_M
        if not math.isfinite(self.near_radius_m) or not low <= self.near_radius_m <= high:
            raise ValueError(f"near radius must be in [{low:g}, {high:g}] metres")
        if self.reference_rir not in REFERENCE_RIR_TYPES:
            raise ValueError(f"reference_rir must be one of {', '.join(REFERENCE_RIR_TYPES)}")
        if len(self.room.receivers) != 1:
            raise ValueError("a scene has one fixed receiver")
        if not 1 <= len(self.sources) <= MAX_SOURCES:
            raise ValueError(f"a scene holds 1 to {MAX_SOURCES} sources")
        ids = [s.source_id for s in self.sources]
        if any(
            not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value)
            for s in self.sources
            for value in (s.source_id, s.asset_id)
        ):
            raise ValueError(
                "source and asset IDs require 1..64 letters, digits, underscores or hyphens"
            )
        mic_position = np.array(self.room.receivers[0].pose.position_m)
        if np.any(mic_position <= 0) or np.any(
            mic_position >= np.array(self.room.dimensions_m)
        ):
            raise ValueError("microphone must be inside room walls")
        if len(set(ids)) != len(ids) or set(ids) != {
            s.transducer_id for s in self.room.sources
        }:
            raise ValueError("dynamic and room source IDs must match uniquely")
        mic = np.array(self.room.receivers[0].pose.position_m)
        dims = np.array(self.room.dimensions_m)
        if any(obj.contains_point(mic) for obj in self.room.objects):
            raise ValueError("microphone is inside an obstacle")
        for source in self.sources:
            if type(source.repeat) is not bool:
                raise ValueError("source repeat must be boolean")
            if source.role not in ROLES:
                raise ValueError(f"source role must be one of {', '.join(ROLES)}")
            if (
                not np.isfinite([source.gain_db, source.start_s]).all()
                or not GAIN_RANGE_DB[0] <= source.gain_db <= GAIN_RANGE_DB[1]
                or not 0 <= source.start_s < self.duration_s
            ):
                raise ValueError("source gain/start is out of range")
            if not 1 <= len(source.keyframes) <= MAX_KEYFRAMES:
                raise ValueError(f"each source needs 1..{MAX_KEYFRAMES} keyframes")
            knots = source.keyframes
            if knots[0].time_s != 0 or any(
                k.time_s < 0 or k.time_s > self.duration_s for k in knots
            ):
                raise ValueError(
                    "keyframes must start at zero and lie inside the duration"
                )
            if any(b.time_s <= a.time_s for a, b in zip(knots, knots[1:])):
                raise ValueError("keyframe times must increase strictly")
            for k in knots:
                point = np.array(k.position_m)
                if np.any(point <= 0) or np.any(point >= dims):
                    raise ValueError("source position must be inside room walls")
                if np.linalg.norm(point - mic) < MIN_MIC_DISTANCE_M - 1e-9:
                    raise ValueError(
                        f"source must stay at least {MIN_MIC_DISTANCE_M:g} m from microphone"
                    )
                if any(obj.contains_point(point) for obj in self.room.objects):
                    raise ValueError("source is inside an obstacle")
            for a, b in zip(knots, knots[1:]):
                p, q = np.array(a.position_m), np.array(b.position_m)
                v = q - p
                if np.linalg.norm(v) / (b.time_s - a.time_s) > MAX_SPEED_M_S + 1e-9:
                    raise ValueError(f"source speed exceeds {MAX_SPEED_M_S:g} m/s")
                u = np.clip(np.dot(mic - p, v) / max(np.dot(v, v), 1e-20), 0, 1)
                if np.linalg.norm(p + u * v - mic) < MIN_MIC_DISTANCE_M - 1e-9:
                    raise ValueError("trajectory passes too close to microphone")
                if np.linalg.norm(v) > 1e-10 and any(
                    segment_intersects_scene_object(p, q, obj)
                    for obj in self.room.objects
                ):
                    raise ValueError("trajectory crosses an obstacle")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "room": self.room.to_dict(),
            "sources": [asdict(s) for s in self.sources],
            "duration_s": self.duration_s,
            "seed": self.seed,
            "near_radius_m": self.near_radius_m,
            "reference_rir": self.reference_rir,
            "sample_rate": self.sample_rate,
            "max_order": self.max_order,
            "late_reverb": self.late_reverb,
        }

    @classmethod
    def from_dict(cls, data):
        """Read a scene; a v1 scene is converted to roles on the way in."""
        if data.get("schema_version") == _V1:
            data = _from_v1(data)
        sources = []
        for s in data["sources"]:
            knots = tuple(
                MotionKeyframe(**{**k, "position_m": tuple(k["position_m"])})
                for k in s["keyframes"]
            )
            sources.append(DynamicSource(**{**s, "keyframes": knots}))
        return cls(
            **{
                **data,
                "room": RoomSceneV2.from_dict(data["room"]),
                "sources": tuple(sources),
            }
        )


def _from_v1(data):
    """v1 named one fixed talker; it becomes the target, other talkers become
    interferers, and the reference keeps v1's early definition."""
    fixed = data.get("fixed_source_id")
    sources = [
        {
            **source,
            "role": "noise"
            if source.get("role") == "noise"
            else "target"
            if source["source_id"] == fixed
            else "interferer",
        }
        for source in data["sources"]
    ]
    converted = {k: v for k, v in data.items() if k != "fixed_source_id"}
    return {
        **converted,
        "sources": sources,
        "reference_rir": "early",
        "schema_version": DYNAMIC_SCENE_VERSION,
    }
