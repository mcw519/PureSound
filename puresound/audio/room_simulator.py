from dataclasses import dataclass
from typing import Optional

import numpy as np
import rir_generator
import torch

from puresound.audio.impluse_response import compute_drr_db


def _sample_range(bounds: list[float]) -> float:
    if len(bounds) != 2:
        raise ValueError(f"Expected [min, max], got {bounds}")
    return float(np.random.uniform(float(bounds[0]), float(bounds[1])))


def _sample_room_dim(room_dim_range: list[list[float]]) -> np.ndarray:
    if len(room_dim_range) != 3:
        raise ValueError("room_dim_range must have x/y/z ranges")
    return np.asarray([_sample_range(axis_range) for axis_range in room_dim_range])


def _sample_point(room_dim: np.ndarray, margin: float) -> np.ndarray:
    margin = float(margin)
    lower = np.full(3, margin)
    upper = np.maximum(room_dim - margin, lower + 0.01)
    return np.random.uniform(lower, upper)


@dataclass
class RoomImpulseResponseSimulator:
    room_dim_range: list[list[float]]
    rt60_range: list[float]
    source_receiver_distance_range: list[float]
    foreground_distance_range: Optional[list[float]] = None
    interferer_distance_range: Optional[list[float]] = None
    media_distance_range: Optional[list[float]] = None
    receiver_margin: float = 0.4
    source_margin: float = 0.4
    # Media sources (TV / loudspeaker) sit close to a wall, which strengthens
    # early reflections relative to a free-standing talker at the same distance.
    media_wall_offset_max: float = 0.2
    sound_speed: float = 343.0
    nsample: Optional[int] = None
    order: int = -1
    hp_filter: bool = True

    def sample_scene(self) -> dict:
        room_dim = _sample_room_dim(self.room_dim_range)
        receiver = _sample_point(room_dim, self.receiver_margin)
        rt60 = _sample_range(self.rt60_range)
        return {
            "room_dim": room_dim,
            "receiver": receiver,
            "rt60": rt60,
        }

    def _distance_range_for_role(
        self,
        source_role: str,
        override: Optional[list[float]] = None,
    ) -> list[float]:
        if override is not None:
            return override
        source_role = source_role.lower()
        if source_role == "foreground" and self.foreground_distance_range is not None:
            return self.foreground_distance_range
        if source_role == "media":
            if self.media_distance_range is not None:
                return self.media_distance_range
            if self.interferer_distance_range is not None:
                return self.interferer_distance_range
        if source_role == "interferer" and self.interferer_distance_range is not None:
            return self.interferer_distance_range
        return self.source_receiver_distance_range

    def _sample_source(
        self,
        room_dim: np.ndarray,
        receiver: np.ndarray,
        source_role: str,
        distance_range_override: Optional[list[float]] = None,
    ) -> np.ndarray:
        min_dist, max_dist = self._distance_range_for_role(
            source_role, distance_range_override
        )
        wall_bias = source_role.lower() == "media"
        for _ in range(64):
            source = _sample_point(room_dim, self.source_margin)
            if wall_bias:
                axis = int(np.random.randint(0, 2))  # x or y wall
                offset = self.source_margin + float(
                    np.random.uniform(0.0, self.media_wall_offset_max)
                )
                if np.random.randint(0, 2) == 0:
                    source[axis] = offset
                else:
                    source[axis] = float(room_dim[axis]) - offset
            distance = float(np.linalg.norm(source - receiver))
            if float(min_dist) <= distance <= float(max_dist):
                return source
        # Thin shells (e.g. [0.3, d] for a small query distance d) make
        # uniform-in-room rejection fail often; an unconstrained fallback here
        # would silently violate the distance range and mislabel the sample,
        # so sample the shell directly instead. Media loses its wall snap in
        # this path: the distance semantics outrank the wall placement.
        return self._sample_source_in_shell(
            room_dim, receiver, float(min_dist), float(max_dist)
        )

    def _sample_source_in_shell(
        self,
        room_dim: np.ndarray,
        receiver: np.ndarray,
        min_dist: float,
        max_dist: float,
    ) -> np.ndarray:
        """Sample a point at distance [min_dist, max_dist] from the receiver:
        uniform radius + uniform direction, rejected only on the room margins.
        When the shell has no practical intersection with the room (the range
        is geometrically unsatisfiable), walk toward the farthest in-room
        corner and clamp the radius to the available span -- the closest
        achievable distance rather than an arbitrary one."""
        lower = np.full(3, float(self.source_margin))
        upper = np.maximum(room_dim - float(self.source_margin), lower + 0.01)
        for _ in range(256):
            radius = float(np.random.uniform(min_dist, max_dist))
            direction = np.random.normal(size=3)
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9:
                continue
            source = receiver + direction / norm * radius
            if np.all(source >= lower) and np.all(source <= upper):
                return source
        corners = np.array(
            [
                [x, y, z]
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            ]
        )
        spans = np.linalg.norm(corners - receiver, axis=1)
        far_corner = corners[int(np.argmax(spans))]
        span = float(spans.max())
        if span < 1e-9:
            return far_corner
        radius = float(np.clip(np.random.uniform(min_dist, max_dist), 0.0, span))
        return receiver + (far_corner - receiver) / span * radius

    def generate(
        self,
        sample_rate: int,
        scene: Optional[dict] = None,
        source_role: str = "source",
        distance_range_override: Optional[list[float]] = None,
    ) -> tuple[torch.Tensor, dict]:
        if scene is None:
            scene = self.sample_scene()
        room_dim = np.asarray(scene["room_dim"])
        receiver = np.asarray(scene["receiver"])
        rt60 = float(scene["rt60"])
        source = self._sample_source(
            room_dim, receiver, source_role, distance_range_override
        )

        rir = rir_generator.generate(
            c=float(self.sound_speed),
            fs=int(sample_rate),
            r=receiver,
            s=source,
            L=room_dim,
            reverberation_time=rt60,
            nsample=self.nsample,
            order=int(self.order),
            hp_filter=bool(self.hp_filter),
        )
        rir = torch.as_tensor(rir.T, dtype=torch.float32)
        metadata = {
            "room_dim": room_dim.tolist(),
            "receiver": receiver.tolist(),
            "source": source.tolist(),
            "rt60": rt60,
            "source_role": source_role,
            "source_receiver_distance": float(np.linalg.norm(source - receiver)),
            "drr_db": compute_drr_db(rir, int(sample_rate)),
        }
        return rir, metadata
