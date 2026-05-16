from dataclasses import dataclass
from typing import Optional

import numpy as np
import rir_generator
import torch


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
    receiver_margin: float = 0.4
    source_margin: float = 0.4
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

    def _distance_range_for_role(self, source_role: str) -> list[float]:
        source_role = source_role.lower()
        if source_role == "foreground" and self.foreground_distance_range is not None:
            return self.foreground_distance_range
        if source_role == "interferer" and self.interferer_distance_range is not None:
            return self.interferer_distance_range
        return self.source_receiver_distance_range

    def _sample_source(
        self, room_dim: np.ndarray, receiver: np.ndarray, source_role: str
    ) -> np.ndarray:
        min_dist, max_dist = self._distance_range_for_role(source_role)
        for _ in range(64):
            source = _sample_point(room_dim, self.source_margin)
            distance = float(np.linalg.norm(source - receiver))
            if float(min_dist) <= distance <= float(max_dist):
                return source
        return _sample_point(room_dim, self.source_margin)

    def generate(
        self,
        sample_rate: int,
        scene: Optional[dict] = None,
        source_role: str = "source",
    ) -> tuple[torch.Tensor, dict]:
        if scene is None:
            scene = self.sample_scene()
        room_dim = np.asarray(scene["room_dim"])
        receiver = np.asarray(scene["receiver"])
        rt60 = float(scene["rt60"])
        source = self._sample_source(room_dim, receiver, source_role)

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
        }
        return rir, metadata
