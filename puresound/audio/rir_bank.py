"""Pre-generated multi-source room RIR bank.

Loads a folder of pre-rendered room items (e.g. the hybrid wave/geometric RIR
dataset produced by ``egs/rir_generation/generate_hybrid_rir.py``) and serves
per-source RIR channels to the augmentation pipeline.

Each room directory can either contain the legacy single ``rir_5ch.wav`` +
``metadata.json`` pair or multiple same-stem RIR/metadata pairs such as
``room_000000_000001.wav`` + ``room_000000_000001.json``. The near channels
feed the foreground talker; the far channels feed interferers. This mirrors the
on-the-fly ``RoomImpulseResponseSimulator`` interface (``sample_scene`` +
per-source selection) so the dataset code path does not change.
"""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch

from puresound.audio.impluse_response import compute_drr_db
from puresound.audio.io import AudioIO


class PreGeneratedRoomBank:
    """Serve RIR channels from a folder of pre-rendered multi-source rooms."""

    def __init__(
        self,
        folder: str,
        near_labels=("near_0", "near_1"),
        far_labels=("far_0", "far_1", "far_2"),
        drr_window_ms: float = 2.5,
        wav_name: str = "rir_5ch.wav",
        meta_name: str = "metadata.json",
        cache_size: int = 64,
    ):
        self.folder = Path(folder)
        self.near_labels = tuple(near_labels)
        self.far_labels = tuple(far_labels)
        self.drr_window_ms = float(drr_window_ms)
        self.wav_name = str(wav_name)
        self.meta_name = str(meta_name)
        self.cache_size = max(1, int(cache_size))
        self._rooms = self._index_rooms()
        if not self._rooms:
            raise FileNotFoundError(
                f"No room items with '{self.wav_name}' + '{self.meta_name}' were "
                f"found under {self.folder}."
            )
        self._wav_cache: "OrderedDict[str, tuple[torch.Tensor, int]]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._rooms)

    def _index_rooms(self) -> list[dict]:
        rooms: list[dict] = []
        for sub in sorted(p for p in self.folder.iterdir() if p.is_dir()):
            wav_path = sub / self.wav_name
            meta_path = sub / self.meta_name
            if wav_path.exists() and meta_path.exists():
                room = self._index_room_item(sub.name, wav_path, meta_path)
                if room is not None:
                    rooms.append(room)
            for wav_path in sorted(sub.glob("*.wav")):
                meta_path = wav_path.with_suffix(".json")
                if wav_path.name == self.wav_name or not meta_path.exists():
                    continue
                room = self._index_room_item(wav_path.stem, wav_path, meta_path)
                if room is not None:
                    rooms.append(room)
        return rooms

    def _index_room_item(
        self, item_id: str, wav_path: Path, meta_path: Path
    ) -> Optional[dict]:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        scene = meta.get("scene", {})
        channel_map = scene.get("channel_map") or []
        channels: list[dict] = []
        for entry in channel_map:
            distance = entry.get("distance_m")
            if distance is None:
                continue
            channels.append(
                {
                    "channel": int(entry["channel"]),
                    "label": str(entry.get("label", "")),
                    "distance": float(distance),
                }
            )
        if not channels:
            return None
        rt60 = scene.get("rt60")
        origin = scene.get("origin")
        return {
            "id": str(item_id),
            "wav_path": str(wav_path),
            "rt60": float(rt60) if rt60 is not None else None,
            # e.g. "real" for measured-RIR banks (real_rir_to_bank.py); None for
            # synthetic banks whose scene json predates this field.
            "origin": str(origin) if origin is not None else None,
            "channels": channels,
        }

    def _split_pools(self, channels: list[dict]) -> tuple[list[dict], list[dict]]:
        near = [c for c in channels if c["label"] in self.near_labels]
        far = [c for c in channels if c["label"] in self.far_labels]
        if near or far:
            return near, far
        # No recognized labels: split by distance around the median so the bank
        # still works on folders that do not follow the near/far naming.
        ordered = sorted(channels, key=lambda c: c["distance"])
        mid = len(ordered) // 2
        return ordered[: max(1, mid)], ordered[max(1, mid):]

    def sample_scene(self) -> dict:
        """Pick a random room and return a per-sample scene handle."""
        room = self._rooms[random.randrange(len(self._rooms))]
        near, far = self._split_pools(room["channels"])
        return {
            "_bank": True,
            "room_id": room["id"],
            "wav_path": room["wav_path"],
            "rt60": room["rt60"],
            "origin": room.get("origin"),
            "near": near,
            "far": far,
            "all": list(room["channels"]),
            "used_channels": set(),
        }

    def select_channel(
        self,
        scene: dict,
        source_role: str = "source",
        distance_range_override: Optional[list[float]] = None,
    ) -> tuple[torch.Tensor, dict, int]:
        """Return ``(impulse[1, L], metadata, sample_rate)`` for one source.

        ``foreground`` draws from the near pool, ``interferer``/``media``/``echo``
        from the far pool, and any other role from all channels.
        """
        role = (source_role or "").lower()
        if role == "foreground":
            pool = scene["near"] or scene["far"] or scene["all"]
        elif role in ("interferer", "media", "echo"):
            pool = scene["far"] or scene["near"] or scene["all"]
        else:
            pool = scene["all"]

        chosen = self._pick(pool, distance_range_override, scene["used_channels"])
        scene["used_channels"].add(chosen["channel"])

        rir, sr = self._load(scene["room_id"], scene["wav_path"])
        idx = int(chosen["channel"])
        idx = min(max(idx, 0), rir.shape[0] - 1)
        impulse = rir[idx : idx + 1].clone()
        metadata = {
            "source_receiver_distance": float(chosen["distance"]),
            "drr_db": compute_drr_db(impulse, sr, self.drr_window_ms),
            "rt60": scene["rt60"],
            "label": chosen["label"],
            "room_id": scene["room_id"],
            "origin": scene.get("origin"),
        }
        return impulse, metadata, sr

    def suggest_query_distance(
        self,
        scene: dict,
        foreground_distance: float,
        near_floor: float = 0.3,
        far_ceiling: float = 5.0,
        margin: float = 0.2,
        peak_distance: Optional[float] = None,
        peak_prob: float = 0.0,
        peak_half_width: float = 0.2,
    ) -> float:
        """Pick a query distance that covers the (fixed) foreground channel.

        The query is sampled in ``[foreground_distance + margin, nearest far
        channel - margin]`` so the near source always falls inside the query
        while every far interferer in the room stays outside it. Because the
        bank's distances are fixed at generation time, deriving the query from
        the realized placement keeps the decision boundary consistent with the
        RIR (unlike sampling the query independently from config).
        """
        fg = float(foreground_distance)
        far_distances = [c["distance"] for c in scene.get("far", [])]
        upper = (min(far_distances) - margin) if far_distances else float(far_ceiling)
        upper = min(upper, float(far_ceiling))
        lower = max(fg + margin, float(near_floor))
        if not (upper > lower):
            # Fixed channels are too close to fit a separating boundary; sit just
            # above the foreground and clamp into the room's valid span.
            return float(min(max(fg + 0.5 * margin, float(near_floor)), float(far_ceiling)))
        if peak_distance is not None and float(torch.rand(1).item()) < float(peak_prob):
            peak = float(peak_distance)
            p_lo = max(lower, peak - float(peak_half_width))
            p_hi = min(upper, peak + float(peak_half_width))
            if p_hi > p_lo:
                lower, upper = p_lo, p_hi
        return float(torch.empty(1).uniform_(lower, upper).item())

    @staticmethod
    def _pick(
        pool: list[dict],
        distance_range_override: Optional[list[float]],
        used: set,
    ) -> dict:
        candidates = [c for c in pool if c["channel"] not in used] or list(pool)
        if distance_range_override is not None:
            lo, hi = float(distance_range_override[0]), float(distance_range_override[1])
            in_range = [c for c in candidates if lo <= c["distance"] <= hi]
            if in_range:
                candidates = in_range
            else:
                # No channel inside the requested band: take the closest one to
                # the band's centre so distance gating degrades gracefully.
                mid = 0.5 * (lo + hi)
                candidates = [min(candidates, key=lambda c: abs(c["distance"] - mid))]
        return candidates[random.randrange(len(candidates))]

    def _load(self, room_id: str, wav_path: str) -> tuple[torch.Tensor, int]:
        cached = self._wav_cache.get(room_id)
        if cached is not None:
            self._wav_cache.move_to_end(room_id)
            return cached
        wav, sr = AudioIO.open(wav_path)
        wav = wav.float()
        if wav.dim() == 1:
            wav = wav.view(1, -1)
        result = (wav, int(sr))
        self._wav_cache[room_id] = result
        if len(self._wav_cache) > self.cache_size:
            self._wav_cache.popitem(last=False)
        return result
