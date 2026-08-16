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

from puresound.audio.impulse_response import compute_drr_db
from puresound.audio.io import AudioIO
from puresound.audio.rir.bank.schema import BANK_SPLITS, RIRBankManifest
from puresound.audio.rir.bank.production import (
    DEFAULT_PRODUCTION_DECISION_NAME,
    validate_m6_production_certificate,
)
from puresound.audio.rir.bank.release import (
    DEFAULT_RELEASE_MANIFEST_NAME,
    RIRBankReleaseManifest,
    audit_m6_variant_release,
)


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
        split: str | None = None,
        manifest_name: str = "rir_bank_manifest.json",
        include_failed_qc: bool = False,
        allow_legacy_layout: bool = True,
    ):
        self.folder = Path(folder)
        self.near_labels = tuple(near_labels)
        self.far_labels = tuple(far_labels)
        self.drr_window_ms = float(drr_window_ms)
        self.wav_name = str(wav_name)
        self.meta_name = str(meta_name)
        self.cache_size = max(1, int(cache_size))
        self.split = str(split) if split is not None else None
        self.manifest_name = str(manifest_name)
        self.include_failed_qc = bool(include_failed_qc)
        self.allow_legacy_layout = bool(allow_legacy_layout)
        if self.split is not None and self.split not in BANK_SPLITS:
            raise ValueError(f"split must be one of {BANK_SPLITS}")
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
        manifest_path = self.folder / self.manifest_name
        if manifest_path.is_file():
            if self.split is None:
                raise ValueError(
                    "M6 multi-split bank requires an explicit split= value; "
                    "refusing to mix train, validation, and test"
                )
            manifest = RIRBankManifest.from_json(
                manifest_path.read_text(encoding="utf-8")
            )
            if manifest.manifest_sha256 != manifest.content_sha256():
                raise ValueError("M6 bank manifest content hash does not match")
            rooms = []
            for item in manifest.items:
                if item.split != self.split:
                    continue
                if item.qc_status == "fail" and not self.include_failed_qc:
                    continue
                if (
                    manifest.release_status in {"candidate", "production"}
                    and item.qc_status == "pending"
                ):
                    continue
                room = self._index_room_item(
                    item.item_id,
                    self.folder / item.rir_path,
                    self.folder / item.metadata_path,
                )
                if room is None:
                    raise ValueError(
                        f"manifest item {item.item_id!r} has unreadable or "
                        "invalid RIR metadata"
                    )
                room.update(
                    {
                        "physical_room_id": item.room_id,
                        "acoustic_space_id": item.acoustic_space_id,
                        "split": item.split,
                        "renderer_profile_id": item.renderer_profile_id,
                        "qc_status": item.qc_status,
                        "qc_report_path": item.qc_report_path,
                        "qc_report_sha256": item.qc_report_sha256,
                    }
                )
                rooms.append(room)
            return rooms
        if self.split is not None:
            raise ValueError("split= requires an M6 rir_bank_manifest.json")
        if not self.allow_legacy_layout:
            raise ValueError(
                "legacy room-bank layout is disabled and no manifest was found"
            )
        if self._manifestless_m6_metadata_detected():
            raise ValueError(
                "M6 item metadata was found but the bank manifest is missing; "
                "refusing legacy fallback because it would mix dataset splits"
            )
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

    def _manifestless_m6_metadata_detected(self) -> bool:
        """Detect a copied/incompletely configured M6 bank before legacy scan."""

        if all(
            (self.folder / "indexes" / f"{split}.jsonl").is_file()
            for split in BANK_SPLITS
        ):
            return True
        inspected = 0
        for meta_path in sorted(self.folder.glob("*/*.json")):
            if meta_path.name == self.meta_name:
                continue
            inspected += 1
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(metadata, dict) and (
                isinstance(metadata.get("m6"), dict)
                or isinstance(metadata.get("m6_variant"), dict)
            ):
                return True
            if inspected >= 32:
                break
        return False

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
            "split": room.get("split"),
            "physical_room_id": room.get("physical_room_id"),
            "acoustic_space_id": room.get("acoustic_space_id"),
            "renderer_profile_id": room.get("renderer_profile_id"),
            "qc_status": room.get("qc_status"),
            "qc_report_path": room.get("qc_report_path"),
            "qc_report_sha256": room.get("qc_report_sha256"),
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
        if not 0 <= idx < rir.shape[0]:
            raise IndexError(
                f"RIR channel index {idx} is outside [0, {rir.shape[0]}) "
                f"for room {scene['room_id']!r}"
            )
        impulse = rir[idx : idx + 1].clone()
        metadata = {
            "source_receiver_distance": float(chosen["distance"]),
            "drr_db": compute_drr_db(impulse, sr, self.drr_window_ms),
            "rt60": scene["rt60"],
            "label": chosen["label"],
            "room_id": scene["room_id"],
            "origin": scene.get("origin"),
            "split": scene.get("split"),
            "physical_room_id": scene.get("physical_room_id"),
            "acoustic_space_id": scene.get("acoustic_space_id"),
            "renderer_profile_id": scene.get("renderer_profile_id"),
            "qc_status": scene.get("qc_status"),
            "qc_report_path": scene.get("qc_report_path"),
            "qc_report_sha256": scene.get("qc_report_sha256"),
        }
        return impulse, metadata, sr

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


class PreGeneratedReleaseBank:
    """Serve one ready M6.4 recipe without mixing dataset splits.

    The recipe first samples an origin according to its frozen weights, then a
    release variant within that origin, and finally an item from that variant.
    This makes future synthetic/real mixtures use recipe semantics instead of
    silently sampling in proportion to the number of files on disk.
    """

    def __init__(
        self,
        folder: str,
        *,
        recipe_id: str,
        split: str,
        release_manifest_name: str = DEFAULT_RELEASE_MANIFEST_NAME,
        require_production: bool = False,
        production_decision_name: str = DEFAULT_PRODUCTION_DECISION_NAME,
        audit: bool = True,
        audit_cache: bool = True,
        **bank_kwargs,
    ):
        self.folder = Path(folder)
        self.recipe_id = str(recipe_id)
        self.split = str(split)
        if self.split not in BANK_SPLITS:
            raise ValueError(f"split must be one of {BANK_SPLITS}")
        # A full audit re-derives every child RIR from its parent's samples, so it
        # reads the whole release. It is memoized beside the release (see
        # audit_m6_variant_release); `audit=False` skips it outright for callers
        # that already validated this release -- e.g. a training run whose two DDP
        # ranks would otherwise each repeat it inside the rendezvous timeout.
        if audit:
            verdict = audit_m6_variant_release(
                self.folder,
                manifest_name=release_manifest_name,
                use_cache=audit_cache,
            )
            if not verdict["valid"]:
                raise ValueError("M6.4 release audit failed")
        release = RIRBankReleaseManifest.from_json(
            (self.folder / release_manifest_name).read_text(encoding="utf-8")
        )
        self._production_certificate_sha256 = None
        if require_production:
            certificate = validate_m6_production_certificate(
                self.folder / production_decision_name,
                release_root=self.folder,
                require_approved=True,
            )
            if not certificate["valid"]:
                raise ValueError(
                    "M6.6 production certificate is missing, invalid, or blocked"
                )
            self._production_certificate_sha256 = certificate["decision_sha256"]
        recipe = next(
            (item for item in release.recipes if item.recipe_id == self.recipe_id),
            None,
        )
        if recipe is None:
            raise ValueError(f"unknown M6.4 release recipe: {self.recipe_id}")
        if recipe.status != "ready":
            reason = "; ".join(recipe.blockers)
            raise ValueError(f"M6.4 release recipe {self.recipe_id!r} is blocked: {reason}")
        variants = {variant.variant_id: variant for variant in release.variants}
        self._banks = {
            variant_id: PreGeneratedRoomBank(
                str(self.folder / variants[variant_id].bank_path),
                split=self.split,
                **bank_kwargs,
            )
            for variant_id in recipe.variant_ids
        }
        self._variant_origins = {
            variant_id: variants[variant_id].origin for variant_id in recipe.variant_ids
        }
        self._origin_weights = dict(recipe.origin_weights)
        self._release_id = release.release_id
        self._release_sha256 = release.release_sha256
        self._item_count = recipe.split_indexes[self.split].item_count
        if sum(len(bank) for bank in self._banks.values()) != self._item_count:
            raise ValueError("M6.4 recipe index count does not match its variants")

    def __len__(self) -> int:
        return self._item_count

    def sample_scene(self) -> dict:
        """Sample a scene according to the recipe's origin weights."""

        origins = tuple(sorted(self._origin_weights))
        origin = random.choices(
            origins,
            weights=[self._origin_weights[value] for value in origins],
            k=1,
        )[0]
        candidates = [
            variant_id
            for variant_id, variant_origin in self._variant_origins.items()
            if variant_origin == origin
        ]
        if not candidates:
            raise ValueError(f"recipe origin {origin!r} has no release variant")
        variant_id = candidates[random.randrange(len(candidates))]
        scene = self._banks[variant_id].sample_scene()
        scene.update(
            {
                "_release_bank": True,
                "release_id": self._release_id,
                "release_sha256": self._release_sha256,
                "release_recipe_id": self.recipe_id,
                "release_variant_id": variant_id,
                "release_origin": origin,
                "production_certificate_sha256": (
                    self._production_certificate_sha256
                ),
            }
        )
        return scene

    def select_channel(
        self,
        scene: dict,
        source_role: str = "source",
        distance_range_override: Optional[list[float]] = None,
    ) -> tuple[torch.Tensor, dict, int]:
        """Select a channel and propagate release identity into metadata."""

        variant_id = str(scene.get("release_variant_id", ""))
        if variant_id not in self._banks:
            raise ValueError("scene does not belong to this M6.4 release recipe")
        impulse, metadata, sample_rate = self._banks[variant_id].select_channel(
            scene,
            source_role=source_role,
            distance_range_override=distance_range_override,
        )
        metadata.update(
            {
                "release_id": self._release_id,
                "release_sha256": self._release_sha256,
                "release_recipe_id": self.recipe_id,
                "release_variant_id": variant_id,
                "release_origin": scene["release_origin"],
                "production_certificate_sha256": (
                    self._production_certificate_sha256
                ),
            }
        )
        return impulse, metadata, sample_rate


class UnionRoomBank:
    """Serve scenes from several banks at fixed per-bank probabilities.

    Use this to widen a recipe's room pool without replacing what already
    works: each member keeps its own layout, labels, DRR window and wav cache,
    and every scene is routed back to the bank that produced it. Weights are
    sampling probabilities, not item counts -- a member holding a tenth of the
    rooms still gets its share of the draws.
    """

    def __init__(self, members: list[dict]):
        if not members:
            raise ValueError("union room bank requires at least one member bank")
        self._names = [str(member["name"]) for member in members]
        self._banks = [member["bank"] for member in members]
        weights = [float(member.get("weight", 1.0)) for member in members]
        if any(weight <= 0 for weight in weights):
            raise ValueError("union room bank weights must be positive")
        total = sum(weights)
        self.weights = [weight / total for weight in weights]

    def __len__(self) -> int:
        return sum(len(bank) for bank in self._banks)

    def describe(self) -> str:
        return ", ".join(
            f"{name} w={weight:.2f} items={len(bank)}"
            for name, weight, bank in zip(self._names, self.weights, self._banks)
        )

    def sample_scene(self) -> dict:
        index = random.choices(range(len(self._banks)), weights=self.weights, k=1)[0]
        scene = self._banks[index].sample_scene()
        scene["_union_member"] = index
        scene["union_member_name"] = self._names[index]
        return scene

    def select_channel(
        self,
        scene: dict,
        source_role: str = "source",
        distance_range_override: Optional[list[float]] = None,
    ) -> tuple[torch.Tensor, dict, int]:
        index = scene.get("_union_member")
        if not isinstance(index, int) or not 0 <= index < len(self._banks):
            raise ValueError("scene does not belong to this union room bank")
        impulse, metadata, sample_rate = self._banks[index].select_channel(
            scene,
            source_role=source_role,
            distance_range_override=distance_range_override,
        )
        metadata["union_member_name"] = self._names[index]
        return impulse, metadata, sample_rate
