"""Versioned, content-addressed production RIR bank contract for M6.

The legacy :class:`puresound.audio.rir.bank.loader.PreGeneratedRoomBank` intentionally
continues to consume same-stem WAV/JSON pairs.  This module adds the release
layer around those pairs: deterministic room-disjoint splits, renderer and
generator provenance, content hashes, and fail-closed production claims.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import soundfile as sf


RIR_BANK_MANIFEST_SCHEMA_VERSION = "puresound.rir_bank.v2"
RIR_BANK_AUDIT_SCHEMA_VERSION = "puresound.rir_bank_audit.v1"
M6_SPLIT_POLICY = "puresound.m6_split.sha256_acoustic_space.v1"
BANK_SPLITS = ("train", "validation", "test")
RELEASE_STATUSES = ("draft", "candidate", "production")
EVIDENCE_TIERS = ("development", "empirical_candidate", "production_approved")
ORIGINS = ("synthetic", "real", "mixed")
SIGNAL_VARIANTS = ("physical", "physical_residual", "measured")
LEVEL_POLICIES = ("calibrated", "peak_normalized", "native_measured")
QC_STATUSES = ("pending", "pass", "fail")


def _required_text(name: str, value: str) -> str:
    result = str(value)
    if not result or not result.strip():
        raise ValueError(f"{name} is required")
    return result


def _sha256(name: str, value: str | None, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    digest = str(value or "").lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must contain 64 hexadecimal characters")
    return digest


def _relative_path(name: str, value: str) -> str:
    path = PurePosixPath(str(value))
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a safe bank-relative path")
    return path.as_posix()


def _safe_path_component(name: str, value: str) -> str:
    result = _required_text(name, value)
    path = PurePosixPath(result)
    if (
        len(path.parts) != 1
        or result in {".", ".."}
        or "/" in result
        or "\\" in result
        or "\0" in result
    ):
        raise ValueError(f"{name} must be a safe single path component")
    return result


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize strict JSON deterministically for content addressing."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Return the SHA-256 of canonical strict JSON."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 for one bank asset."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_float_wav_header(path: str | Path) -> None:
    """Zero libsndfile's wall-clock PEAK timestamp in a RIFF/WAV file.

    IEEE-float WAV writers commonly add a valid ``PEAK`` chunk whose timestamp
    changes once per second.  That field is unrelated to audio but otherwise
    defeats byte-for-byte M6 reproducibility.  Unknown/non-RIFF files are left
    untouched; chunk sizes and audio samples are never changed.
    """

    wav_path = Path(path)
    with wav_path.open("r+b") as handle:
        if handle.read(4) != b"RIFF":
            return
        handle.seek(8)
        if handle.read(4) != b"WAVE":
            return
        while True:
            chunk_id = handle.read(4)
            size_bytes = handle.read(4)
            if len(chunk_id) != 4 or len(size_bytes) != 4:
                return
            chunk_size = int.from_bytes(size_bytes, "little")
            payload_start = handle.tell()
            if chunk_id == b"PEAK" and chunk_size >= 8:
                handle.seek(payload_start + 4)
                handle.write(b"\0\0\0\0")
                return
            handle.seek(payload_start + chunk_size + (chunk_size & 1))


@dataclass(frozen=True)
class BankSplitPolicy:
    """Deterministically assign an acoustic space to exactly one split."""

    seed: int
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    policy_id: str = M6_SPLIT_POLICY

    def __post_init__(self) -> None:
        if self.policy_id != M6_SPLIT_POLICY:
            raise ValueError("unsupported M6 bank split policy")
        fractions = (
            float(self.train_fraction),
            float(self.validation_fraction),
            float(self.test_fraction),
        )
        if any(value <= 0.0 for value in fractions):
            raise ValueError("every M6 split fraction must be positive")
        if abs(sum(fractions) - 1.0) > 1e-12:
            raise ValueError("M6 split fractions must sum to one")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "train_fraction", fractions[0])
        object.__setattr__(self, "validation_fraction", fractions[1])
        object.__setattr__(self, "test_fraction", fractions[2])

    def assign(self, acoustic_space_id: str) -> str:
        identity = _required_text("acoustic_space_id", acoustic_space_id)
        payload = f"{self.policy_id}\0{self.seed}\0{identity}".encode()
        unit = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
        if unit < self.train_fraction:
            return "train"
        if unit < self.train_fraction + self.validation_fraction:
            return "validation"
        return "test"

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "seed": self.seed,
            "fractions": {
                "train": self.train_fraction,
                "validation": self.validation_fraction,
                "test": self.test_fraction,
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BankSplitPolicy":
        fractions = value["fractions"]
        return cls(
            policy_id=str(value["policy_id"]),
            seed=int(value["seed"]),
            train_fraction=float(fractions["train"]),
            validation_fraction=float(fractions["validation"]),
            test_fraction=float(fractions["test"]),
        )


@dataclass(frozen=True)
class BankGeneratorProvenance:
    """Pinned recipe and code identity shared by every item in a bank."""

    generator_id: str
    generator_version: str
    code_revision: str
    config_sha256: str
    task_plan_sha256: str
    seed: int

    def __post_init__(self) -> None:
        for name in ("generator_id", "generator_version", "code_revision"):
            object.__setattr__(self, name, _required_text(name, getattr(self, name)))
        object.__setattr__(
            self, "config_sha256", _sha256("generator config sha256", self.config_sha256)
        )
        object.__setattr__(
            self,
            "task_plan_sha256",
            _sha256("task plan sha256", self.task_plan_sha256),
        )
        object.__setattr__(self, "seed", int(self.seed))

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator_id": self.generator_id,
            "generator_version": self.generator_version,
            "code_revision": self.code_revision,
            "config_sha256": self.config_sha256,
            "task_plan_sha256": self.task_plan_sha256,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BankGeneratorProvenance":
        return cls(
            generator_id=str(value["generator_id"]),
            generator_version=str(value["generator_version"]),
            code_revision=str(value["code_revision"]),
            config_sha256=str(value["config_sha256"]),
            task_plan_sha256=str(value["task_plan_sha256"]),
            seed=int(value["seed"]),
        )


@dataclass(frozen=True)
class BankRendererProfile:
    """Renderer identity and evidence tier referenced by bank items."""

    profile_id: str
    renderer_id: str
    renderer_version: str
    low_backend: str
    high_backend: str
    scene_schema_version: str
    renderer_config_sha256: str
    evidence_tier: str
    calibration_report_sha256: str | None = None
    residual_model_sha256: str | None = None
    approval_report_sha256: str | None = None

    def __post_init__(self) -> None:
        required = (
            "profile_id",
            "renderer_id",
            "renderer_version",
            "low_backend",
            "high_backend",
            "scene_schema_version",
        )
        for name in required:
            object.__setattr__(self, name, _required_text(name, getattr(self, name)))
        if self.evidence_tier not in EVIDENCE_TIERS:
            raise ValueError(f"evidence_tier must be one of {EVIDENCE_TIERS}")
        object.__setattr__(
            self,
            "renderer_config_sha256",
            _sha256("renderer config sha256", self.renderer_config_sha256),
        )
        for name in (
            "calibration_report_sha256",
            "residual_model_sha256",
            "approval_report_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(name.replace("_", " "), getattr(self, name), optional=True),
            )
        if (
            self.evidence_tier == "production_approved"
            and self.approval_report_sha256 is None
        ):
            raise ValueError("production-approved renderer needs an approval report")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "renderer_id": self.renderer_id,
            "renderer_version": self.renderer_version,
            "low_backend": self.low_backend,
            "high_backend": self.high_backend,
            "scene_schema_version": self.scene_schema_version,
            "renderer_config_sha256": self.renderer_config_sha256,
            "evidence_tier": self.evidence_tier,
            "calibration_report_sha256": self.calibration_report_sha256,
            "residual_model_sha256": self.residual_model_sha256,
            "approval_report_sha256": self.approval_report_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BankRendererProfile":
        return cls(
            profile_id=str(value["profile_id"]),
            renderer_id=str(value["renderer_id"]),
            renderer_version=str(value["renderer_version"]),
            low_backend=str(value["low_backend"]),
            high_backend=str(value["high_backend"]),
            scene_schema_version=str(value["scene_schema_version"]),
            renderer_config_sha256=str(value["renderer_config_sha256"]),
            evidence_tier=str(value["evidence_tier"]),
            calibration_report_sha256=value.get("calibration_report_sha256"),
            residual_model_sha256=value.get("residual_model_sha256"),
            approval_report_sha256=value.get("approval_report_sha256"),
        )


@dataclass(frozen=True)
class RIRBankItem:
    """One content-addressed same-stem WAV/JSON RIR bank item."""

    item_id: str
    room_id: str
    acoustic_space_id: str
    scene_id: str
    split: str
    generation_seed: int
    origin: str
    renderer_profile_id: str
    signal_variant: str
    level_policy: str
    rir_path: str
    metadata_path: str
    rir_sha256: str
    metadata_sha256: str
    scene_sha256: str
    sample_rate: int
    channel_count: int
    frame_count: int
    qc_status: str = "pending"
    qc_report_path: str | None = None
    qc_report_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "item_id",
            _safe_path_component("item_id", self.item_id),
        )
        for name in (
            "room_id",
            "acoustic_space_id",
            "scene_id",
            "renderer_profile_id",
        ):
            object.__setattr__(self, name, _required_text(name, getattr(self, name)))
        if self.split not in BANK_SPLITS:
            raise ValueError(f"item split must be one of {BANK_SPLITS}")
        object.__setattr__(self, "generation_seed", int(self.generation_seed))
        if self.origin not in ORIGINS:
            raise ValueError(f"item origin must be one of {ORIGINS}")
        if self.signal_variant not in SIGNAL_VARIANTS:
            raise ValueError(f"signal_variant must be one of {SIGNAL_VARIANTS}")
        if self.level_policy not in LEVEL_POLICIES:
            raise ValueError(f"level_policy must be one of {LEVEL_POLICIES}")
        if self.qc_status not in QC_STATUSES:
            raise ValueError(f"qc_status must be one of {QC_STATUSES}")
        object.__setattr__(self, "rir_path", _relative_path("rir_path", self.rir_path))
        object.__setattr__(
            self,
            "metadata_path",
            _relative_path("metadata_path", self.metadata_path),
        )
        if self.rir_path == self.metadata_path:
            raise ValueError("RIR and metadata paths must differ")
        for name in ("rir_sha256", "metadata_sha256", "scene_sha256"):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        object.__setattr__(
            self,
            "qc_report_sha256",
            _sha256("QC report sha256", self.qc_report_sha256, optional=True),
        )
        if self.qc_report_path is not None:
            object.__setattr__(
                self,
                "qc_report_path",
                _relative_path("QC report path", self.qc_report_path),
            )
        if self.qc_status == "pending":
            if self.qc_report_path is not None or self.qc_report_sha256 is not None:
                raise ValueError("pending QC cannot reference a QC report")
        elif self.qc_report_path is None or self.qc_report_sha256 is None:
            raise ValueError("completed QC requires a content-addressed QC report")
        for name in ("sample_rate", "channel_count", "frame_count"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        qc = {
            "status": self.qc_status,
            "report_sha256": self.qc_report_sha256,
        }
        # Preserve the canonical M6.1/M6.2 representation for pending items.
        # Completed M6.3 items add the independently verifiable report path.
        if self.qc_report_path is not None:
            qc["report_path"] = self.qc_report_path
        return {
            "item_id": self.item_id,
            "room_id": self.room_id,
            "acoustic_space_id": self.acoustic_space_id,
            "scene_id": self.scene_id,
            "split": self.split,
            "generation_seed": self.generation_seed,
            "origin": self.origin,
            "renderer_profile_id": self.renderer_profile_id,
            "signal_variant": self.signal_variant,
            "level_policy": self.level_policy,
            "assets": {
                "rir": {"path": self.rir_path, "sha256": self.rir_sha256},
                "metadata": {
                    "path": self.metadata_path,
                    "sha256": self.metadata_sha256,
                },
            },
            "scene_sha256": self.scene_sha256,
            "audio": {
                "sample_rate": self.sample_rate,
                "channel_count": self.channel_count,
                "frame_count": self.frame_count,
            },
            "qc": qc,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RIRBankItem":
        assets = value["assets"]
        audio = value["audio"]
        qc = value.get("qc", {})
        return cls(
            item_id=str(value["item_id"]),
            room_id=str(value["room_id"]),
            acoustic_space_id=str(value["acoustic_space_id"]),
            scene_id=str(value["scene_id"]),
            split=str(value["split"]),
            generation_seed=int(value["generation_seed"]),
            origin=str(value["origin"]),
            renderer_profile_id=str(value["renderer_profile_id"]),
            signal_variant=str(value["signal_variant"]),
            level_policy=str(value["level_policy"]),
            rir_path=str(assets["rir"]["path"]),
            metadata_path=str(assets["metadata"]["path"]),
            rir_sha256=str(assets["rir"]["sha256"]),
            metadata_sha256=str(assets["metadata"]["sha256"]),
            scene_sha256=str(value["scene_sha256"]),
            sample_rate=int(audio["sample_rate"]),
            channel_count=int(audio["channel_count"]),
            frame_count=int(audio["frame_count"]),
            qc_status=str(qc.get("status", "pending")),
            qc_report_path=qc.get("report_path"),
            qc_report_sha256=qc.get("report_sha256"),
        )


@dataclass(frozen=True)
class BankSplitIndex:
    """Content-addressed JSONL index for one bank split."""

    path: str
    sha256: str
    item_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path("split index path", self.path))
        object.__setattr__(self, "sha256", _sha256("split index sha256", self.sha256))
        if int(self.item_count) <= 0:
            raise ValueError("split index item_count must be positive")
        object.__setattr__(self, "item_count", int(self.item_count))

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "item_count": self.item_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BankSplitIndex":
        return cls(
            path=str(value["path"]),
            sha256=str(value["sha256"]),
            item_count=int(value["item_count"]),
        )


def split_index_rows(
    items: Sequence[RIRBankItem],
    split: str,
) -> list[dict[str, Any]]:
    """Return the canonical, item-id-sorted JSONL rows for one split."""

    if split not in BANK_SPLITS:
        raise ValueError(f"split must be one of {BANK_SPLITS}")
    return [
        {
            "item_id": item.item_id,
            "room_id": item.room_id,
            "acoustic_space_id": item.acoustic_space_id,
            "rir_path": item.rir_path,
            "metadata_path": item.metadata_path,
        }
        for item in sorted(
            (item for item in items if item.split == split),
            key=lambda item: item.item_id,
        )
    ]


def task_plan_rows(items: Sequence[RIRBankItem]) -> list[dict[str, Any]]:
    """Return the canonical deterministic task plan represented by items."""

    return [
        {
            "item_id": item.item_id,
            "room_id": item.room_id,
            "acoustic_space_id": item.acoustic_space_id,
            "scene_id": item.scene_id,
            "scene_sha256": item.scene_sha256,
            "split": item.split,
            "generation_seed": item.generation_seed,
            "renderer_profile_id": item.renderer_profile_id,
            "sample_rate": item.sample_rate,
            "channel_count": item.channel_count,
            "frame_count": item.frame_count,
        }
        for item in sorted(items, key=lambda item: item.item_id)
    ]


def write_split_indexes(
    root: str | Path,
    items: Sequence[RIRBankItem],
    *,
    directory: str = "indexes",
) -> dict[str, BankSplitIndex]:
    """Write canonical train/validation/test JSONL indexes and hash them."""

    bank_root = Path(root)
    relative_directory = _relative_path("split index directory", directory)
    index_dir = bank_root / relative_directory
    index_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, BankSplitIndex] = {}
    for split in BANK_SPLITS:
        rows = split_index_rows(items, split)
        if not rows:
            raise ValueError(f"cannot write an empty {split} split index")
        path = index_dir / f"{split}.jsonl"
        payload = "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
        result[split] = BankSplitIndex(
            path=path.relative_to(bank_root).as_posix(),
            sha256=sha256_file(path),
            item_count=len(rows),
        )
    return result


@dataclass(frozen=True)
class RIRBankManifest:
    """Complete M6 release contract spanning train/validation/test items."""

    bank_id: str
    release_status: str
    split_policy: BankSplitPolicy
    generator: BankGeneratorProvenance
    renderer_profiles: tuple[BankRendererProfile, ...]
    items: tuple[RIRBankItem, ...]
    split_indexes: Mapping[str, BankSplitIndex]
    manifest_sha256: str | None = None
    schema_version: str = RIR_BANK_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RIR_BANK_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported RIR bank manifest schema")
        object.__setattr__(self, "bank_id", _required_text("bank_id", self.bank_id))
        if self.release_status not in RELEASE_STATUSES:
            raise ValueError(f"release_status must be one of {RELEASE_STATUSES}")
        if not self.renderer_profiles or not self.items:
            raise ValueError("bank needs at least one renderer profile and item")
        profile_ids = [profile.profile_id for profile in self.renderer_profiles]
        item_ids = [item.item_id for item in self.items]
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError("renderer profile ids must be unique")
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("bank item ids must be unique")
        if any(item.renderer_profile_id not in profile_ids for item in self.items):
            raise ValueError("bank item references an unknown renderer profile")
        indexes = {str(key): value for key, value in self.split_indexes.items()}
        if set(indexes) != set(BANK_SPLITS):
            raise ValueError("split_indexes must contain train, validation, and test")
        object.__setattr__(self, "split_indexes", indexes)
        object.__setattr__(
            self,
            "manifest_sha256",
            _sha256("manifest sha256", self.manifest_sha256, optional=True),
        )

    def to_dict(self, *, include_manifest_sha256: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "bank_id": self.bank_id,
            "release_status": self.release_status,
            "split_policy": self.split_policy.to_dict(),
            "generator": self.generator.to_dict(),
            "renderer_profiles": [item.to_dict() for item in self.renderer_profiles],
            "items": [item.to_dict() for item in self.items],
            "split_indexes": {
                split: self.split_indexes[split].to_dict() for split in BANK_SPLITS
            },
        }
        if include_manifest_sha256:
            result["manifest_sha256"] = self.manifest_sha256
        return result

    def content_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict(include_manifest_sha256=False))

    def with_content_sha256(self) -> "RIRBankManifest":
        return replace(self, manifest_sha256=self.content_sha256())

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RIRBankManifest":
        return cls(
            schema_version=str(value["schema_version"]),
            bank_id=str(value["bank_id"]),
            release_status=str(value["release_status"]),
            split_policy=BankSplitPolicy.from_dict(value["split_policy"]),
            generator=BankGeneratorProvenance.from_dict(value["generator"]),
            renderer_profiles=tuple(
                BankRendererProfile.from_dict(item)
                for item in value["renderer_profiles"]
            ),
            items=tuple(RIRBankItem.from_dict(item) for item in value["items"]),
            split_indexes={
                split: BankSplitIndex.from_dict(value["split_indexes"][split])
                for split in BANK_SPLITS
            },
            manifest_sha256=value.get("manifest_sha256"),
        )

    @classmethod
    def from_json(cls, value: str) -> "RIRBankManifest":
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("RIR bank manifest JSON root must be an object")
        return cls.from_dict(parsed)


def _metadata_identity_matches(item: RIRBankItem, metadata: Mapping[str, Any]) -> bool:
    scene = metadata.get("scene")
    if not isinstance(scene, Mapping):
        return False
    identities = (
        (metadata.get("sample_id"), item.item_id),
        (metadata.get("room_id"), item.room_id),
        (scene.get("scene_id"), item.scene_id),
    )
    return all(value is None or str(value) == expected for value, expected in identities)


def audit_rir_bank_manifest(
    manifest: RIRBankManifest,
    root: str | Path,
    *,
    verify_hashes: bool = True,
    inspect_audio: bool = True,
) -> dict[str, Any]:
    """Audit M6 split, provenance, asset, audio, and production semantics."""

    bank_root = Path(root)
    split_counts = {
        split: sum(item.split == split for item in manifest.items)
        for split in BANK_SPLITS
    }
    acoustic_splits: dict[str, set[str]] = {}
    room_splits: dict[str, set[str]] = {}
    for item in manifest.items:
        acoustic_splits.setdefault(item.acoustic_space_id, set()).add(item.split)
        room_splits.setdefault(item.room_id, set()).add(item.split)
    rir_paths = [item.rir_path for item in manifest.items]
    metadata_paths = [item.metadata_path for item in manifest.items]
    qc_report_paths = [
        item.qc_report_path
        for item in manifest.items
        if item.qc_report_path is not None
    ]
    index_paths = [manifest.split_indexes[split].path for split in BANK_SPLITS]
    all_paths = rir_paths + metadata_paths + qc_report_paths + index_paths
    missing_paths = sorted(path for path in all_paths if not (bank_root / path).is_file())
    hash_mismatches: list[str] = []
    audio_mismatches: list[str] = []
    metadata_mismatches: list[str] = []
    scene_hash_mismatches: list[str] = []
    qc_report_mismatches: list[str] = []
    split_index_mismatches: list[str] = []
    for item in manifest.items:
        rir_path = bank_root / item.rir_path
        metadata_path = bank_root / item.metadata_path
        if verify_hashes:
            for path, expected in (
                (rir_path, item.rir_sha256),
                (metadata_path, item.metadata_sha256),
            ):
                if path.is_file() and sha256_file(path) != expected:
                    hash_mismatches.append(path.relative_to(bank_root).as_posix())
        if inspect_audio and rir_path.is_file():
            try:
                info = sf.info(rir_path)
                if (
                    info.samplerate != item.sample_rate
                    or info.channels != item.channel_count
                    or info.frames != item.frame_count
                ):
                    audio_mismatches.append(item.rir_path)
            except (RuntimeError, OSError):
                audio_mismatches.append(item.rir_path)
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if not isinstance(metadata, dict) or not _metadata_identity_matches(
                    item, metadata
                ):
                    metadata_mismatches.append(item.metadata_path)
                scene = metadata.get("scene") if isinstance(metadata, dict) else None
                if (
                    not isinstance(scene, Mapping)
                    or canonical_json_sha256(scene) != item.scene_sha256
                ):
                    scene_hash_mismatches.append(item.metadata_path)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                metadata_mismatches.append(item.metadata_path)
                scene_hash_mismatches.append(item.metadata_path)
        if item.qc_report_path is not None:
            qc_report_path = bank_root / item.qc_report_path
            if (
                verify_hashes
                and qc_report_path.is_file()
                and sha256_file(qc_report_path) != item.qc_report_sha256
            ):
                qc_report_mismatches.append(item.qc_report_path)
    for split in BANK_SPLITS:
        index = manifest.split_indexes[split]
        index_path = bank_root / index.path
        if verify_hashes and index_path.is_file():
            if sha256_file(index_path) != index.sha256:
                hash_mismatches.append(index.path)
        if index_path.is_file():
            try:
                rows = [
                    json.loads(line)
                    for line in index_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                expected = split_index_rows(manifest.items, split)
                if rows != expected or index.item_count != len(expected):
                    split_index_mismatches.append(index.path)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                split_index_mismatches.append(index.path)
    profile_by_id = {
        profile.profile_id: profile for profile in manifest.renderer_profiles
    }
    production_claim_backed = manifest.release_status != "production" or (
        all(
            profile.evidence_tier == "production_approved"
            and profile.approval_report_sha256 is not None
            for profile in profile_by_id.values()
        )
        and all(
            item.qc_status == "pass"
            and item.qc_report_path is not None
            and item.qc_report_sha256 is not None
            for item in manifest.items
        )
        and manifest.generator.code_revision.lower() not in {"dirty", "unknown"}
        and not manifest.generator.code_revision.lower().endswith("-dirty")
    )
    checks = {
        "manifest_content_hash_matches": (
            manifest.manifest_sha256 is not None
            and manifest.manifest_sha256 == manifest.content_sha256()
        ),
        "train_validation_test_items_present": all(
            split_counts[split] > 0 for split in BANK_SPLITS
        ),
        "deterministic_acoustic_space_assignments_match": all(
            item.split == manifest.split_policy.assign(item.acoustic_space_id)
            for item in manifest.items
        ),
        "acoustic_spaces_are_split_disjoint": all(
            len(splits) == 1 for splits in acoustic_splits.values()
        ),
        "room_ids_are_split_disjoint": all(
            len(splits) == 1 for splits in room_splits.values()
        ),
        "asset_paths_are_unique": len(set(all_paths)) == len(all_paths),
        "all_assets_exist": not missing_paths,
        "all_asset_hashes_match": not hash_mismatches,
        "audio_headers_match_manifest": not audio_mismatches,
        "metadata_identity_matches_manifest": not metadata_mismatches,
        "scene_hashes_match_metadata": not scene_hash_mismatches,
        "qc_report_hashes_match_manifest": not qc_report_mismatches,
        "split_indexes_match_manifest_items": not split_index_mismatches,
        "task_plan_hash_matches_manifest_items": (
            manifest.generator.task_plan_sha256
            == canonical_json_sha256(task_plan_rows(manifest.items))
        ),
        "production_claim_is_evidence_backed": production_claim_backed,
    }
    valid = bool(all(checks.values()))
    return {
        "schema_version": RIR_BANK_AUDIT_SCHEMA_VERSION,
        "bank_id": manifest.bank_id,
        "root": str(bank_root),
        "release_status": manifest.release_status,
        "item_count": len(manifest.items),
        "acoustic_space_count": len(acoustic_splits),
        "room_count": len(room_splits),
        "renderer_profile_count": len(profile_by_id),
        "split_item_counts": split_counts,
        "missing_asset_paths": missing_paths,
        "hash_mismatch_paths": sorted(set(hash_mismatches)),
        "audio_header_mismatch_paths": sorted(set(audio_mismatches)),
        "metadata_identity_mismatch_paths": sorted(set(metadata_mismatches)),
        "scene_hash_mismatch_paths": sorted(set(scene_hash_mismatches)),
        "qc_report_mismatch_paths": sorted(set(qc_report_mismatches)),
        "split_index_mismatch_paths": sorted(set(split_index_mismatches)),
        "checks": checks,
        "ready_for_m6_bank_generation": valid,
        "ready_for_production": bool(valid and manifest.release_status == "production"),
    }


__all__ = [
    "BANK_SPLITS",
    "M6_SPLIT_POLICY",
    "RIR_BANK_AUDIT_SCHEMA_VERSION",
    "RIR_BANK_MANIFEST_SCHEMA_VERSION",
    "BankGeneratorProvenance",
    "BankRendererProfile",
    "BankSplitPolicy",
    "BankSplitIndex",
    "RIRBankItem",
    "RIRBankManifest",
    "audit_rir_bank_manifest",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "canonicalize_float_wav_header",
    "sha256_file",
    "split_index_rows",
    "task_plan_rows",
    "write_split_indexes",
]
