"""M6.4 content-addressed RIR variant and distribution releases."""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

from puresound.audio.rir.bank.schema import (
    BANK_SPLITS,
    BankGeneratorProvenance,
    RIRBankItem,
    RIRBankManifest,
    canonical_json_sha256,
    canonicalize_float_wav_header,
    sha256_file,
    task_plan_rows,
    write_split_indexes,
)
from puresound.audio.rir.bank.qc import (
    audit_rir_bank_qc_release,
    run_rir_bank_qc,
)


RIR_BANK_RELEASE_SCHEMA_VERSION = "puresound.rir_bank_release.v1"
RIR_BANK_DISTRIBUTION_SCHEMA_VERSION = "puresound.rir_bank_distribution.v1"
RELEASE_RECIPE_STATUSES = ("ready", "blocked")
DEFAULT_RELEASE_MANIFEST_NAME = "rir_bank_release.json"


def _required_text(name: str, value: Any) -> str:
    result = str(value)
    if not result.strip():
        raise ValueError(f"{name} is required")
    return result


def _sha256(name: str, value: Any) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must contain 64 hexadecimal characters")
    return digest


def _relative_path(name: str, value: Any) -> str:
    path = PurePosixPath(str(value))
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a safe release-relative path")
    return path.as_posix()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


@dataclass(frozen=True)
class ReleaseIndex:
    path: str
    sha256: str
    item_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path("release index path", self.path))
        object.__setattr__(self, "sha256", _sha256("release index sha256", self.sha256))
        if int(self.item_count) < 0:
            raise ValueError("release index item_count cannot be negative")
        object.__setattr__(self, "item_count", int(self.item_count))

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "item_count": self.item_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReleaseIndex":
        return cls(
            path=str(value["path"]),
            sha256=str(value["sha256"]),
            item_count=int(value["item_count"]),
        )


@dataclass(frozen=True)
class ReleaseVariant:
    variant_id: str
    origin: str
    signal_variant: str
    level_policy: str
    bank_path: str
    manifest_path: str
    manifest_file_sha256: str
    qc_summary_path: str
    qc_summary_file_sha256: str
    distribution_path: str
    distribution_file_sha256: str
    item_count: int
    split_item_counts: Mapping[str, int]
    transform: str
    parent_variant_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("variant_id", "origin", "signal_variant", "level_policy", "transform"):
            object.__setattr__(self, name, _required_text(name, getattr(self, name)))
        for name in ("bank_path", "manifest_path", "qc_summary_path", "distribution_path"):
            object.__setattr__(self, name, _relative_path(name, getattr(self, name)))
        for name in (
            "manifest_file_sha256",
            "qc_summary_file_sha256",
            "distribution_file_sha256",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        if int(self.item_count) <= 0:
            raise ValueError("release variant must contain at least one item")
        object.__setattr__(self, "item_count", int(self.item_count))
        counts = {str(key): int(value) for key, value in self.split_item_counts.items()}
        if set(counts) != set(BANK_SPLITS) or any(value <= 0 for value in counts.values()):
            raise ValueError("variant split counts must contain positive train/validation/test")
        if sum(counts.values()) != self.item_count:
            raise ValueError("variant split counts must sum to item_count")
        object.__setattr__(self, "split_item_counts", counts)
        if self.parent_variant_id is not None:
            object.__setattr__(
                self,
                "parent_variant_id",
                _required_text("parent_variant_id", self.parent_variant_id),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "origin": self.origin,
            "signal_variant": self.signal_variant,
            "level_policy": self.level_policy,
            "bank_path": self.bank_path,
            "manifest": {
                "path": self.manifest_path,
                "file_sha256": self.manifest_file_sha256,
            },
            "qc_summary": {
                "path": self.qc_summary_path,
                "file_sha256": self.qc_summary_file_sha256,
            },
            "distribution": {
                "path": self.distribution_path,
                "file_sha256": self.distribution_file_sha256,
            },
            "item_count": self.item_count,
            "split_item_counts": {
                split: self.split_item_counts[split] for split in BANK_SPLITS
            },
            "transform": self.transform,
            "parent_variant_id": self.parent_variant_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReleaseVariant":
        return cls(
            variant_id=str(value["variant_id"]),
            origin=str(value["origin"]),
            signal_variant=str(value["signal_variant"]),
            level_policy=str(value["level_policy"]),
            bank_path=str(value["bank_path"]),
            manifest_path=str(value["manifest"]["path"]),
            manifest_file_sha256=str(value["manifest"]["file_sha256"]),
            qc_summary_path=str(value["qc_summary"]["path"]),
            qc_summary_file_sha256=str(value["qc_summary"]["file_sha256"]),
            distribution_path=str(value["distribution"]["path"]),
            distribution_file_sha256=str(value["distribution"]["file_sha256"]),
            item_count=int(value["item_count"]),
            split_item_counts=value["split_item_counts"],
            transform=str(value["transform"]),
            parent_variant_id=value.get("parent_variant_id"),
        )


@dataclass(frozen=True)
class ReleaseRecipe:
    recipe_id: str
    status: str
    variant_ids: tuple[str, ...]
    origin_weights: Mapping[str, float]
    split_indexes: Mapping[str, ReleaseIndex] | None = None
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "recipe_id", _required_text("recipe_id", self.recipe_id))
        if self.status not in RELEASE_RECIPE_STATUSES:
            raise ValueError(f"recipe status must be one of {RELEASE_RECIPE_STATUSES}")
        variants = tuple(_required_text("variant_id", value) for value in self.variant_ids)
        object.__setattr__(self, "variant_ids", variants)
        weights = {str(key): float(value) for key, value in self.origin_weights.items()}
        if any(not math.isfinite(value) or value <= 0 for value in weights.values()):
            raise ValueError("origin weights must be finite and positive")
        if weights and not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError("origin weights must sum to one")
        object.__setattr__(self, "origin_weights", weights)
        indexes = (
            None
            if self.split_indexes is None
            else {str(key): value for key, value in self.split_indexes.items()}
        )
        blockers = tuple(_required_text("recipe blocker", value) for value in self.blockers)
        object.__setattr__(self, "blockers", blockers)
        if self.status == "ready":
            if not variants or set(indexes or {}) != set(BANK_SPLITS) or blockers:
                raise ValueError("ready recipe needs variants, three indexes, and no blockers")
            if any((indexes or {})[split].item_count <= 0 for split in BANK_SPLITS):
                raise ValueError("ready recipe indexes cannot be empty")
        elif indexes is not None or not blockers:
            raise ValueError("blocked recipe needs blockers and cannot publish indexes")
        object.__setattr__(self, "split_indexes", indexes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "status": self.status,
            "variant_ids": list(self.variant_ids),
            "origin_weights": dict(sorted(self.origin_weights.items())),
            "split_indexes": (
                None
                if self.split_indexes is None
                else {
                    split: self.split_indexes[split].to_dict()
                    for split in BANK_SPLITS
                }
            ),
            "blockers": list(self.blockers),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReleaseRecipe":
        indexes = value.get("split_indexes")
        return cls(
            recipe_id=str(value["recipe_id"]),
            status=str(value["status"]),
            variant_ids=tuple(str(item) for item in value["variant_ids"]),
            origin_weights=value["origin_weights"],
            split_indexes=(
                None
                if indexes is None
                else {
                    split: ReleaseIndex.from_dict(indexes[split])
                    for split in BANK_SPLITS
                }
            ),
            blockers=tuple(str(item) for item in value.get("blockers", ())),
        )


@dataclass(frozen=True)
class RIRBankReleaseManifest:
    release_id: str
    release_status: str
    variants: tuple[ReleaseVariant, ...]
    recipes: tuple[ReleaseRecipe, ...]
    release_sha256: str | None = None
    schema_version: str = RIR_BANK_RELEASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RIR_BANK_RELEASE_SCHEMA_VERSION:
            raise ValueError("unsupported RIR bank release schema")
        object.__setattr__(self, "release_id", _required_text("release_id", self.release_id))
        if self.release_status not in {"draft", "candidate", "production"}:
            raise ValueError("invalid release status")
        if not self.variants or not self.recipes:
            raise ValueError("release needs variants and recipes")
        variant_ids = [variant.variant_id for variant in self.variants]
        recipe_ids = [recipe.recipe_id for recipe in self.recipes]
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError("release variant ids must be unique")
        if len(set(recipe_ids)) != len(recipe_ids):
            raise ValueError("release recipe ids must be unique")
        if any(
            referenced not in variant_ids
            for recipe in self.recipes
            for referenced in recipe.variant_ids
        ):
            raise ValueError("release recipe references an unknown variant")
        if self.release_sha256 is not None:
            object.__setattr__(
                self, "release_sha256", _sha256("release sha256", self.release_sha256)
            )

    def to_dict(self, *, include_release_sha256: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "release_id": self.release_id,
            "release_status": self.release_status,
            "variants": [variant.to_dict() for variant in self.variants],
            "recipes": [recipe.to_dict() for recipe in self.recipes],
        }
        if include_release_sha256:
            result["release_sha256"] = self.release_sha256
        return result

    def content_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict(include_release_sha256=False))

    def with_content_sha256(self) -> "RIRBankReleaseManifest":
        return replace(self, release_sha256=self.content_sha256())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RIRBankReleaseManifest":
        return cls(
            schema_version=str(value["schema_version"]),
            release_id=str(value["release_id"]),
            release_status=str(value["release_status"]),
            variants=tuple(ReleaseVariant.from_dict(item) for item in value["variants"]),
            recipes=tuple(ReleaseRecipe.from_dict(item) for item in value["recipes"]),
            release_sha256=value.get("release_sha256"),
        )

    @classmethod
    def from_json(cls, value: str) -> "RIRBankReleaseManifest":
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("RIR bank release root must be an object")
        return cls.from_dict(parsed)


def _numeric_summary(values: Sequence[Any]) -> dict[str, float | int | None]:
    finite = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if not finite.size:
        return {
            "count": 0,
            "minimum": None,
            "p05": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "count": int(finite.size),
        "minimum": float(np.min(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "maximum": float(np.max(finite)),
    }


def compute_bank_distribution(
    bank_root: str | Path,
    *,
    variant_id: str,
) -> dict[str, Any]:
    """Freeze item/channel acoustic and scene distributions from QC reports."""

    root = Path(bank_root)
    manifest = RIRBankManifest.from_json(
        (root / "rir_bank_manifest.json").read_text(encoding="utf-8")
    )
    item_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    room_types: Counter[str] = Counter()
    origins: Counter[str] = Counter()
    for item in sorted(manifest.items, key=lambda value: value.item_id):
        if item.qc_status != "pass" or item.qc_report_path is None:
            continue
        metadata = json.loads((root / item.metadata_path).read_text(encoding="utf-8"))
        report = json.loads((root / item.qc_report_path).read_text(encoding="utf-8"))
        scene = metadata["scene"]
        dimensions = scene.get("dimensions_m", scene.get("room_dim"))
        volume = None
        if isinstance(dimensions, list) and len(dimensions) == 3:
            volume = float(np.prod(np.asarray(dimensions, dtype=np.float64)))
        room_type = str(scene.get("room_type", "unknown"))
        room_types[room_type] += 1
        origins[item.origin] += 1
        item_rows.append(
            {
                "item_id": item.item_id,
                "room_id": item.room_id,
                "acoustic_space_id": item.acoustic_space_id,
                "split": item.split,
                "origin": item.origin,
                "signal_variant": item.signal_variant,
                "level_policy": item.level_policy,
                "room_type": room_type,
                "room_volume_m3": volume,
                "rt60_metadata_s": scene.get("rt60"),
            }
        )
        for channel in report["channels"]:
            metrics = channel.get("metrics", {})
            echo = metrics.get("echo_density", {})
            channel_rows.append(
                {
                    "item_id": item.item_id,
                    "acoustic_space_id": item.acoustic_space_id,
                    "split": item.split,
                    "channel": int(channel["channel"]),
                    "label": str(channel.get("label", "")),
                    "distance_m": channel.get("distance_m"),
                    "peak_abs": channel.get("energy", {}).get("peak_abs"),
                    "tail_energy_fraction": channel.get("energy", {}).get(
                        "tail_fraction"
                    ),
                    "drr_db": metrics.get("drr_db"),
                    "c50_db": metrics.get("c50_db"),
                    "c80_db": metrics.get("c80_db"),
                    "t20_s": metrics.get("t20_s"),
                    "spectral_tilt_db_per_octave": metrics.get(
                        "spectral_tilt_db_per_octave"
                    ),
                    "mixing_time_s": (
                        echo.get("mixing_time_s") if isinstance(echo, Mapping) else None
                    ),
                    "late_median_normalized_density": (
                        echo.get("late_median_normalized_density")
                        if isinstance(echo, Mapping)
                        else None
                    ),
                }
            )
    metric_names = (
        "distance_m",
        "peak_abs",
        "tail_energy_fraction",
        "drr_db",
        "c50_db",
        "c80_db",
        "t20_s",
        "spectral_tilt_db_per_octave",
        "mixing_time_s",
        "late_median_normalized_density",
    )
    distribution: dict[str, Any] = {
        "schema_version": RIR_BANK_DISTRIBUTION_SCHEMA_VERSION,
        "variant_id": str(variant_id),
        "bank_id": manifest.bank_id,
        "manifest_sha256": manifest.manifest_sha256,
        "counts": {
            "items": len(item_rows),
            "channels": len(channel_rows),
            "acoustic_spaces": len(
                {row["acoustic_space_id"] for row in item_rows}
            ),
            "by_split": {
                split: sum(row["split"] == split for row in item_rows)
                for split in BANK_SPLITS
            },
            "room_types": dict(sorted(room_types.items())),
            "origins": dict(sorted(origins.items())),
        },
        "item_metrics": {
            "room_volume_m3": _numeric_summary(
                [row["room_volume_m3"] for row in item_rows]
            ),
            "rt60_metadata_s": _numeric_summary(
                [row["rt60_metadata_s"] for row in item_rows]
            ),
        },
        "channel_metrics": {
            name: _numeric_summary([row[name] for row in channel_rows])
            for name in metric_names
        },
        "items": item_rows,
        "channels": channel_rows,
    }
    distribution["distribution_sha256"] = canonical_json_sha256(distribution)
    return distribution


def _write_distribution(root: Path, variant_id: str) -> Path:
    path = root / "distribution.json"
    _atomic_json(path, compute_bank_distribution(root, variant_id=variant_id))
    return path


def _completed_items(manifest: RIRBankManifest) -> tuple[RIRBankItem, ...]:
    return tuple(item for item in manifest.items if item.qc_status == "pass")


def _materialize_peak_normalized_bank(
    source_root: Path,
    output_root: Path,
    *,
    target_peak: float,
    qc_workers: int = 1,
) -> RIRBankManifest:
    if not 0.0 < float(target_peak) <= 1.0:
        raise ValueError("target_peak must be within (0, 1]")
    source = RIRBankManifest.from_json(
        (source_root / "rir_bank_manifest.json").read_text(encoding="utf-8")
    )
    source_items = _completed_items(source)
    output_root.mkdir(parents=True, exist_ok=False)
    items: list[RIRBankItem] = []
    bank_id = f"{source.bank_id}-peak-normalized"
    for parent in sorted(source_items, key=lambda value: value.item_id):
        source_rir_path = source_root / parent.rir_path
        source_metadata_path = source_root / parent.metadata_path
        rir, sample_rate = sf.read(source_rir_path, always_2d=True, dtype="float64")
        peak = float(np.max(np.abs(rir))) if rir.size else 0.0
        if not math.isfinite(peak) or peak <= 0.0:
            raise ValueError(f"cannot normalize silent or non-finite item {parent.item_id}")
        normalized = rir * (float(target_peak) / peak)
        item_id = f"{parent.item_id}__peak_normalized"
        parent_rir = PurePosixPath(parent.rir_path)
        relative_dir = parent_rir.parent
        rir_path = output_root / relative_dir / f"{item_id}.wav"
        metadata_path = output_root / relative_dir / f"{item_id}.json"
        rir_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(rir_path, normalized, int(sample_rate), subtype="FLOAT")
        canonicalize_float_wav_header(rir_path)
        rir_sha256 = sha256_file(rir_path)
        metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        metadata["sample_id"] = item_id
        metadata["room_id"] = parent.room_id
        metadata["m6_variant"] = {
            "variant_id": "synthetic_peak_normalized",
            "parent_item_id": parent.item_id,
            "parent_rir_sha256": parent.rir_sha256,
            "transform": "one_common_item_gain_to_peak",
            "target_peak_abs": float(target_peak),
        }
        if isinstance(metadata.get("m6"), dict):
            metadata["m6"].update(
                {
                    "bank_id": bank_id,
                    "rir_sha256": rir_sha256,
                    "sample_rate": int(sample_rate),
                    "channel_count": int(normalized.shape[1]),
                    "frame_count": int(normalized.shape[0]),
                }
            )
        _atomic_json(metadata_path, metadata)
        items.append(
            replace(
                parent,
                item_id=item_id,
                level_policy="peak_normalized",
                rir_path=rir_path.relative_to(output_root).as_posix(),
                metadata_path=metadata_path.relative_to(output_root).as_posix(),
                rir_sha256=rir_sha256,
                metadata_sha256=sha256_file(metadata_path),
                qc_status="pending",
                qc_report_path=None,
                qc_report_sha256=None,
            )
        )
    item_tuple = tuple(items)
    split_indexes = write_split_indexes(output_root, item_tuple)
    transform_config = {
        "schema_version": "puresound.rir_bank_variant_transform.v1",
        "parent_manifest_sha256": source.manifest_sha256,
        "transform": "one_common_item_gain_to_peak",
        "target_peak_abs": float(target_peak),
    }
    generator = BankGeneratorProvenance(
        generator_id="puresound.audio.rir.bank.release",
        generator_version="M6.4",
        code_revision=source.generator.code_revision,
        config_sha256=canonical_json_sha256(transform_config),
        task_plan_sha256=canonical_json_sha256(task_plan_rows(item_tuple)),
        seed=source.generator.seed,
    )
    manifest = RIRBankManifest(
        bank_id=bank_id,
        release_status="draft",
        split_policy=source.split_policy,
        generator=generator,
        renderer_profiles=source.renderer_profiles,
        items=item_tuple,
        split_indexes=split_indexes,
    ).with_content_sha256()
    _atomic_json(output_root / "rir_bank_manifest.json", manifest.to_dict())
    run_rir_bank_qc(output_root, workers=qc_workers)
    return RIRBankManifest.from_json(
        (output_root / "rir_bank_manifest.json").read_text(encoding="utf-8")
    )


def _variant_reference(
    release_root: Path,
    bank_root: Path,
    *,
    variant_id: str,
    transform: str,
    parent_variant_id: str | None,
) -> ReleaseVariant:
    manifest_path = bank_root / "rir_bank_manifest.json"
    summary_path = bank_root / "rir_bank_qc_summary.json"
    manifest = RIRBankManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    passed = _completed_items(manifest)
    origins = {item.origin for item in passed}
    signals = {item.signal_variant for item in passed}
    levels = {item.level_policy for item in passed}
    if len(origins) != 1 or len(signals) != 1 or len(levels) != 1:
        raise ValueError("release variant items need uniform origin/signal/level semantics")
    distribution_path = _write_distribution(bank_root, variant_id)
    return ReleaseVariant(
        variant_id=variant_id,
        origin=next(iter(origins)),
        signal_variant=next(iter(signals)),
        level_policy=next(iter(levels)),
        bank_path=bank_root.relative_to(release_root).as_posix(),
        manifest_path=manifest_path.relative_to(release_root).as_posix(),
        manifest_file_sha256=sha256_file(manifest_path),
        qc_summary_path=summary_path.relative_to(release_root).as_posix(),
        qc_summary_file_sha256=sha256_file(summary_path),
        distribution_path=distribution_path.relative_to(release_root).as_posix(),
        distribution_file_sha256=sha256_file(distribution_path),
        item_count=len(passed),
        split_item_counts={
            split: sum(item.split == split for item in passed) for split in BANK_SPLITS
        },
        transform=transform,
        parent_variant_id=parent_variant_id,
    )


def _recipe_rows(
    release_root: Path,
    variants: Mapping[str, ReleaseVariant],
    variant_ids: Sequence[str],
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant_id in sorted(variant_ids):
        variant = variants[variant_id]
        bank_root = release_root / variant.bank_path
        manifest = RIRBankManifest.from_json(
            (release_root / variant.manifest_path).read_text(encoding="utf-8")
        )
        for item in sorted(manifest.items, key=lambda value: value.item_id):
            if item.split != split or item.qc_status != "pass":
                continue
            rows.append(
                {
                    "release_item_id": f"{variant_id}:{item.item_id}",
                    "variant_id": variant_id,
                    "item_id": item.item_id,
                    "room_id": item.room_id,
                    "acoustic_space_id": item.acoustic_space_id,
                    "split": split,
                    "origin": item.origin,
                    "signal_variant": item.signal_variant,
                    "level_policy": item.level_policy,
                    "bank_path": variant.bank_path,
                    "rir_path": item.rir_path,
                    "metadata_path": item.metadata_path,
                    "rir_sha256": item.rir_sha256,
                    "metadata_sha256": item.metadata_sha256,
                    "qc_report_sha256": item.qc_report_sha256,
                }
            )
        if not bank_root.is_dir():
            raise FileNotFoundError(bank_root)
    return rows


def _ready_recipe(
    release_root: Path,
    variants: Mapping[str, ReleaseVariant],
    *,
    recipe_id: str,
    variant_ids: tuple[str, ...],
    origin_weights: Mapping[str, float],
) -> ReleaseRecipe:
    indexes: dict[str, ReleaseIndex] = {}
    for split in BANK_SPLITS:
        rows = _recipe_rows(release_root, variants, variant_ids, split)
        path = release_root / "recipes" / recipe_id / f"{split}.jsonl"
        _atomic_jsonl(path, rows)
        indexes[split] = ReleaseIndex(
            path=path.relative_to(release_root).as_posix(),
            sha256=sha256_file(path),
            item_count=len(rows),
        )
    return ReleaseRecipe(
        recipe_id=recipe_id,
        status="ready",
        variant_ids=variant_ids,
        origin_weights=origin_weights,
        split_indexes=indexes,
    )


#: Sampling proportions for ``mixed_calibrated_real``. Equal weighting is a
#: neutral default, not a measured optimum: no downstream experiment has compared
#: mixtures yet, and the two pools differ in size by orders of magnitude, so any
#: other split would encode an assumption nothing has tested.
DEFAULT_MIXED_ORIGIN_WEIGHTS = {"synthetic": 0.5, "real": 0.5}


def build_m6_variant_release(
    source_bank_root: str | Path,
    output_root: str | Path,
    *,
    release_id: str = "puresound-m6-candidate",
    normalized_peak: float = 0.98,
    qc_workers: int = 1,
    measured_bank_root: str | Path | None = None,
    mixed_origin_weights: Mapping[str, float] | None = None,
) -> RIRBankReleaseManifest:
    """Build the synthetic candidate variants, and a measured one when supplied.

    Without ``measured_bank_root`` the real and mixed recipes stay blocked, which
    is the honest state of a release that has no measured data in it. Supplying a
    QC-passed measured bank — see
    :func:`puresound.audio.rir.bank.measured_ingest.build_measured_m6_bank` —
    adds a ``measured_native`` variant and makes both recipes ready.
    """

    source_root = Path(source_bank_root)
    release_root = Path(output_root)
    measured_root = Path(measured_bank_root) if measured_bank_root else None
    weights = dict(mixed_origin_weights or DEFAULT_MIXED_ORIGIN_WEIGHTS)
    if release_root.exists():
        raise FileExistsError(f"release output already exists: {release_root}")
    source_audit = audit_rir_bank_qc_release(source_root)
    if not source_audit["valid"]:
        raise ValueError("source bank is not a valid M6.3 QC release")
    if measured_root is not None:
        if not audit_rir_bank_qc_release(measured_root)["valid"]:
            raise ValueError("measured bank is not a valid M6.3 QC release")
        if set(weights) != {"synthetic", "real"} or any(
            value <= 0.0 for value in weights.values()
        ):
            raise ValueError(
                "mixed_origin_weights needs a positive synthetic and real weight"
            )
    release_root.mkdir(parents=True)
    calibrated_root = release_root / "variants" / "calibrated"
    normalized_root = release_root / "variants" / "peak_normalized"
    shutil.copytree(source_root, calibrated_root)
    _materialize_peak_normalized_bank(
        calibrated_root,
        normalized_root,
        target_peak=normalized_peak,
        qc_workers=qc_workers,
    )
    calibrated = _variant_reference(
        release_root,
        calibrated_root,
        variant_id="synthetic_calibrated",
        transform="identity_copy_of_qc_candidate",
        parent_variant_id=None,
    )
    normalized = _variant_reference(
        release_root,
        normalized_root,
        variant_id="synthetic_peak_normalized",
        transform="one_common_item_gain_to_peak",
        parent_variant_id=calibrated.variant_id,
    )
    built = [calibrated, normalized]
    measured: ReleaseVariant | None = None
    if measured_root is not None:
        measured_variant_root = release_root / "variants" / "measured_native"
        shutil.copytree(measured_root, measured_variant_root)
        measured = _variant_reference(
            release_root,
            measured_variant_root,
            variant_id="measured_native",
            transform="iso3382_onset_aligned_copy_of_measured_qc_release",
            parent_variant_id=None,
        )
        built.append(measured)
    variants = {item.variant_id: item for item in built}
    if measured is None:
        real_and_mixed = (
            ReleaseRecipe(
                recipe_id="real_native",
                status="blocked",
                variant_ids=(),
                origin_weights={},
                blockers=("no QC-passed measured M6 variant was supplied",),
            ),
            ReleaseRecipe(
                recipe_id="mixed_calibrated_real",
                status="blocked",
                variant_ids=(),
                origin_weights={},
                blockers=("real_native recipe is unavailable",),
            ),
        )
    else:
        real_and_mixed = (
            _ready_recipe(
                release_root,
                variants,
                recipe_id="real_native",
                variant_ids=(measured.variant_id,),
                origin_weights={"real": 1.0},
            ),
            _ready_recipe(
                release_root,
                variants,
                recipe_id="mixed_calibrated_real",
                variant_ids=(calibrated.variant_id, measured.variant_id),
                origin_weights=weights,
            ),
        )
    recipes = (
        _ready_recipe(
            release_root,
            variants,
            recipe_id="synthetic_calibrated",
            variant_ids=(calibrated.variant_id,),
            origin_weights={"synthetic": 1.0},
        ),
        _ready_recipe(
            release_root,
            variants,
            recipe_id="synthetic_peak_normalized",
            variant_ids=(normalized.variant_id,),
            origin_weights={"synthetic": 1.0},
        ),
        *real_and_mixed,
    )
    release = RIRBankReleaseManifest(
        release_id=release_id,
        release_status="candidate",
        variants=tuple(variants.values()),
        recipes=recipes,
    ).with_content_sha256()
    _atomic_json(release_root / DEFAULT_RELEASE_MANIFEST_NAME, release.to_dict())
    return release


def audit_m6_variant_release(
    root: str | Path,
    *,
    manifest_name: str = DEFAULT_RELEASE_MANIFEST_NAME,
) -> dict[str, Any]:
    """Audit variant lineage, distribution snapshots, and recipe membership."""

    release_root = Path(root)
    release = RIRBankReleaseManifest.from_json(
        (release_root / manifest_name).read_text(encoding="utf-8")
    )
    variant_checks: dict[str, bool] = {}
    manifests: dict[str, RIRBankManifest] = {}
    for variant in release.variants:
        bank_root = release_root / variant.bank_path
        manifest_path = release_root / variant.manifest_path
        summary_path = release_root / variant.qc_summary_path
        distribution_path = release_root / variant.distribution_path
        try:
            manifest = RIRBankManifest.from_json(
                manifest_path.read_text(encoding="utf-8")
            )
            distribution = json.loads(distribution_path.read_text(encoding="utf-8"))
            passed = _completed_items(manifest)
            expected_distribution = compute_bank_distribution(
                bank_root, variant_id=variant.variant_id
            )
            valid = bool(
                sha256_file(manifest_path) == variant.manifest_file_sha256
                and sha256_file(summary_path) == variant.qc_summary_file_sha256
                and sha256_file(distribution_path) == variant.distribution_file_sha256
                and audit_rir_bank_qc_release(bank_root)["valid"]
                and distribution == expected_distribution
                and distribution.get("distribution_sha256")
                == canonical_json_sha256(
                    {
                        key: value
                        for key, value in distribution.items()
                        if key != "distribution_sha256"
                    }
                )
                and len(passed) == variant.item_count
                and all(item.origin == variant.origin for item in passed)
                and all(item.signal_variant == variant.signal_variant for item in passed)
                and all(item.level_policy == variant.level_policy for item in passed)
                and all(
                    sum(item.split == split for item in passed)
                    == variant.split_item_counts[split]
                    for split in BANK_SPLITS
                )
            )
            manifests[variant.variant_id] = manifest
        except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError):
            valid = False
        variant_checks[variant.variant_id] = valid

    lineage_checks: dict[str, bool] = {}
    for variant in release.variants:
        if variant.parent_variant_id is None:
            continue
        child = manifests.get(variant.variant_id)
        parent = manifests.get(variant.parent_variant_id)
        if child is None or parent is None:
            lineage_checks[variant.variant_id] = False
            continue
        child_items = tuple(
            item for item in child.items if item.qc_status == "pass"
        )
        parent_items = {
            item.item_id: item
            for item in parent.items
            if item.qc_status == "pass"
        }
        child_spaces = {
            (item.acoustic_space_id, item.room_id, item.scene_id, item.split)
            for item in child_items
        }
        parent_spaces = {
            (item.acoustic_space_id, item.room_id, item.scene_id, item.split)
            for item in parent_items.values()
        }
        lineage_valid = child_spaces == parent_spaces
        child_bank_root = release_root / variant.bank_path
        parent_variant = next(
            item
            for item in release.variants
            if item.variant_id == variant.parent_variant_id
        )
        parent_bank_root = release_root / parent_variant.bank_path
        for child_item in child_items:
            try:
                metadata = json.loads(
                    (child_bank_root / child_item.metadata_path).read_text(
                        encoding="utf-8"
                    )
                )
                lineage = metadata.get("m6_variant")
                if not isinstance(lineage, Mapping):
                    raise ValueError("child metadata has no variant lineage")
                parent_item = parent_items[str(lineage["parent_item_id"])]
                target_peak = float(lineage["target_peak_abs"])
                parent_audio, parent_sr = sf.read(
                    parent_bank_root / parent_item.rir_path,
                    always_2d=True,
                    dtype="float64",
                )
                child_audio, child_sr = sf.read(
                    child_bank_root / child_item.rir_path,
                    always_2d=True,
                    dtype="float64",
                )
                parent_peak = float(np.max(np.abs(parent_audio)))
                common_gain = target_peak / parent_peak
                expected_child = np.asarray(
                    parent_audio * common_gain,
                    dtype=np.float32,
                ).astype(np.float64)
                lineage_valid = bool(
                    lineage_valid
                    and lineage.get("transform")
                    == "one_common_item_gain_to_peak"
                    and lineage.get("parent_rir_sha256")
                    == parent_item.rir_sha256
                    and sha256_file(parent_bank_root / parent_item.rir_path)
                    == parent_item.rir_sha256
                    and parent_sr == child_sr
                    and parent_audio.shape == child_audio.shape
                    and parent_peak > 0.0
                    and 0.0 < target_peak <= 1.0
                    and np.array_equal(child_audio, expected_child)
                    and (
                        child_item.acoustic_space_id,
                        child_item.room_id,
                        child_item.scene_id,
                        child_item.split,
                    )
                    == (
                        parent_item.acoustic_space_id,
                        parent_item.room_id,
                        parent_item.scene_id,
                        parent_item.split,
                    )
                )
            except (
                OSError,
                RuntimeError,
                ValueError,
                KeyError,
                TypeError,
                json.JSONDecodeError,
            ):
                lineage_valid = False
        lineage_checks[variant.variant_id] = lineage_valid

    variants = {variant.variant_id: variant for variant in release.variants}
    recipe_checks: dict[str, bool] = {}
    ready_recipe_count = 0
    for recipe in release.recipes:
        if recipe.status == "blocked":
            recipe_checks[recipe.recipe_id] = bool(
                recipe.split_indexes is None and recipe.blockers
            )
            continue
        ready_recipe_count += 1
        valid = True
        for split in BANK_SPLITS:
            index = recipe.split_indexes[split]  # type: ignore[index]
            path = release_root / index.path
            expected = _recipe_rows(
                release_root, variants, recipe.variant_ids, split
            )
            try:
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                valid = bool(
                    valid
                    and rows == expected
                    and sha256_file(path) == index.sha256
                    and index.item_count == len(expected)
                    and len(expected) > 0
                )
            except (OSError, json.JSONDecodeError):
                valid = False
        recipe_checks[recipe.recipe_id] = valid
    acoustic_splits: dict[str, set[str]] = {}
    for manifest in manifests.values():
        for item in manifest.items:
            if item.qc_status == "pass":
                acoustic_splits.setdefault(item.acoustic_space_id, set()).add(item.split)
    checks = {
        "release_content_hash_matches": (
            release.release_sha256 == release.content_sha256()
        ),
        "all_variant_assets_and_distributions_audit": all(variant_checks.values()),
        "variant_lineage_preserves_identity_parent_hash_and_transform": all(
            lineage_checks.values()
        ),
        "all_ready_and_blocked_recipes_are_consistent": all(recipe_checks.values()),
        "at_least_one_ready_recipe_exists": ready_recipe_count > 0,
        "acoustic_spaces_remain_split_disjoint_across_variants": all(
            len(splits) == 1 for splits in acoustic_splits.values()
        ),
        "release_does_not_claim_production": release.release_status != "production",
    }
    return {
        "schema_version": "puresound.rir_bank_release_audit.v1",
        "release_id": release.release_id,
        "variant_checks": variant_checks,
        "lineage_checks": lineage_checks,
        "recipe_checks": recipe_checks,
        "checks": checks,
        "valid": bool(all(checks.values())),
        "ready_recipe_count": ready_recipe_count,
        "production_ready": False,
    }


__all__ = [
    "DEFAULT_RELEASE_MANIFEST_NAME",
    "RIR_BANK_DISTRIBUTION_SCHEMA_VERSION",
    "RIR_BANK_RELEASE_SCHEMA_VERSION",
    "RIRBankReleaseManifest",
    "ReleaseIndex",
    "ReleaseRecipe",
    "ReleaseVariant",
    "audit_m6_variant_release",
    "build_m6_variant_release",
    "compute_bank_distribution",
]
