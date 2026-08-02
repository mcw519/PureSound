#!/usr/bin/env python3
"""Build and validate the deterministic M6.1 production-bank contract fixture."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from puresound.audio.rir_bank import PreGeneratedRoomBank
from puresound.audio.rir_bank_manifest import (
    BANK_SPLITS,
    BankGeneratorProvenance,
    BankRendererProfile,
    BankSplitPolicy,
    RIRBankItem,
    RIRBankManifest,
    audit_rir_bank_manifest,
    canonical_json_sha256,
    canonicalize_float_wav_header,
    sha256_file,
    task_plan_rows,
    write_split_indexes,
)
from puresound.audio.rir_scene import SCENE_SCHEMA_VERSION


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "egs/rir_generation/exp/rir_realism/m6/rir_m6_bank_contract"
DEFAULT_OUTPUT_REPORT = (
    REPO_ROOT / "egs/rir_generation/phases/m6_bank/reports/m6_bank_contract_report.json"
)
DEFAULT_MANIFEST_NAME = "rir_bank_manifest.json"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _one_space_per_split(policy: BankSplitPolicy) -> dict[str, str]:
    result: dict[str, str] = {}
    index = 0
    while len(result) < len(BANK_SPLITS):
        identity = f"m6-fixture-acoustic-space-{index:04d}"
        result.setdefault(policy.assign(identity), identity)
        index += 1
        if index > 10000:
            raise RuntimeError("failed to find fixture identity for every split")
    return result


def _write_fixture_item(
    output_root: Path,
    *,
    split: str,
    acoustic_space_id: str,
    renderer_profile_id: str,
    sample_rate: int = 16000,
    frame_count: int = 2400,
) -> RIRBankItem:
    room_id = f"m6_{split}_room"
    item_id = f"{room_id}_000000"
    item_dir = output_root / room_id
    item_dir.mkdir(parents=True, exist_ok=True)
    rir_path = item_dir / f"{item_id}.wav"
    metadata_path = item_dir / f"{item_id}.json"
    distances = (0.65, 0.90, 2.25, 3.50, 4.60)
    labels = ("near_0", "near_1", "far_0", "far_1", "far_2")
    rir = np.zeros((frame_count, len(distances)), dtype=np.float32)
    for channel, distance in enumerate(distances):
        direct = int(round(distance / 343.0 * sample_rate))
        rir[direct, channel] = np.float32(1.0 / distance)
        rir[direct + 120, channel] = np.float32(0.12 / distance)
    sf.write(rir_path, rir, sample_rate, subtype="FLOAT")
    canonicalize_float_wav_header(rir_path)
    scene = {
        "schema_version": SCENE_SCHEMA_VERSION,
        "scene_id": item_id,
        "origin": "synthetic",
        "rt60": 0.42,
        "channel_map": [
            {
                "channel": channel,
                "label": label,
                "distance_m": distance,
            }
            for channel, (label, distance) in enumerate(zip(labels, distances))
        ],
    }
    metadata = {
        "sample_id": item_id,
        "room_id": room_id,
        "scene": scene,
        "fixture_scope": "M6.1 contract only; not acoustic evidence",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return RIRBankItem(
        item_id=item_id,
        room_id=room_id,
        acoustic_space_id=acoustic_space_id,
        scene_id=item_id,
        split=split,
        generation_seed=20260803 + BANK_SPLITS.index(split),
        origin="synthetic",
        renderer_profile_id=renderer_profile_id,
        signal_variant="physical",
        level_policy="calibrated",
        rir_path=rir_path.relative_to(output_root).as_posix(),
        metadata_path=metadata_path.relative_to(output_root).as_posix(),
        rir_sha256=sha256_file(rir_path),
        metadata_sha256=sha256_file(metadata_path),
        scene_sha256=canonical_json_sha256(scene),
        sample_rate=sample_rate,
        channel_count=len(distances),
        frame_count=frame_count,
        qc_status="pending",
    )


def build_fixture(output_root: Path) -> RIRBankManifest:
    """Write a deterministic three-split non-production M6.1 bank fixture."""

    output_root.mkdir(parents=True, exist_ok=True)
    policy = BankSplitPolicy(
        seed=20260803,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    renderer_config = {
        "low_backend": "analytic",
        "high_backend": "path-events-m4",
        "fdn_mixing_time_ms": 24.0,
        "output_mode": "calibrated",
        "fixture": True,
    }
    profile = BankRendererProfile(
        profile_id="m6-fixture-path-events-m4",
        renderer_id="puresound.hybrid_rir",
        renderer_version="M4-implementation",
        low_backend="analytic",
        high_backend="path-events-m4",
        scene_schema_version=SCENE_SCHEMA_VERSION,
        renderer_config_sha256=canonical_json_sha256(renderer_config),
        evidence_tier="development",
        calibration_report_sha256=sha256_file(
            REPO_ROOT / "egs/rir_generation/phases/m5_calibration/reports/m5_exit_report.json"
        ),
    )
    spaces = _one_space_per_split(policy)
    items = tuple(
        _write_fixture_item(
            output_root,
            split=split,
            acoustic_space_id=spaces[split],
            renderer_profile_id=profile.profile_id,
        )
        for split in BANK_SPLITS
    )
    generation_config = {
        "fixture": "M6.1",
        "sample_rate": 16000,
        "frame_count": 2400,
        "splits": list(BANK_SPLITS),
        "split_policy": policy.to_dict(),
    }
    generator = BankGeneratorProvenance(
        generator_id="egs.rir_generation.phases.m6_bank.scripts.validate_m6_bank_contract",
        generator_version="M6.1",
        code_revision="m6.1-contract-fixture",
        config_sha256=canonical_json_sha256(generation_config),
        task_plan_sha256=canonical_json_sha256(task_plan_rows(items)),
        seed=20260803,
    )
    split_indexes = write_split_indexes(output_root, items)
    manifest = RIRBankManifest(
        bank_id="puresound-m6-contract-fixture",
        release_status="draft",
        split_policy=policy,
        generator=generator,
        renderer_profiles=(profile,),
        items=items,
        split_indexes=split_indexes,
    ).with_content_sha256()
    (output_root / DEFAULT_MANIFEST_NAME).write_text(
        manifest.to_json() + "\n",
        encoding="utf-8",
    )
    return manifest


def _with_items(
    manifest: RIRBankManifest,
    items: tuple[RIRBankItem, ...],
) -> RIRBankManifest:
    return replace(manifest, items=items, manifest_sha256=None).with_content_sha256()


def build_report(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    manifest = build_fixture(output_root)
    round_trip = RIRBankManifest.from_json(manifest.to_json())
    audit = audit_rir_bank_manifest(round_trip, output_root)

    train_item = next(item for item in manifest.items if item.split == "train")
    test_index = next(
        index for index, item in enumerate(manifest.items) if item.split == "test"
    )
    leaked_items = list(manifest.items)
    leaked_items[test_index] = replace(
        leaked_items[test_index],
        acoustic_space_id=train_item.acoustic_space_id,
    )
    leaked = _with_items(manifest, tuple(leaked_items))
    leaked_audit = audit_rir_bank_manifest(leaked, output_root)

    wrong_hash_items = list(manifest.items)
    wrong_hash_items[0] = replace(wrong_hash_items[0], rir_sha256="0" * 64)
    wrong_hash = _with_items(manifest, tuple(wrong_hash_items))
    wrong_hash_audit = audit_rir_bank_manifest(wrong_hash, output_root)

    tampered_payload = manifest.to_dict()
    tampered_payload["bank_id"] = "tampered-without-rehash"
    tampered = RIRBankManifest.from_dict(tampered_payload)
    tampered_audit = audit_rir_bank_manifest(tampered, output_root)

    false_production = replace(
        manifest,
        release_status="production",
        manifest_sha256=None,
    ).with_content_sha256()
    false_production_audit = audit_rir_bank_manifest(false_production, output_root)

    unsafe_path_rejected = False
    try:
        replace(manifest.items[0], rir_path="../escape.wav")
    except ValueError:
        unsafe_path_rejected = True

    unsplit_reader_rejected = False
    try:
        PreGeneratedRoomBank(str(output_root))
    except ValueError:
        unsplit_reader_rejected = True
    split_reader = PreGeneratedRoomBank(str(output_root), split="train")
    deterministic_rebuild = replace(
        manifest,
        manifest_sha256=None,
    ).with_content_sha256()
    checks = {
        "strict_schema_round_trip": round_trip.to_dict() == manifest.to_dict(),
        "canonical_manifest_digest_is_deterministic": (
            deterministic_rebuild.manifest_sha256 == manifest.manifest_sha256
        ),
        "all_three_splits_present": audit["checks"][
            "train_validation_test_items_present"
        ],
        "split_assignment_is_deterministic": audit["checks"][
            "deterministic_acoustic_space_assignments_match"
        ],
        "acoustic_space_leakage_is_rejected": (
            not leaked_audit["checks"]["acoustic_spaces_are_split_disjoint"]
        ),
        "room_id_leakage_is_absent": audit["checks"][
            "room_ids_are_split_disjoint"
        ],
        "asset_hash_tamper_is_rejected": (
            not wrong_hash_audit["checks"]["all_asset_hashes_match"]
        ),
        "manifest_tamper_is_rejected": (
            not tampered_audit["checks"]["manifest_content_hash_matches"]
        ),
        "scene_and_audio_contracts_pass": (
            audit["checks"]["scene_hashes_match_metadata"]
            and audit["checks"]["audio_headers_match_manifest"]
            and audit["checks"]["metadata_identity_matches_manifest"]
        ),
        "unsafe_relative_path_is_rejected": unsafe_path_rejected,
        "development_renderer_cannot_claim_production": (
            not false_production_audit["checks"][
                "production_claim_is_evidence_backed"
            ]
            and not false_production_audit["ready_for_production"]
        ),
        "reader_is_backward_compatible_but_m6_split_safe": (
            unsplit_reader_rejected
            and len(split_reader)
            == sum(item.split == "train" for item in manifest.items)
        ),
    }
    passed = bool(all(checks.values()) and audit["ready_for_m6_bank_generation"])
    return {
        "schema_version": "puresound.m6_bank_contract_validation.v1",
        "milestone": "M6.1",
        "scope": "bank_contract_and_room_disjoint_release_provenance",
        "fixture": {
            "root": _display_path(output_root),
            "manifest": _display_path(output_root / DEFAULT_MANIFEST_NAME),
            "explicitly_not_acoustic_or_production_evidence": True,
        },
        "contract": {
            "manifest_schema": manifest.schema_version,
            "manifest_sha256": manifest.manifest_sha256,
            "split_policy": manifest.split_policy.policy_id,
            "release_status": manifest.release_status,
            "renderer_evidence_tier": manifest.renderer_profiles[0].evidence_tier,
        },
        "audit": audit,
        "negative_controls": {
            "cross_split_acoustic_space": leaked_audit["checks"],
            "wrong_asset_hash": wrong_hash_audit["checks"],
            "manifest_without_rehash": tampered_audit["checks"],
            "false_production_release": false_production_audit["checks"],
        },
        "checks": checks,
        "exit": {
            "passed": passed,
            "m6_1_bank_contract_complete": passed,
            "production_bank_complete": False,
            "next_milestone": "M6.2 reproducible generator integration",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    args = parser.parse_args()
    report = build_report(args.output_root)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for name, passed in report["checks"].items():
        print(f"{name}\t{'PASS' if passed else 'FAIL'}")
    print(
        "# M6.1 bank contract: "
        f"{'PASS' if report['exit']['passed'] else 'FAIL'}"
    )
    print("# M6 production bank: OPEN")
    print(f"# wrote {args.output_report}")
    return 0 if report["exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
