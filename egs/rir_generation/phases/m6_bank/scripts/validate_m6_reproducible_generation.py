#!/usr/bin/env python3
"""Validate M6.2 fresh/parallel/resume/config-isolation generation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from egs.rir_generation.generate_hybrid_rir import _m6_task_seed
from puresound.audio.rir.bank.loader import PreGeneratedRoomBank
from puresound.audio.rir.bank.schema import RIRBankManifest


REPO_ROOT = Path(__file__).resolve().parents[5]
GENERATOR = REPO_ROOT / "egs/rir_generation/generate_hybrid_rir.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "egs/rir_generation/exp/rir_realism/m6/rir_m6_reproducible_generation"
DEFAULT_OUTPUT_REPORT = (
    REPO_ROOT / "egs/rir_generation/phases/m6_bank/reports/m6_reproducible_generation_report.json"
)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _common_args(
    output: Path,
    *,
    duration: float = 0.2,
    code_revision: str = "m6.2-validation-revision",
) -> list[str]:
    return [
        str(GENERATOR),
        "--output-dir",
        str(output),
        "--n-rooms",
        "6",
        "--rir-per-room",
        "1",
        "--seed",
        "7331",
        "--sample-rate",
        "8000",
        "--duration",
        str(duration),
        "--crossover-hz",
        "800",
        "--low-fmin-hz",
        "40",
        "--low-fmax-hz",
        "1000",
        "--scene-version",
        "v1",
        "--room-type",
        "office",
        "--no-record-realized-metrics",
        "--low-backend",
        "analytic",
        "--high-backend",
        "pyroomacoustics",
        "--pra-max-order",
        "1",
        "--pra-n-rays",
        "2000",
        "--room-x",
        "4",
        "4",
        "--room-y",
        "4",
        "4",
        "--room-z",
        "2.8",
        "2.8",
        "--rt60",
        "0.3",
        "0.3",
        "--obstacles",
        "0",
        "0",
        "--near-dist",
        "0.4",
        "0.6",
        "--far-dist",
        "1.2",
        "1.8",
        "--emit-m6-manifest",
        "--m6-bank-id",
        "puresound-m6-reproducibility-fixture",
        "--m6-code-revision",
        code_revision,
        "--m6-train-fraction",
        "0.5",
        "--m6-validation-fraction",
        "0.25",
        "--m6-test-fraction",
        "0.25",
    ]


def _run_generator(
    output: Path,
    *,
    workers: int,
    resume: bool = False,
    duration: float = 0.2,
    code_revision: str = "m6.2-validation-revision",
) -> dict[str, Any]:
    command = [
        sys.executable,
        *_common_args(
            output,
            duration=duration,
            code_revision=code_revision,
        ),
    ]
    command.extend(["--num-workers", str(workers)])
    if resume:
        command.append("--resume")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment.setdefault("MPLCONFIGDIR", "/tmp/puresound-m6-matplotlib")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    manifest_path = output / "rir_bank_manifest.json"
    audit_path = output / "rir_bank_generation_audit.json"
    return {
        "stdout_summary": [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("[generate_hybrid_rir]")
        ],
        "manifest": RIRBankManifest.from_json(
            manifest_path.read_text(encoding="utf-8")
        ),
        "audit": json.loads(audit_path.read_text(encoding="utf-8")),
        "manifest_path": manifest_path,
        "audit_path": audit_path,
    }


def _item_content(manifest: RIRBankManifest) -> dict[str, tuple[str, str]]:
    return {
        item.item_id: (item.rir_sha256, item.metadata_sha256)
        for item in manifest.items
    }


def _reset_output(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def build_report(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    serial_root = output_root / "serial"
    fresh_repeat_root = output_root / "fresh_repeat"
    parallel_root = output_root / "parallel"
    resumed_root = output_root / "resumed"
    config_change_root = output_root / "config_change"
    code_change_root = output_root / "code_change"
    _reset_output(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    serial = _run_generator(serial_root, workers=1)
    fresh_repeat = _run_generator(fresh_repeat_root, workers=1)
    parallel = _run_generator(parallel_root, workers=2)

    shutil.copytree(serial_root, resumed_root)
    tampered_item = serial["manifest"].items[0]
    with (resumed_root / tampered_item.rir_path).open("ab") as handle:
        handle.write(b"m6.2-integrity-negative-control")
    resumed = _run_generator(resumed_root, workers=2, resume=True)

    shutil.copytree(serial_root, config_change_root)
    config_change = _run_generator(
        config_change_root,
        workers=1,
        resume=True,
        duration=0.22,
    )
    shutil.copytree(serial_root, code_change_root)
    code_change = _run_generator(
        code_change_root,
        workers=2,
        resume=True,
        code_revision="m6.2-validation-revision-next",
    )

    serial_manifest = serial["manifest"]
    fresh_repeat_manifest = fresh_repeat["manifest"]
    parallel_manifest = parallel["manifest"]
    resumed_manifest = resumed["manifest"]
    changed_manifest = config_change["manifest"]
    code_changed_manifest = code_change["manifest"]
    alternate_seeds = {
        item.item_id: _m6_task_seed(7332, item.item_id)
        for item in serial_manifest.items
    }
    unsplit_reader_rejected = False
    try:
        PreGeneratedRoomBank(str(serial_root))
    except ValueError:
        unsplit_reader_rejected = True
    split_reader_counts = {
        split: len(PreGeneratedRoomBank(str(serial_root), split=split))
        for split in ("train", "validation", "test")
    }
    checks = {
        "serial_generation_audit_passed": serial["audit"][
            "ready_for_m6_bank_generation"
        ],
        "fresh_repeat_generation_audit_passed": fresh_repeat["audit"][
            "ready_for_m6_bank_generation"
        ],
        "serial_fresh_repeat_manifest_and_item_hashes_match": (
            serial_manifest.manifest_sha256
            == fresh_repeat_manifest.manifest_sha256
            and _item_content(serial_manifest)
            == _item_content(fresh_repeat_manifest)
        ),
        "parallel_generation_audit_passed": parallel["audit"][
            "ready_for_m6_bank_generation"
        ],
        "serial_parallel_manifest_hashes_match": (
            serial_manifest.manifest_sha256 == parallel_manifest.manifest_sha256
        ),
        "serial_parallel_item_content_hashes_match": (
            _item_content(serial_manifest) == _item_content(parallel_manifest)
        ),
        "serial_parallel_task_plan_hashes_match": (
            serial_manifest.generator.task_plan_sha256
            == parallel_manifest.generator.task_plan_sha256
        ),
        "all_splits_are_present_and_disjoint": all(
            serial["audit"]["checks"][name]
            for name in (
                "train_validation_test_items_present",
                "deterministic_acoustic_space_assignments_match",
                "acoustic_spaces_are_split_disjoint",
                "room_ids_are_split_disjoint",
            )
        )
        and unsplit_reader_rejected
        and split_reader_counts == serial["audit"]["split_item_counts"],
        "tampered_item_is_the_only_resume_regeneration": (
            resumed["audit"]["generation_run"]["items_generated"] == 1
            and resumed["audit"]["generation_run"]["items_skipped_as_complete"]
            == len(serial_manifest.items) - 1
        ),
        "resume_restores_exact_manifest_and_assets": (
            resumed_manifest.manifest_sha256 == serial_manifest.manifest_sha256
            and _item_content(resumed_manifest) == _item_content(serial_manifest)
        ),
        "config_change_invalidates_every_old_item": (
            config_change["audit"]["generation_run"]["items_generated"]
            == len(serial_manifest.items)
            and config_change["audit"]["generation_run"][
                "items_skipped_as_complete"
            ]
            == 0
        ),
        "config_change_produces_new_manifest_and_frame_count": (
            changed_manifest.manifest_sha256 != serial_manifest.manifest_sha256
            and changed_manifest.generator.config_sha256
            != serial_manifest.generator.config_sha256
            and {item.frame_count for item in changed_manifest.items} == {1760}
        ),
        "code_revision_change_invalidates_every_old_item": (
            code_change["audit"]["generation_run"]["items_generated"]
            == len(serial_manifest.items)
            and code_change["audit"]["generation_run"][
                "items_skipped_as_complete"
            ]
            == 0
        ),
        "code_revision_change_is_recorded_in_manifest": (
            code_changed_manifest.generator.code_revision
            == "m6.2-validation-revision-next"
            and code_changed_manifest.manifest_sha256
            != serial_manifest.manifest_sha256
        ),
        "changed_seed_changes_every_item_seed": all(
            alternate_seeds[item.item_id] != item.generation_seed
            for item in serial_manifest.items
        ),
        "split_indexes_and_task_plan_are_content_addressed": all(
            serial["audit"]["checks"][name]
            for name in (
                "split_indexes_match_manifest_items",
                "task_plan_hash_matches_manifest_items",
                "manifest_content_hash_matches",
            )
        ),
        "release_remains_draft_and_not_production_ready": (
            serial_manifest.release_status == "draft"
            and all(
                profile.evidence_tier == "development"
                for profile in serial_manifest.renderer_profiles
            )
            and not serial["audit"]["ready_for_production"]
        ),
    }
    passed = bool(all(checks.values()))
    runs = {
        name: {
            "root": _display_path(root),
            "manifest": _display_path(result["manifest_path"]),
            "audit": _display_path(result["audit_path"]),
            "manifest_sha256": result["manifest"].manifest_sha256,
            "generation_run": result["audit"]["generation_run"],
            "stdout_summary": result["stdout_summary"],
        }
        for name, root, result in (
            ("serial", serial_root, serial),
            ("fresh_repeat", fresh_repeat_root, fresh_repeat),
            ("parallel", parallel_root, parallel),
            ("resumed_after_tamper", resumed_root, resumed),
            ("changed_config", config_change_root, config_change),
            ("changed_code_revision", code_change_root, code_change),
        )
    }
    return {
        "schema_version": "puresound.m6_reproducible_generation_validation.v1",
        "milestone": "M6.2",
        "scope": "actual_generator_fresh_parallel_resume_and_config_isolation",
        "fixture": {
            "room_count": 6,
            "rir_per_room": 1,
            "sample_rate": 8000,
            "duration_s": 0.2,
            "low_backend": "analytic",
            "high_backend": "pyroomacoustics",
            "pyroomacoustics_ray_count": 2000,
            "explicitly_not_acoustic_or_production_evidence": True,
        },
        "runs": runs,
        "checks": checks,
        "exit": {
            "passed": passed,
            "m6_2_reproducible_generator_integration_complete": passed,
            "production_bank_complete": False,
            "next_milestone": "M6.3 per-item QC and quarantine",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    args = parser.parse_args(argv)
    report = build_report(args.output_root)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for name, passed in report["checks"].items():
        print(f"{name}\t{'PASS' if passed else 'FAIL'}")
    print(
        "# M6.2 reproducible generator integration: "
        f"{'PASS' if report['exit']['passed'] else 'FAIL'}"
    )
    print("# M6 production bank: OPEN")
    print(f"# wrote {args.output_report}")
    return 0 if report["exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
