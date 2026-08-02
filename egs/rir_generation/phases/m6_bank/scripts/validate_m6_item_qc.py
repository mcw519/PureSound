#!/usr/bin/env python3
"""Validate M6.3 per-item acoustic QC, admission, and quarantine semantics."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import soundfile as sf

from egs.rir_generation.phases.m6_bank.scripts.validate_m6_reproducible_generation import _run_generator
from puresound.audio.rir_bank import PreGeneratedRoomBank
from puresound.audio.rir_bank_manifest import (
    RIRBankItem,
    RIRBankManifest,
    audit_rir_bank_manifest,
    sha256_file,
)
from puresound.audio.rir_bank_qc import (
    audit_rir_bank_qc_release,
    run_rir_bank_qc,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "egs/rir_generation/exp/rir_realism/m6/rir_m6_item_qc"
DEFAULT_OUTPUT_REPORT = REPO_ROOT / "egs/rir_generation/phases/m6_bank/reports/m6_item_qc_report.json"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _reset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _report_hashes(root: Path, manifest: RIRBankManifest) -> dict[str, str]:
    return {
        item.item_id: str(item.qc_report_sha256)
        for item in manifest.items
        if item.qc_report_path is not None
        and (root / item.qc_report_path).is_file()
    }


def _read_reports(root: Path, manifest: RIRBankManifest) -> dict[str, dict[str, Any]]:
    return {
        item.item_id: json.loads(
            (root / str(item.qc_report_path)).read_text(encoding="utf-8")
        )
        for item in manifest.items
    }


def _rewrite_item(
    root: Path,
    item: RIRBankItem,
    mutate: Callable[[np.ndarray, dict[str, Any], int], None],
) -> RIRBankItem:
    rir_path = root / item.rir_path
    metadata_path = root / item.metadata_path
    rir, sample_rate = sf.read(rir_path, always_2d=True, dtype="float64")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutate(rir, metadata, int(sample_rate))
    sf.write(rir_path, rir, sample_rate, subtype="FLOAT")
    rir_sha256 = sha256_file(rir_path)
    m6 = metadata.get("m6")
    if isinstance(m6, dict):
        m6["rir_sha256"] = rir_sha256
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return replace(
        item,
        rir_sha256=rir_sha256,
        metadata_sha256=sha256_file(metadata_path),
    )


def _build_negative_controls(root: Path) -> dict[str, str]:
    manifest_path = root / "rir_bank_manifest.json"
    manifest = RIRBankManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    by_split = {
        split: [item for item in manifest.items if item.split == split]
        for split in ("train", "validation", "test")
    }
    silent_id = by_split["train"][0].item_id
    prearrival_id = by_split["validation"][0].item_id
    sparse_id = by_split["test"][0].item_id
    late_arrival_id = by_split["train"][1].item_id

    def silent(rir: np.ndarray, _metadata: dict[str, Any], _sr: int) -> None:
        rir.fill(0.0)

    def prearrival(rir: np.ndarray, _metadata: dict[str, Any], _sr: int) -> None:
        rir[0, 0] = max(0.5, float(np.max(np.abs(rir[:, 0]))))

    def sparse(rir: np.ndarray, metadata: dict[str, Any], sr: int) -> None:
        rir.fill(0.0)
        scene = metadata["scene"]
        sound_speed = float(scene.get("environment", {}).get("sound_speed_m_s", 343.0))
        for channel in scene["channel_map"]:
            index = int(channel["channel"])
            direct = int(math.floor(float(channel["distance_m"]) / sound_speed * sr))
            rir[direct, index] = 0.1

    def late_arrival(rir: np.ndarray, _metadata: dict[str, Any], sr: int) -> None:
        delay = max(1, int(round(3.0e-3 * sr)))
        original = rir.copy()
        rir.fill(0.0)
        rir[delay:] = original[:-delay]

    mutations = {
        silent_id: silent,
        prearrival_id: prearrival,
        sparse_id: sparse,
        late_arrival_id: late_arrival,
    }
    updated_items = tuple(
        _rewrite_item(root, item, mutations[item.item_id])
        if item.item_id in mutations
        else item
        for item in manifest.items
    )
    updated = replace(
        manifest,
        items=updated_items,
        manifest_sha256=None,
    ).with_content_sha256()
    manifest_path.write_text(updated.to_json() + "\n", encoding="utf-8")
    if not audit_rir_bank_manifest(updated, root)["ready_for_m6_bank_generation"]:
        raise RuntimeError("negative-control bank no longer satisfies the M6 asset contract")
    return {
        "silent": silent_id,
        "prearrival": prearrival_id,
        "sparse_late_field": sparse_id,
        "late_arrival": late_arrival_id,
    }


def build_report(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    raw_root = output_root / "raw_generator_output"
    positive_a_root = output_root / "positive_a"
    positive_b_root = output_root / "positive_b"
    negative_root = output_root / "negative_controls"
    tampered_root = output_root / "tampered_qc_report"
    _reset(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    generated = _run_generator(raw_root, workers=1)
    for destination in (positive_a_root, positive_b_root, negative_root):
        shutil.copytree(raw_root, destination)

    positive_a_summary = run_rir_bank_qc(positive_a_root)
    positive_b_summary = run_rir_bank_qc(positive_b_root)
    positive_a_manifest = RIRBankManifest.from_json(
        (positive_a_root / "rir_bank_manifest.json").read_text(encoding="utf-8")
    )
    positive_b_manifest = RIRBankManifest.from_json(
        (positive_b_root / "rir_bank_manifest.json").read_text(encoding="utf-8")
    )
    positive_a_audit = audit_rir_bank_qc_release(positive_a_root)
    positive_b_audit = audit_rir_bank_qc_release(positive_b_root)
    positive_reports = _read_reports(positive_a_root, positive_a_manifest)

    negative_ids = _build_negative_controls(negative_root)
    negative_summary = run_rir_bank_qc(negative_root)
    negative_manifest = RIRBankManifest.from_json(
        (negative_root / "rir_bank_manifest.json").read_text(encoding="utf-8")
    )
    negative_audit = audit_rir_bank_qc_release(negative_root)
    negative_reports = _read_reports(negative_root, negative_manifest)

    positive_reader_counts = {
        split: len(PreGeneratedRoomBank(str(positive_a_root), split=split))
        for split in ("train", "validation", "test")
    }
    expected_positive_counts = {
        split: positive_a_summary["candidate_indexes"][split]["item_count"]
        for split in ("train", "validation", "test")
    }
    failed_reader_excluded = False
    try:
        PreGeneratedRoomBank(str(negative_root), split="test")
    except FileNotFoundError:
        failed_reader_excluded = True
    debug_reader_includes_failed = (
        len(
            PreGeneratedRoomBank(
                str(negative_root),
                split="test",
                include_failed_qc=True,
            )
        )
        == 1
    )

    shutil.copytree(positive_a_root, tampered_root)
    tampered_manifest = RIRBankManifest.from_json(
        (tampered_root / "rir_bank_manifest.json").read_text(encoding="utf-8")
    )
    tampered_report_path = tampered_root / str(tampered_manifest.items[0].qc_report_path)
    with tampered_report_path.open("ab") as handle:
        handle.write(b"m6.3-qc-report-tamper")
    tampered_audit = audit_rir_bank_qc_release(tampered_root)

    silent_reasons = negative_reports[negative_ids["silent"]]["failure_reasons"]
    prearrival_reasons = negative_reports[negative_ids["prearrival"]][
        "failure_reasons"
    ]
    sparse_reasons = negative_reports[negative_ids["sparse_late_field"]][
        "failure_reasons"
    ]
    late_arrival_reasons = negative_reports[negative_ids["late_arrival"]][
        "failure_reasons"
    ]
    positive_spatial = [report["spatial"]["status"] for report in positive_reports.values()]
    checks = {
        "actual_m6_generator_output_passes_source_audit": generated["audit"][
            "ready_for_m6_bank_generation"
        ],
        "positive_items_all_pass_and_candidate_splits_are_complete": (
            positive_a_summary["counts"]["passed"] == 6
            and positive_a_summary["counts"]["quarantined"] == 0
            and positive_a_summary["release"]["ready_for_candidate"]
            and positive_reader_counts == expected_positive_counts
        ),
        "qc_reports_and_indexes_are_deterministic": (
            positive_a_manifest.manifest_sha256
            == positive_b_manifest.manifest_sha256
            and positive_a_summary["summary_sha256"]
            == positive_b_summary["summary_sha256"]
            and _report_hashes(positive_a_root, positive_a_manifest)
            == _report_hashes(positive_b_root, positive_b_manifest)
        ),
        "positive_qc_release_audits_pass": (
            positive_a_audit["valid"] and positive_b_audit["valid"]
        ),
        "source_indexed_channels_do_not_claim_spatial_metrics": (
            positive_spatial == ["not_applicable"] * len(positive_spatial)
        ),
        "silent_item_is_quarantined": (
            "audio_is_non_silent" in silent_reasons
            and next(
                item for item in negative_manifest.items if item.item_id == negative_ids["silent"]
            ).qc_status
            == "fail"
        ),
        "prearrival_item_is_quarantined": any(
            reason.endswith("prearrival_energy") for reason in prearrival_reasons
        ),
        "sparse_late_field_is_quarantined": (
            "decay_fit_coverage" in sparse_reasons
            and any(reason.endswith("insufficient_tail_energy") for reason in sparse_reasons)
            and any(
                reason.endswith("insufficient_late_echo_density")
                for reason in sparse_reasons
            )
        ),
        "late_direct_arrival_is_quarantined": any(
            reason.endswith("direct_arrival_timing")
            for reason in late_arrival_reasons
        ),
        "negative_controls_are_only_in_quarantine": (
            negative_summary["counts"]["quarantined"] == 4
            and negative_summary["quarantine_index"]["item_count"] == 4
            and negative_audit["valid"]
        ),
        "empty_candidate_split_fails_closed": (
            negative_summary["candidate_indexes"]["test"]["item_count"] == 0
            and not negative_summary["release"]["ready_for_candidate"]
            and negative_summary["release"]["status"] == "draft"
            and failed_reader_excluded
            and debug_reader_includes_failed
        ),
        "candidate_is_explicitly_not_production": (
            positive_a_summary["release"]["status"] == "candidate"
            and not positive_a_summary["release"]["ready_for_production"]
        ),
        "tampered_qc_report_fails_content_audit": (
            not tampered_audit["valid"]
            and not tampered_audit["checks"]["item_report_hashes_and_statuses_match"]
        ),
    }
    passed = bool(all(checks.values()))
    return {
        "schema_version": "puresound.m6_item_qc_validation.v1",
        "milestone": "M6.3",
        "scope": "per_item_physical_qc_candidate_admission_and_quarantine",
        "fixture": {
            "source": "actual M6.2 generator output",
            "room_count": 6,
            "sample_rate": 8000,
            "duration_s": 0.2,
            "positive_roots": [
                _display_path(positive_a_root),
                _display_path(positive_b_root),
            ],
            "negative_root": _display_path(negative_root),
            "negative_item_ids": negative_ids,
            "explicitly_not_measured_or_production_evidence": True,
        },
        "positive": {
            "manifest_sha256": positive_a_manifest.manifest_sha256,
            "summary_sha256": positive_a_summary["summary_sha256"],
            "counts": positive_a_summary["counts"],
            "audit": positive_a_audit,
        },
        "negative": {
            "counts": negative_summary["counts"],
            "failure_counts": negative_summary["failure_counts"],
            "release": negative_summary["release"],
            "audit": negative_audit,
        },
        "checks": checks,
        "exit": {
            "passed": passed,
            "m6_3_item_qc_and_quarantine_complete": passed,
            "production_bank_complete": False,
            "next_milestone": "M6.4 distribution and variant release",
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
    print(f"# M6.3 item QC/quarantine: {'PASS' if report['exit']['passed'] else 'FAIL'}")
    print("# M6 production bank: OPEN")
    print(f"# wrote {args.output_report}")
    return 0 if report["exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
