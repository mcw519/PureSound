import json
from dataclasses import replace

import pytest

from egs.rir_generation.phases.m6_bank.scripts.validate_m6_bank_contract import build_fixture
from puresound.audio.rir_bank_manifest import (
    RIRBankManifest,
    canonical_json_sha256,
)
from puresound.audio.rir_bank_qc import (
    RIRBankQCPolicy,
    audit_rir_bank_qc_release,
    run_rir_bank_qc,
)


def test_qc_policy_is_versioned_content_addressed_and_json_safe():
    policy = RIRBankQCPolicy()
    round_trip = RIRBankQCPolicy.from_dict(policy.to_dict())

    assert round_trip == policy
    assert len(policy.content_sha256()) == 64
    json.dumps(policy.to_dict(), allow_nan=False)
    with pytest.raises(ValueError, match="within"):
        replace(policy, minimum_decay_coverage_fraction=1.1)
    with pytest.raises(ValueError, match="search window"):
        replace(
            policy,
            direct_search_after_expected_ms=policy.maximum_arrival_error_ms,
        )


def test_sparse_contract_fixture_is_quarantined_without_deleting_assets(tmp_path):
    original = build_fixture(tmp_path)
    original_assets = {
        item.item_id: (item.rir_sha256, item.metadata_sha256)
        for item in original.items
    }

    summary = run_rir_bank_qc(tmp_path)
    repeated_summary = run_rir_bank_qc(tmp_path)
    completed = RIRBankManifest.from_json(
        (tmp_path / "rir_bank_manifest.json").read_text(encoding="utf-8")
    )
    audit = audit_rir_bank_qc_release(tmp_path)

    assert summary["counts"]["passed"] == 0
    assert repeated_summary["summary_sha256"] == summary["summary_sha256"]
    assert summary["counts"]["quarantined"] == len(original.items)
    assert summary["release"]["status"] == "draft"
    assert summary["release"]["ready_for_candidate"] is False
    assert all(item.qc_status == "fail" for item in completed.items)
    assert all(item.qc_report_path for item in completed.items)
    assert {
        item.item_id: (item.rir_sha256, item.metadata_sha256)
        for item in completed.items
    } == original_assets
    assert audit["valid"] is True

    summary_path = tmp_path / "rir_bank_qc_summary.json"
    unsafe_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    unsafe_summary["candidate_indexes"]["train"]["path"] = "../outside.jsonl"
    unsafe_summary.pop("summary_sha256")
    unsafe_summary["summary_sha256"] = canonical_json_sha256(unsafe_summary)
    summary_path.write_text(json.dumps(unsafe_summary), encoding="utf-8")
    unsafe_audit = audit_rir_bank_qc_release(tmp_path)
    assert unsafe_audit["valid"] is False
    assert unsafe_audit["checks"]["candidate_indexes_match_passed_items"] is False


def test_completed_qc_requires_both_safe_report_path_and_hash(tmp_path):
    manifest = build_fixture(tmp_path)
    item = manifest.items[0]

    with pytest.raises(ValueError, match="single path component"):
        replace(item, item_id="../../../outside")

    with pytest.raises(ValueError, match="completed QC"):
        replace(item, qc_status="pass", qc_report_sha256="0" * 64)
    with pytest.raises(ValueError, match="pending QC"):
        replace(
            item,
            qc_report_path="qc/items/report.json",
            qc_report_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="safe bank-relative"):
        replace(
            item,
            qc_status="fail",
            qc_report_path="../report.json",
            qc_report_sha256="0" * 64,
        )
