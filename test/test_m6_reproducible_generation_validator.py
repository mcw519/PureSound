import json

import pytest

from egs.rir_generation.phases.m6_bank.scripts import (
    validate_m6_reproducible_generation,
)

# End-to-end evidence-chain validator: builds, QCs and releases a real bank, so it
# runs for tens of seconds. Excluded by `run_repo_checks.py --suite standard`.
pytestmark = pytest.mark.slow


def test_m6_2_actual_generator_is_reproducible_and_resume_safe(tmp_path):
    report = validate_m6_reproducible_generation.build_report(tmp_path)

    assert report["milestone"] == "M6.2"
    assert report["exit"]["passed"] is True
    assert (
        report["exit"]["m6_2_reproducible_generator_integration_complete"]
        is True
    )
    assert report["exit"]["production_bank_complete"] is False
    assert all(report["checks"].values())
    assert (
        report["runs"]["serial"]["manifest_sha256"]
        == report["runs"]["fresh_repeat"]["manifest_sha256"]
        == report["runs"]["parallel"]["manifest_sha256"]
        == report["runs"]["resumed_after_tamper"]["manifest_sha256"]
    )
    assert report["checks"]["fresh_repeat_generation_audit_passed"] is True
    assert (
        report["checks"][
            "serial_fresh_repeat_manifest_and_item_hashes_match"
        ]
        is True
    )
    assert report["runs"]["resumed_after_tamper"]["generation_run"][
        "items_generated"
    ] == 1
    assert report["runs"]["changed_config"]["generation_run"][
        "items_generated"
    ] == 6
    json.dumps(report, allow_nan=False)
