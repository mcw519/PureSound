import json

import pytest

from egs.rir_generation.phases.m6_bank.scripts import validate_m6_item_qc

# End-to-end evidence-chain validator: builds, QCs and releases a real bank, so it
# runs for tens of seconds. Excluded by `run_repo_checks.py --suite standard`.
pytestmark = pytest.mark.slow


def test_m6_3_item_qc_and_quarantine_validator(tmp_path):
    report = validate_m6_item_qc.build_report(tmp_path)

    assert report["milestone"] == "M6.3"
    assert report["exit"]["passed"] is True
    assert report["exit"]["m6_3_item_qc_and_quarantine_complete"] is True
    assert report["exit"]["production_bank_complete"] is False
    assert all(report["checks"].values())
    assert report["positive"]["counts"]["passed"] == 6
    assert report["negative"]["counts"]["quarantined"] == 4
    assert report["negative"]["release"]["ready_for_candidate"] is False
    json.dumps(report, allow_nan=False)
