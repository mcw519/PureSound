import json

import pytest

from egs.rir_generation.phases.m6_bank.scripts import validate_m6_production_decision

# End-to-end evidence-chain validator: builds, QCs and releases a real bank, so it
# runs for tens of seconds. Excluded by `run_repo_checks.py --suite standard`.
pytestmark = pytest.mark.slow


def test_m6_6_production_decision_validator(tmp_path):
    report = validate_m6_production_decision.build_report(tmp_path)

    assert report["milestone"] == "M6.6"
    assert report["exit"]["passed"] is True
    assert report["exit"]["m6_6_production_decision_implementation_complete"] is True
    assert report["exit"]["production_promotion_complete"] is False
    assert all(report["checks"].values())
    assert report["decision"]["decision"] == "blocked"
    assert report["decision"]["production_ready"] is False
    assert report["decision"]["blockers"]
    json.dumps(report, allow_nan=False)
