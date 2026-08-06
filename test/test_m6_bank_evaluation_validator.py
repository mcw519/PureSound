import json

import pytest

from egs.rir_generation.phases.m6_bank.scripts import validate_m6_bank_evaluation

# End-to-end evidence-chain validator: builds, QCs and releases a real bank, so it
# runs for tens of seconds. Excluded by `run_repo_checks.py --suite standard`.
pytestmark = pytest.mark.slow


def test_m6_5_bank_evaluation_validator(tmp_path):
    report = validate_m6_bank_evaluation.build_report(tmp_path)

    assert report["milestone"] == "M6.5"
    assert report["exit"]["passed"] is True
    assert report["exit"]["m6_5_bank_evaluation_implementation_complete"] is True
    assert report["exit"]["m6_5_empirical_exit_complete"] is False
    assert report["exit"]["production_bank_complete"] is False
    assert all(report["checks"].values())
    assert report["evaluation"]["implementation_exit"]["passed"] is True
    assert report["evaluation"]["empirical_exit"]["passed"] is False
    assert report["evaluation"]["production_enablement"]["ready"] is False
    json.dumps(report, allow_nan=False)
