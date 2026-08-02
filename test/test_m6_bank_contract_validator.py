import json

from egs.rir_generation.phases.m6_bank.scripts import validate_m6_bank_contract


def test_m6_1_bank_contract_validator_passes_without_production_claim(tmp_path):
    report = validate_m6_bank_contract.build_report(tmp_path)

    assert report["milestone"] == "M6.1"
    assert report["exit"]["passed"] is True
    assert report["exit"]["m6_1_bank_contract_complete"] is True
    assert report["exit"]["production_bank_complete"] is False
    assert all(report["checks"].values())
    assert report["audit"]["ready_for_m6_bank_generation"] is True
    assert report["audit"]["ready_for_production"] is False
    json.dumps(report, allow_nan=False)
