import json

from egs.rir_generation.phases.m6_bank.scripts import validate_m6_variant_release


def test_m6_4_variant_release_validator(tmp_path):
    report = validate_m6_variant_release.build_report(tmp_path)

    assert report["milestone"] == "M6.4"
    assert report["exit"]["passed"] is True
    assert report["exit"]["m6_4_distribution_and_variant_release_complete"] is True
    assert report["exit"]["measured_and_mixed_recipes_complete"] is False
    assert report["exit"]["production_bank_complete"] is False
    assert all(report["checks"].values())
    assert report["release"]["recipe_statuses"]["synthetic_calibrated"] == "ready"
    assert report["release"]["recipe_statuses"]["real_native"] == "blocked"
    json.dumps(report, allow_nan=False)
