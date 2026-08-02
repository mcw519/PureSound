import json

from egs.rir_generation.phases.m5_calibration.scripts.validate_m5_measurement_contract import (
    audit_legacy_bank_metadata,
    build_report,
)
from puresound.audio.rir.calibration.measured_campaign import RIRMeasurementCampaign


def test_m5_1_implementation_exit_is_separate_from_measurement_readiness(tmp_path):
    report, template = build_report(
        {
            "train_view": tmp_path / "missing-train",
            "heldout_view": tmp_path / "missing-heldout",
        },
        sample_limit=2,
    )

    assert report["implementation_exit"]["passed"] is True
    assert report["controlled_measurement_exit"]["passed"] is False
    assert report["next_stage"]["milestone"] == "M5.2"
    assert template.provenance["template_only"] is True
    assert RIRMeasurementCampaign.from_json(template.to_json()) == template
    json.dumps(report, allow_nan=False)


def test_legacy_source_channel_bank_is_not_promoted_to_controlled_evidence(tmp_path):
    items = tmp_path / "bank/items"
    items.mkdir(parents=True)
    (items / "item.json").write_text(
        json.dumps(
            {
                "scene": {
                    "origin": "real",
                    "rt60": 0.63,
                    "channel_map": [
                        {"channel": 0, "label": "near_0", "distance_m": 0.9},
                        {"channel": 1, "label": "far_0", "distance_m": 2.7},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    audit = audit_legacy_bank_metadata(tmp_path / "bank", sample_limit=4)

    assert audit["sampled_json_count"] == 1
    assert audit["ready_for_m5_inverse_calibration"] is False
    assert not any(audit["all_sampled_items_contain"].values())
