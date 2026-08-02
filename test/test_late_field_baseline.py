import json
import random
import sys

import numpy as np
import soundfile as sf

from egs.rir_generation.phases.m4_spatial_late_field.scripts import validate_late_field_baseline


def _write_late_field_bank(root, *, dense: bool) -> None:
    sample_rate = 8000
    for item_index in range(3):
        item_dir = root / f"room_{item_index:06d}"
        item_dir.mkdir(parents=True)
        item_id = f"room_{item_index:06d}_000000"
        response = np.zeros((sample_rate, 2), dtype=np.float32)
        for channel in range(2):
            response[10 + channel, channel] = 1.0
            if dense:
                response[12 + channel :, channel] += (
                    np.random.default_rng(100 + 10 * item_index + channel)
                    .normal(0.0, 0.02, sample_rate - 12 - channel)
                    .astype(np.float32)
                )
            else:
                response[80 + channel :: 160, channel] = 0.1
        sf.write(item_dir / f"{item_id}.wav", response, sample_rate, subtype="FLOAT")
        (item_dir / f"{item_id}.json").write_text(
            json.dumps(
                {
                    "scene": {
                        "room_id": f"room_{item_index:06d}",
                        "origin": "real" if dense else "synthetic",
                        "channel_map": [
                            {
                                "channel": 0,
                                "label": "near_0",
                                "distance_m": 0.8,
                            },
                            {
                                "channel": 1,
                                "label": "far_0",
                                "distance_m": 2.4,
                            },
                        ],
                    }
                }
            ),
            encoding="utf-8",
        )


def test_late_field_report_separates_dense_and_sparse_banks(tmp_path):
    measured = tmp_path / "measured"
    sparse = tmp_path / "sparse"
    _write_late_field_bank(measured, dense=True)
    _write_late_field_bank(sparse, dense=False)

    report = validate_late_field_baseline.build_report(
        [("measured", measured), ("sparse", sparse)],
        per_bank=3,
        seed=4,
        reference_tag="measured",
        window_ms=20.0,
        hop_ms=1.0,
        threshold=0.9,
        minimum_sustain_ms=10.0,
    )

    measured_summary = report["banks"]["measured"]["summary"]["overall"]
    sparse_summary = report["banks"]["sparse"]["summary"]["overall"]
    comparison = report["reference_comparisons"]["sparse"]
    assert report["metric_policy"] == "puresound.abel_huang_echo_density.v1"
    assert measured_summary["mixing_time_coverage"] == 1.0
    assert sparse_summary["mixing_time_coverage"] == 0.0
    assert measured_summary["late_median_normalized_density"]["median"] > 0.9
    assert sparse_summary["late_median_normalized_density"]["median"] < 0.1
    assert comparison["all_evaluated_checks_passed"] is False
    json.dumps(report, allow_nan=False)


def test_late_field_sampling_is_seed_deterministic(tmp_path):
    bank = tmp_path / "bank"
    _write_late_field_bank(bank, dense=True)

    first, candidates = validate_late_field_baseline.sample_bank(
        bank,
        count=2,
        rng=random.Random("fixed"),
        window_ms=20.0,
        hop_ms=2.0,
        threshold=0.9,
        minimum_sustain_ms=0.0,
    )
    second, second_candidates = validate_late_field_baseline.sample_bank(
        bank,
        count=2,
        rng=random.Random("fixed"),
        window_ms=20.0,
        hop_ms=2.0,
        threshold=0.9,
        minimum_sustain_ms=0.0,
    )

    assert candidates == second_candidates == 3
    assert first == second


def test_late_field_cli_writes_strict_json(tmp_path, monkeypatch):
    bank = tmp_path / "bank"
    _write_late_field_bank(bank, dense=True)
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_late_field_baseline.py",
            f"measured={bank}",
            "--reference-tag",
            "measured",
            "--per-bank",
            "2",
            "--json-output",
            str(output),
        ],
    )

    assert validate_late_field_baseline.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["banks"]["measured"]["sampled_channels"] == 2
