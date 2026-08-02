import json
import sys

import numpy as np
import soundfile as sf

from egs.rir_generation.phases.m4_spatial_late_field.scripts import validate_multiband_late_field


def _write_bank(root, *, dense: bool) -> None:
    sample_rate = 8000
    for item_index in range(3):
        item_dir = root / f"room_{item_index:06d}"
        item_dir.mkdir(parents=True)
        item_id = f"room_{item_index:06d}_000000"
        response = np.zeros((sample_rate, 1), dtype=np.float32)
        response[10, 0] = 1.0
        if dense:
            tail = np.random.default_rng(200 + item_index).normal(
                0.0,
                0.02,
                sample_rate - 12,
            )
            response[12:, 0] += tail.astype(np.float32)
        else:
            response[400::1600, 0] = 0.1
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
                                "label": "source_0",
                                "distance_m": 1.5,
                            }
                        ],
                    }
                }
            ),
            encoding="utf-8",
        )


def test_multiband_report_freezes_mono_and_spatial_contracts(tmp_path):
    measured = tmp_path / "measured"
    sparse = tmp_path / "sparse"
    _write_bank(measured, dense=True)
    _write_bank(sparse, dense=False)

    report = validate_multiband_late_field.build_report(
        [("measured", measured), ("sparse", sparse)],
        per_bank=3,
        seed=8,
        reference_tag="measured",
        centers_hz=(500.0, 1000.0),
        minimum_decay_r_squared=0.9,
        base_window_ms=20.0,
        minimum_window_cycles=4.0,
        hop_ms=2.0,
        threshold=0.9,
        minimum_sustain_ms=10.0,
    )

    measured_summary = report["banks"]["measured"]["summary"]["overall"]
    sparse_summary = report["banks"]["sparse"]["summary"]["overall"]
    comparison = report["reference_comparisons"]["sparse"]
    assert report["milestone"] == "M4.2"
    assert report["metric_policy"] == "puresound.multiband_late_field.v1"
    assert set(measured_summary["bands"]) == {"500", "1000"}
    assert measured_summary["bands"]["500"]["mixing_time_coverage"] == 1.0
    assert sparse_summary["bands"]["500"]["mixing_time_coverage"] == 0.0
    assert comparison["evaluated_checks"] > 0
    assert comparison["all_evaluated_checks_passed"] is False
    assert report["spatial_contract"]["evaluable_from_these_banks"] is False
    assert report["spatial_contract"]["binaural_policy"] == (
        "puresound.binaural_iacc.v1"
    )
    json.dumps(report, allow_nan=False)


def test_multiband_cli_writes_strict_json(tmp_path, monkeypatch):
    bank = tmp_path / "bank"
    _write_bank(bank, dense=True)
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_multiband_late_field.py",
            f"measured={bank}",
            "--reference-tag",
            "measured",
            "--per-bank",
            "2",
            "--octave-centers-hz",
            "500",
            "1000",
            "--json-output",
            str(output),
        ],
    )

    assert validate_multiband_late_field.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["banks"]["measured"]["sampled_channels"] == 2
