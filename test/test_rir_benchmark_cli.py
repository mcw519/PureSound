import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "egs" / "rir_generation" / "compare_bank_acoustics.py"
MODAL_SCRIPT = (
    REPO_ROOT / "egs" / "rir_generation" / "compare_modal_acoustics.py"
)


def _write_fixture_bank(root: Path, sample_rate: int = 16000) -> None:
    room = root / "room_000000"
    room.mkdir(parents=True)
    rir = torch.zeros(2, sample_rate)
    time_s = torch.arange(sample_rate, dtype=torch.float64) / sample_rate
    tail = torch.exp(-math.log(1000.0) * time_s / 0.4).float()
    for channel, direct in enumerate((24, 96)):
        available = sample_rate - direct
        rir[channel, direct:] = tail[:available]
    torchaudio.save(
        str(room / "room_000000_000000.wav"),
        rir,
        sample_rate,
        encoding="PCM_F",
    )
    metadata = {
        "scene": {
            "origin": "test",
            "rt60": 0.4,
            "channel_map": [
                {"channel": 0, "label": "near_0", "distance_m": 0.5},
                {"channel": 1, "label": "far_0", "distance_m": 2.5},
            ],
        }
    }
    (room / "room_000000_000000.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def test_compare_bank_acoustics_writes_strict_json(tmp_path):
    bank = tmp_path / "bank"
    output = tmp_path / "report.json"
    _write_fixture_bank(bank)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            f"fixture={bank}",
            "--per-bank",
            "4",
            "--seed",
            "7",
            "--octave-bands",
            "--json-output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["seed"] == 7
    assert report["min_decay_fit_r_squared"] == 0.9
    assert report["banks"]["fixture"]["metadata_candidates"] == 1
    assert report["banks"]["fixture"]["sampled_channels"] == 1
    assert sum(
        bucket["channels"]
        for bucket in report["banks"]["fixture"]["summary"].values()
    ) == 1
    only_bucket = next(iter(report["banks"]["fixture"]["summary"].values()))
    assert only_bucket["valid_decay_fits"]["t30"] == 1
    assert only_bucket["valid_decay_fraction"]["t30"] == 1.0
    assert "noise_floor_correction_fraction" in only_bucket
    assert "decay_dynamic_range_db" in only_bucket["median"]
    assert report["banks"]["fixture"]["channels"][0]["metrics"]["octave_bands"]
    assert "wrote" in completed.stdout


def test_compare_bank_acoustics_is_deterministic(tmp_path):
    bank = tmp_path / "bank"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_fixture_bank(bank)
    base_command = [
        sys.executable,
        str(SCRIPT),
        f"fixture={bank}",
        "--per-bank",
        "1",
        "--seed",
        "19",
    ]

    subprocess.run(
        [*base_command, "--json-output", str(first)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [*base_command, "--json-output", str(second)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(first.read_text()) == json.loads(second.read_text())


def test_compare_modal_acoustics_reports_peak_and_q_distributions(tmp_path):
    bank = tmp_path / "bank"
    room = bank / "room_000000"
    room.mkdir(parents=True)
    sample_rate = 4000
    time_s = torch.arange(2 * sample_rate, dtype=torch.float64) / sample_rate
    rir = (
        torch.exp(-10.0 * time_s)
        * torch.sin(2.0 * math.pi * 80.0 * time_s)
    ).float()[None, :]
    torchaudio.save(
        str(room / "room_000000_000000.wav"),
        rir,
        sample_rate,
        encoding="PCM_F",
    )
    (room / "room_000000_000000.json").write_text(
        json.dumps(
            {
                "scene": {
                    "channel_map": [
                        {
                            "channel": 0,
                            "label": "near_0",
                            "distance_m": 0.5,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "modal.json"

    subprocess.run(
        [
            sys.executable,
            str(MODAL_SCRIPT),
            f"fixture={bank}",
            "--per-bank",
            "1",
            "--min-frequency-hz",
            "50",
            "--max-frequency-hz",
            "110",
            "--post-direct-delay-ms",
            "0",
            "--analysis-duration-s",
            "1.9",
            "--min-prominence-db",
            "10",
            "--reference-tag",
            "fixture",
            "--json-output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text())
    summary = report["banks"]["fixture"]["summary"]
    assert report["analysis"]["q_definition"].startswith("peak_frequency")
    assert summary["total_peaks"] == 1
    assert summary["resolved_q_count"] == 1
    assert summary["q_factor"]["median"] == pytest.approx(25.13, rel=0.03)
    assert (
        report["banks"]["fixture"]["distance_to_reference"]["q_factor"][
            "wasserstein"
        ]
        == 0.0
    )
