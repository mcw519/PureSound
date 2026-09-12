import json
import sys

import numpy as np
import soundfile as sf

from egs.rir_generation.tools.audition import build_audition_bank


def _write_item(root, index, *, prearrival=False):
    sample_rate = 1000
    sound_speed = 100.0
    distances = (0.1, 0.2, 1.0, 1.2, 1.4)
    labels = ("near_0", "near_1", "far_0", "far_1", "far_2")
    item_dir = root / f"room_{index:06d}"
    item_dir.mkdir(parents=True)
    sample_id = f"room_{index:06d}_000000"
    wav_path = item_dir / f"{sample_id}.wav"
    metadata_path = item_dir / f"{sample_id}.json"
    rir = np.zeros((sample_rate, 5), dtype=np.float32)
    channel_map = []
    realized = []

    for channel, (label, distance) in enumerate(zip(labels, distances)):
        direct = int(np.floor(distance / sound_speed * sample_rate))
        time = np.arange(sample_rate - direct)
        rir[direct:, channel] = (
            0.1
            / (channel + 1)
            * np.exp(-time / 150.0)
            * np.cos(0.13 * time)
        )
        if prearrival and channel == 2:
            rir[0, channel] = 1e-3
        channel_map.append(
            {
                "channel": channel,
                "label": label,
                "distance_m": distance,
            }
        )
        drr = 8.0 if label.startswith("near") else -2.0
        realized.append(
            {
                "channel": channel,
                "label": label,
                "distance_m": distance,
                "metrics": {
                    "drr_db": drr,
                    "edt": {"rt60_s": 0.4, "r_squared": 0.99},
                    "t20": {"rt60_s": 0.5, "r_squared": 0.99},
                    "t30": {"rt60_s": 0.6, "r_squared": 0.99},
                },
            }
        )
    sf.write(wav_path, rir, sample_rate, subtype="FLOAT")
    metadata_path.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "room_id": f"room_{index:06d}",
                "scene": {
                    "schema_version": "rir_scene.v2",
                    "room_type": "office",
                    "rt60": 0.4 + index * 0.1,
                    "environment": {
                        "sound_speed_m_s": sound_speed,
                    },
                    "channel_map": channel_map,
                },
                "realized_acoustics": {"channels": realized},
            }
        ),
        encoding="utf-8",
    )


def test_audition_builder_validates_bank_and_writes_previews(tmp_path, monkeypatch):
    bank = tmp_path / "bank"
    _write_item(bank, 0)
    _write_item(bank, 1)
    dry = tmp_path / "dry.wav"
    time = np.arange(500, dtype=np.float64) / 1000.0
    sf.write(dry, 0.2 * np.sin(2.0 * np.pi * 80.0 * time), 1000)
    output = tmp_path / "audition"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_audition_bank.py",
            "--bank",
            str(bank),
            "--output-dir",
            str(output),
            "--dry-wav",
            str(dry),
            "--sample-rate",
            "1000",
            "--min-items",
            "2",
            "--num-previews",
            "1",
        ],
    )

    assert build_audition_bank.main() == 0

    report = json.loads((output / "report.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert report["quality_passed"] is True
    assert report["summary"]["prearrival_peak_abs"]["maximum"] == 0.0
    assert report["summary"]["median_near_far_drr_gap_db"] == 10.0
    assert len(manifest["previews"]) == 1
    assert len(list((output / "previews").glob("*.wav"))) == 2
    assert len(list((output / "items").glob("*.wav"))) == 2
    assert all(path.is_symlink() for path in (output / "items").glob("*"))


def test_audition_inspection_rejects_prearrival_energy(tmp_path):
    bank = tmp_path / "bank"
    _write_item(bank, 0, prearrival=True)

    rows, summary = build_audition_bank.inspect_bank(
        bank,
        expected_sample_rate=1000,
        expected_channels=5,
        maximum_rir_peak=1.0,
        minimum_tail_energy_fraction=1e-5,
        minimum_decay_r_squared=0.9,
    )

    assert "prearrival_energy" in rows[0]["failures"]
    assert summary["failure_counts"]["prearrival_energy"] == 1
