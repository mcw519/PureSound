import json
import sys

import numpy as np
import pytest
import soundfile as sf

from egs.rir_generation.tools.measured import real_rir_to_bank as converter

SAMPLE_RATE = 16000


def _write_rir(path, channels=1, seconds=0.2):
    path.parent.mkdir(parents=True, exist_ok=True)
    rir = np.zeros((int(SAMPLE_RATE * seconds), channels), dtype=np.float32)
    rir[50, :] = 1.0
    rir[200:, :] = 0.01
    sf.write(path, rir, SAMPLE_RATE, subtype="FLOAT")


def _read_manifest(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_slr28_scan_assigns_documented_distances_per_channel(tmp_path):
    _write_rir(tmp_path / "in" / "RVB2014_type1_rir_smallroom1_near_angla.wav", channels=2)
    _write_rir(tmp_path / "in" / "RVB2014_type1_rir_smallroom1_far_angla.wav", channels=2)
    manifest = tmp_path / "m.jsonl"

    converter.scan_slr28(tmp_path / "in", tmp_path / "stage", manifest)

    entries = _read_manifest(manifest)
    assert sorted((e["distance_m"], e["channel"]) for e in entries) == [
        (0.5, 0),
        (0.5, 1),
        (2.0, 0),
        (2.0, 1),
    ]
    assert {e["room_id"] for e in entries} == {"reverb_smallroom1"}


def test_ace_scan_skips_positions_missing_from_the_distance_table(tmp_path):
    _write_rir(tmp_path / "in" / "Office_1" / "1" / "Single_a_RIR.wav")
    _write_rir(tmp_path / "in" / "Office_1" / "9" / "Single_b_RIR.wav")
    manifest = tmp_path / "m.jsonl"

    converter.scan_ace(tmp_path / "in", tmp_path / "stage", manifest)

    entries = _read_manifest(manifest)
    assert [(e["room_id"], e["distance_m"]) for e in entries] == [("ace_Office_1", 1.16)]


def test_empty_diffrir_scan_warns_instead_of_failing(tmp_path, capsys):
    (tmp_path / "in").mkdir()
    manifest = tmp_path / "m.jsonl"

    converter.scan_diffrir(tmp_path / "in", tmp_path / "stage", manifest)

    assert "0 entries" in capsys.readouterr().err
    assert manifest.read_text() == ""


@pytest.mark.parametrize("scanner", ["scan_dechorate", "scan_brudex"])
def test_hdf5_corpora_name_the_missing_dependency(scanner, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "h5py", None)

    with pytest.raises(SystemExit, match="pip install h5py"):
        getattr(converter, scanner)(tmp_path, tmp_path / "stage", tmp_path / "m.jsonl")


def test_bank_assembly_is_deterministic_and_splits_by_distance(tmp_path):
    entries = []
    for index, distance in enumerate((0.3, 0.7, 1.5, 2.5, 3.5)):
        path = tmp_path / "rirs" / f"mic{index}.wav"
        _write_rir(path)
        entries.append(
            {
                "room_id": "room",
                "rir_path": str(path),
                "channel": 0,
                "distance_m": distance,
                "rt60": 0.4,
            }
        )
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")

    first = converter.manifest_to_bank(str(manifest), str(tmp_path / "a"), 1.0, SAMPLE_RATE, 3, seed=5)
    converter.manifest_to_bank(str(manifest), str(tmp_path / "b"), 1.0, SAMPLE_RATE, 3, seed=5)

    assert first["items_written"] == 3
    for index in range(3):
        sidecars = [
            json.loads((tmp_path / name / f"room_{index:04d}" / f"room_{index:04d}.json").read_text())
            for name in ("a", "b")
        ]
        assert sidecars[0] == sidecars[1]
        channel_map = sidecars[0]["scene"]["channel_map"]
        assert sidecars[0]["scene"]["origin"] == "real"
        for channel in channel_map:
            expected = "near" if channel["distance_m"] < 1.0 else "far"
            assert channel["label"].startswith(expected)
