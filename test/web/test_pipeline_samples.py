"""The pipeline inspector's bundled samples: every file the index names exists,
noise is 16 kHz mono, and the rooms load as a room bank with geometry."""

import json
from pathlib import Path

import soundfile as sf

import puresound.web
from puresound.audio.rir.bank.loader import PreGeneratedRoomBank

ROOT = Path(puresound.web.__file__).with_name("static") / "samples" / "pipeline"


def test_the_pipeline_samples_are_consistent_and_load_as_a_room_bank():
    index = json.loads((ROOT / "index.json").read_text())
    assert [talker["id"] for talker in index["talkers"]] == ["reader-61", "reader-1272"]
    for talker in index["talkers"]:
        for file in talker["files"]:
            assert (ROOT / file).resolve().is_file(), file
    assert {noise["id"] for noise in index["noises"]} == {"pink", "rumble", "hum", "fan", "babble", "clatter"}
    for noise in index["noises"]:
        info = sf.info(ROOT / noise["file"])
        assert (info.samplerate, info.channels) == (16000, 1)
    assert len(index["rooms"]) == 4
    assert len(PreGeneratedRoomBank(folder=str(ROOT / "rooms"))) == 4
    rt60s = [room["rt60"] for room in index["rooms"]]
    assert rt60s == sorted(rt60s) and rt60s[-1] - rt60s[0] > 0.3
    for room in index["rooms"]:
        scene = json.loads((ROOT / room["file"]).with_suffix(".json").read_text())["scene"]
        assert scene["room_dim"] and scene["mic_pos"]
        assert all("source_pos" in entry and "distance_m" in entry for entry in scene["channel_map"])
    assert "CC BY 4.0" in (ROOT / "README.md").read_text()
    babble = next(noise for noise in index["noises"] if noise["id"] == "babble")
    assert "talker samples" in babble["description"]  # it is those two voices, reversed
