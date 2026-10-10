"""Noise corpora: only what a commercial model may train on, and no voices."""
import csv
import json

from puresound.dataset.corpus.noise_corpora import commercial_ok, fsd50k_records, needs_attribution


def test_licences_a_commercial_model_can_use():
    assert commercial_ok("http://creativecommons.org/publicdomain/zero/1.0/")
    assert commercial_ok("http://creativecommons.org/licenses/by/3.0/")
    assert not commercial_ok("http://creativecommons.org/licenses/by-nc/3.0/")
    assert not commercial_ok("http://creativecommons.org/licenses/sampling+/1.0/")
    assert not commercial_ok("")


def test_fsd50k_drops_nc_clips_and_labelled_voices_but_keeps_crowds(tmp_path, write_tone_wav):
    (tmp_path / "FSD50K.metadata").mkdir()
    (tmp_path / "FSD50K.ground_truth").mkdir()
    by = "http://creativecommons.org/licenses/by/3.0/"
    clips = {
        "1": ("Subway_and_metro_and_underground,Vehicle", by),
        "2": ("Subway_and_metro_and_underground,Vehicle", "http://creativecommons.org/licenses/by-nc/3.0/"),
        "3": ("Male_speech_and_man_speaking,Speech,Human_voice", by),
        "4": ("Chatter,Crowd,Human_group_actions", by),
    }
    for split in ("dev", "eval"):
        info = {k: {"license": lic, "uploader": "u", "title": "t"} for k, (_, lic) in clips.items()} if split == "dev" else {}
        (tmp_path / "FSD50K.metadata" / f"{split}_clips_info_FSD50K.json").write_text(json.dumps(info))
        with open(tmp_path / "FSD50K.ground_truth" / f"{split}.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["fname", "labels", "mids", "split"])
            if split == "dev":
                for fname, (labels, _) in clips.items():
                    writer.writerow([fname, labels, "", "train"])
                    write_tone_wav(tmp_path / "FSD50K.dev_audio" / f"{fname}.wav")

    kept, why = fsd50k_records(tmp_path)

    assert sorted(r.tags["freesound_id"] for r in kept) == ["1", "4"]
    assert why["licence"] == 1 and why["voice label"] == 1
    assert all(needs_attribution(r.tags) for r in kept)
