"""Merging corpora into one pool: what it refuses, and what it reports.

The concatenation is trivial. The refusals are not -- a duplicate id across
corpora is a file the dataset would silently never draw, or two corpora's
speakers fused into one sampler class -- and the report exists because batch
share follows speaker count, which is easy to mis-estimate by hand.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from puresound.dataset.corpus import AudioRecord, read_metafile, write_metafile
from puresound.dataset.corpus.pool import cap_speakers, main, merge_records, screen_audio


def _records(prefix: str, speakers: int, per_speaker: int, seconds: float = 4.0):
    out = []
    for s in range(speakers):
        for u in range(per_speaker):
            out.append(AudioRecord(
                uttid=f"{prefix}_{s:02d}_{u:02d}", spkid=f"{prefix}_{s:02d}", gender="None",
                path=Path(f"/audio/{prefix}/{s}_{u}.wav"), length=int(seconds * 16000),
                sample_rate=16000, channels=1,
            ))
    return out


def _duplicate_uttid():
    a = _records("x", 1, 2)
    return a, [AudioRecord(**{**a[0].__dict__, "spkid": "y_00", "path": Path("/other.wav")})]


def _duplicate_spkid():
    # The worse of the two: two corpora's speakers fused into one sampler class.
    a = _records("p", 2, 2)
    return a, [AudioRecord(**{**r.__dict__, "uttid": r.uttid + "_dup"}) for r in _records("p", 1, 2)]


@pytest.mark.parametrize("sources, field", [(_duplicate_uttid, "uttid"), (_duplicate_spkid, "spkid")])
def test_an_id_shared_by_two_sources_is_refused(sources, field):
    a, b = sources()
    with pytest.raises(ValueError, match=field):
        merge_records([("first", a), ("second", b)])


def test_the_cap_drops_whole_speakers_deterministically():
    records = _records("c", 10, 3)
    kept = cap_speakers(records, 4, seed=7)
    assert len({r.spkid for r in kept}) == 4 and len(kept) == 12
    assert [r.uttid for r in kept] == [r.uttid for r in cap_speakers(records, 4, seed=7)]
    assert cap_speakers(records, 10, seed=7) == records
    assert cap_speakers(records, None, seed=7) == records


def _audio_records(tmp_path, prefix: str, kinds: list[str]) -> list[AudioRecord]:
    """One record per kind: "speech" (noise at speech level), "silent" (all zeros),
    "empty" (a header and no samples) or "missing" (no file at the path)."""
    rng = np.random.default_rng(0)
    out = []
    for i, kind in enumerate(kinds):
        path = tmp_path / f"{prefix}_{i}_{kind}.wav"
        samples = {"speech": rng.standard_normal(16000).astype("float32") * 0.1,
                   "silent": np.zeros(16000, "float32"), "empty": np.zeros(0, "float32")}.get(kind)
        if samples is not None:
            sf.write(path, samples, 16000)
        out.append(AudioRecord(uttid=f"{prefix}_{i}", spkid=f"{prefix}_{i % 2}", gender="None",
                               path=path, length=len(samples) if samples is not None else 16000,
                               sample_rate=16000, channels=1))
    return out


@pytest.mark.parametrize("jobs", [1, 2])
def test_screening_drops_silent_empty_and_unreadable_files(tmp_path, jobs):
    """A pool row whose audio is all zeros, empty, or unreadable is a row the
    dataset cannot train on -- and one the coverage sampler, which names the
    utterance to load, treats as an error rather than re-drawing."""
    records = _audio_records(tmp_path, "a", ["speech", "silent", "speech", "empty", "missing"])
    kept, silent, unreadable = screen_audio(records, jobs=jobs)
    assert [r.uttid for r in kept] == ["a_0", "a_2"]
    assert [r.uttid for r in silent] == ["a_1", "a_3"]
    assert [r.uttid for r in unreadable] == ["a_4"]


def test_the_cli_drops_unusable_audio_unless_told_not_to_read_it(tmp_path, capsys):
    one, two = tmp_path / "one_train.csv", tmp_path / "two_train.csv"
    write_metafile(one, _audio_records(tmp_path, "one", ["speech", "silent", "speech"]))
    write_metafile(two, _audio_records(tmp_path, "two", ["speech", "missing"]))
    out = tmp_path / "pool_train.csv"
    assert main(["merge", "--source", str(one), "--source", str(two), "--out", str(out), "--jobs", "1"]) == 0
    assert [r.uttid for r in read_metafile(out)] == ["one_0", "one_2", "two_0"]
    text = capsys.readouterr().out
    assert "one_train: dropped 1 silent" in text and "two_train: dropped 1 unreadable" in text

    assert main(["merge", "--source", str(one), "--source", str(two), "--out", str(out),
                 "--skip-audio-check"]) == 0
    assert len(read_metafile(out)) == 5


def test_the_cli_does_not_replace_an_existing_pool_when_all_audio_is_unusable(tmp_path, capsys):
    one, two = tmp_path / "one_train.csv", tmp_path / "two_train.csv"
    write_metafile(one, _audio_records(tmp_path, "one", ["silent", "empty"]))
    write_metafile(two, _audio_records(tmp_path, "two", ["missing"]))
    out = tmp_path / "pool_train.csv"
    original = "existing pool\n"
    out.write_text(original)

    assert main(["merge", "--source", str(one), "--source", str(two), "--out", str(out),
                 "--jobs", "1"]) == 1
    assert out.read_text() == original
    assert "No usable audio remains" in capsys.readouterr().err


def test_the_cli_writes_the_pool_and_reports_the_share_by_speaker_count(tmp_path, capsys):
    """10 speakers against 90 is 10% of the batches however many hours the 10
    hold, and that is what the printed share must say."""
    big, small = tmp_path / "big_train.csv", tmp_path / "small_train.csv"
    write_metafile(big, _records("big", 90, 2, seconds=1.0))
    write_metafile(small, _records("small", 10, 20, seconds=10.0))
    out = tmp_path / "pool" / "pool_train.csv"
    # The records point at no real audio; this test is about the report.
    assert main(["merge", "--source", str(big), "--source", str(small), "--out", str(out),
                 "--skip-audio-check"]) == 0
    rows = read_metafile(out)
    assert len(rows) == 90 * 2 + 10 * 20
    assert {r.spkid for r in rows} == {f"big_{s:02d}" for s in range(90)} | {f"small_{s:02d}" for s in range(10)}
    text = capsys.readouterr().out
    # small holds most of the hours (2000 s vs 180 s) and 10% of the speakers.
    assert "small_train" in text and "10.0% of batches" in text
    assert "big_train" in text and "90.0% of batches" in text


def test_the_cli_refuses_a_single_source_and_a_malformed_cap(tmp_path, capsys):
    one = tmp_path / "one_train.csv"
    write_metafile(one, _records("one", 2, 2))
    assert main(["merge", "--source", str(one), "--out", str(tmp_path / "o.csv")]) == 1
    two = tmp_path / "two_train.csv"
    write_metafile(two, _records("two", 2, 2))
    assert main(["merge", "--source", str(one), "--source", str(two),
                 "--out", str(tmp_path / "o.csv"), "--max-speakers", "nonsense"]) == 1
    assert "STEM=N" in capsys.readouterr().err


def test_listed_speakers_are_left_out_of_every_source(tmp_path):
    a, b = tmp_path / "a_train.csv", tmp_path / "b_train.csv"
    write_metafile(a, _records("a", 3, 2))
    write_metafile(b, _records("b", 3, 2))
    listing = tmp_path / "exclude.txt"
    listing.write_text("a_01\nb_00\n")
    out = tmp_path / "pool.csv"
    assert main(["merge", "--source", str(a), "--source", str(b), "--out", str(out),
                 "--exclude-speakers", str(listing), "--skip-audio-check"]) == 0
    assert {r.spkid for r in read_metafile(out)} == {"a_00", "a_02", "b_01", "b_02"}
