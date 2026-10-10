"""Writing a benchmark that already exists on disk into the scored-set format."""

import json

import pytest

from puresound.dataset.corpus.paired import read_transcript, write_paired_set


@pytest.fixture
def corpus(tmp_path, write_tone_wav):
    noisy, clean, text = (tmp_path / n for n in ("noisy", "clean", "txt"))
    text.mkdir()
    for index in range(3):
        write_tone_wav(noisy / f"u{index}.wav", sample_rate=48000, duration=0.5, freq=200 + index)
        write_tone_wav(clean / f"u{index}.wav", sample_rate=48000, duration=0.5, freq=200 + index)
        (text / f"u{index}.txt").write_text(f"utterance  {index}\n", encoding="utf-8")
    write_tone_wav(noisy / "orphan.wav", sample_rate=48000, duration=0.5)  # no clean side
    return noisy, clean, text, tmp_path / "out"


def _write(corpus, **kwargs):
    noisy, clean, _, out = corpus
    report = write_paired_set(noisy, clean, out, sample_rate=16000, **kwargs)
    rows = [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return report, rows, out


def test_paired_files_are_written_at_the_requested_rate_and_an_orphan_is_named(corpus):
    """A set quietly smaller than the corpus has numbers nobody else's match."""
    report, rows, out = _write(corpus, extra_tags=lambda item_id: {"split": "test"})
    assert [r["id"] for r in rows] == ["u0", "u1", "u2"]
    assert report.written == 3
    assert report.skipped_unpaired == ["orphan.wav"]
    assert "unpaired" in report.summary()
    for row in rows:
        for key in ("id", "mix", "clean", "sample_rate", "samples", "snr_db", "snr_band", "source"):
            assert key in row, key
        assert row["sample_rate"] == 16000 and row["split"] == "test"
        assert (out / row["mix"]).is_file() and (out / row["clean"]).is_file()


def test_transcripts_are_optional_normalised_and_a_missing_one_is_reported(corpus, tmp_path):
    _, without, _ = _write(corpus)
    assert "transcript" not in without[0]

    report, with_text, _ = _write(corpus, transcript_dir=corpus[2])
    assert with_text[0]["transcript"] == "utterance 0"
    assert report.missing_transcript == []

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "u0.txt").write_text("only this one", encoding="utf-8")
    report, rows, _ = _write(corpus, transcript_dir=partial)
    assert report.missing_transcript == ["u1", "u2"]
    assert "transcript" not in rows[1]

    (tmp_path / "a.txt").write_text("  hello \n  world  \n", encoding="utf-8")
    assert read_transcript(tmp_path / "a.txt") == "hello world"


def test_no_overlap_is_an_error_not_an_empty_set(tmp_path, write_tone_wav):
    left, right = tmp_path / "a", tmp_path / "b"
    write_tone_wav(left / "x.wav")
    write_tone_wav(right / "y.wav")
    with pytest.raises(ValueError, match="No filename matches"):
        write_paired_set(left, right, tmp_path / "out")


def test_provenance_records_the_audio_origin_and_is_rewritten_every_time(corpus):
    """An imported set's audio is fixed, so what a scoring run needs to know is
    its origin and what was done to it -- not a synthesis commit it has none of."""
    _, _, out = _write(corpus, transcript_dir=corpus[2], source="demo")
    record = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert record["kind"] == "imported"
    assert record["source"] == "demo"
    assert record["sample_rate"] == 16000
    assert record["items"] == 3
    assert record["skipped_unpaired"] == 1
    assert len(record["manifest_sha256"]) == 64

    # A stale provenance cannot describe new audio.
    (out / "provenance.json").write_text('{"kind": "stale"}', encoding="utf-8")
    _write(corpus, limit=2)
    record = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert record["kind"] == "imported" and record["items"] == 2
