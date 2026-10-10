"""Converting a corpus once, resumably, and rebuilding records from what was written."""

from pathlib import Path

import pytest

from puresound.dataset.corpus import (
    audio_info,
    build_resampled_tree,
    mirrored_path,
    scan_folder,
)
from puresound.dataset.corpus.resample import default_resample_root


@pytest.fixture
def corpus_48k(tmp_path, write_tone_wav):
    root = tmp_path / "clean_fullband"
    for speaker in ("spk0", "spk1"):
        for index in range(2):
            write_tone_wav(root / speaker / f"{speaker}_{index}.wav", sample_rate=48000)
    return root


def _build(records, source_root, dest, rate=16000, **kwargs):
    return build_resampled_tree(
        records, source_root=source_root, dest_root=dest, target_sample_rate=rate,
        **{"jobs": 1, **kwargs},
    )


def test_conversion_mirrors_the_layout_and_records_what_was_written(corpus_48k, tmp_path):
    records = scan_folder(corpus_48k, speaker_id_strategy="parent", tagger=lambda p: {"k": "v"})
    converted, report = _build(records, corpus_48k, tmp_path / "clean_16k", jobs=2)

    assert report.converted == 4 and report.reused == 0 and not report.failed
    assert {record.path.parent.name for record in converted} == {"spk0", "spk1"}
    # Identity survives; the length comes from the written file, not from scaling.
    assert [record.uttid for record in converted] == [record.uttid for record in records]
    assert [record.spkid for record in converted] == [record.spkid for record in records]
    assert all(record.tags == {"k": "v"} for record in converted)
    for record in converted:
        sample_rate, samples, _, _ = audio_info(record.path)
        assert (record.sample_rate, record.length) == (16000, samples)
        assert sample_rate == 16000


@pytest.mark.parametrize(
    "first_rate, overwrite, expected",
    [(16000, False, (0, 4)), (16000, True, (4, 0)), (32000, False, (4, 0))],
    ids=["second-run-reuses", "overwrite-reconverts", "stale-rate-reconverts"],
)
def test_a_second_run_reuses_only_what_is_already_at_the_target_rate(
    corpus_48k, tmp_path, first_rate, overwrite, expected
):
    records = scan_folder(corpus_48k, speaker_id_strategy="parent")
    dest = tmp_path / "out"
    _build(records, corpus_48k, dest, rate=first_rate)

    converted, report = _build(records, corpus_48k, dest, overwrite=overwrite)
    assert (report.converted, report.reused) == expected
    assert {record.sample_rate for record in converted} == {16000}


def test_an_unreadable_source_is_reported_and_dropped(corpus_48k, tmp_path):
    records = scan_folder(corpus_48k, speaker_id_strategy="parent")
    records[0].path.write_bytes(b"not audio")

    converted, report = _build(records, corpus_48k, tmp_path / "out")

    assert len(report.failed) == 1
    assert len(converted) == 3
    # A dropped file must not leave a metafile row pointing at a missing file.
    assert all(record.path.is_file() for record in converted)
    assert report.total == 4
    assert "failed" in report.summary()


def test_destination_paths_mirror_the_source_and_sit_beside_the_corpus(tmp_path):
    source = tmp_path / "src" / "a" / "b.flac"
    assert mirrored_path(source, tmp_path / "src", tmp_path / "dst") == tmp_path / "dst" / "a" / "b.wav"
    for part in ("clean_fullband", "noise_fullband"):
        assert default_resample_root(Path(f"/data/dns-5/datasets_fullband/{part}"), 16000) == Path(
            f"/data/dns-5/datasets_fullband_16k/{part}"
        )


def test_two_recordings_that_mirror_to_one_file_are_refused(tmp_path, write_tone_wav):
    """``a.wav`` and ``a.flac`` land on the same converted file; the second would
    be taken as already converted and its record would point at the first's audio."""
    root = tmp_path / "corpus"
    write_tone_wav(root / "a.wav", sample_rate=48000)
    write_tone_wav(root / "a.flac", sample_rate=48000, duration=0.5)
    records = scan_folder(root, speaker_id_strategy="per-file")

    with pytest.raises(ValueError, match="both mirror to"):
        _build(records, root, tmp_path / "out")

    assert not (tmp_path / "out").exists()
