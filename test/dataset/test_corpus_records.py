"""The metafile stays seven columns; everything else a corpus knows rides in tags."""

from pathlib import Path

import pytest

from puresound.dataset.corpus import (
    METAFILE_HEADER,
    AudioRecord,
    read_inventory,
    read_metafile,
    tag_histogram,
    write_inventory,
    write_metafile,
)


def _record(index: int, **overrides) -> AudioRecord:
    fields = dict(
        uttid=f"utt{index}",
        spkid=f"spk{index % 2}",
        gender="other",
        path=Path(f"/corpus/utt{index}.wav"),
        length=16000,
        sample_rate=16000,
        channels=1,
    )
    fields.update(overrides)
    return AudioRecord(**fields)


def test_metafile_round_trip_preserves_every_column_and_no_tag(tmp_path):
    records = [_record(index) for index in range(4)] + [_record(4, tags={"category": "fan"})]
    path = tmp_path / "train.csv"

    assert write_metafile(path, records) == 5
    body = path.read_text(encoding="utf-8")
    assert body.splitlines()[0] == METAFILE_HEADER
    assert "fan" not in body
    assert len(body.splitlines()[-1].split(",")) == 7

    loaded = read_metafile(path)
    assert [item.uttid for item in loaded] == [item.uttid for item in records]
    assert [item.path for item in loaded] == [item.path for item in records]
    assert all(item.sample_rate == 16000 and item.channels == 1 for item in loaded)
    assert loaded[0].duration == 1.0
    assert _record(0, sample_rate=0).duration == 0.0  # a missing rate is not a crash


def test_inventory_round_trip_keeps_tags(tmp_path):
    records = [_record(0, tags={"category": "fan", "device": "hplaptop"}), _record(1)]
    path = tmp_path / "inventory.jsonl"

    assert write_inventory(path, records) == 2
    loaded = list(read_inventory(path))
    assert loaded[0].tags == {"category": "fan", "device": "hplaptop"}
    assert loaded[1].tags == {}
    assert loaded[0].path == records[0].path


def test_a_row_that_would_shift_the_columns_is_refused_both_ways(tmp_path):
    short = tmp_path / "bad.csv"
    short.write_text(f"{METAFILE_HEADER}\nutt0, spk0, other, /a.wav\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 7 columns"):
        read_metafile(short)

    comma = _record(0, path=Path("/a/alger_jr.,_64kb.flac"))
    with pytest.raises(ValueError, match="comma"):
        write_metafile(tmp_path / "m.csv", [comma])


def test_histogram_counts_untagged_records_as_unknown():
    records = [
        _record(0, tags={"category": "fan"}),
        _record(1, tags={"category": "fan"}),
        _record(2, tags={"category": "typing"}),
        _record(3),
    ]

    histogram = tag_histogram(records, "category")
    assert histogram == {"fan": 2, "typing": 1, "unknown": 1}
    assert sum(histogram.values()) == len(records)
